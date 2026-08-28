from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .ellipsoid import ellipsoid_contact, shape_from_rotation_scale


def random_rotations(count: int, *, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn((count, 3, 3), device="cuda", generator=generator)
    q, _ = torch.linalg.qr(matrix)
    determinant = torch.linalg.det(q)
    q[:, :, -1] *= torch.where(determinant < 0, -1.0, 1.0).unsqueeze(-1)
    return q


def fibonacci_sphere(count: int) -> torch.Tensor:
    index = torch.arange(count, device="cuda", dtype=torch.float32) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0))
    phi = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack((radius * torch.cos(phi), radius * torch.sin(phi), z), dim=-1)


@torch.no_grad()
def dense_reference(
    center_a: torch.Tensor,
    shape_a: torch.Tensor,
    center_b: torch.Tensor,
    shape_b: torch.Tensor,
    directions: torch.Tensor,
    *,
    case_chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    gaps = []
    normals = []
    for start in range(0, center_a.shape[0], case_chunk):
        stop = min(start + case_chunk, center_a.shape[0])
        delta = center_b[start:stop] - center_a[start:stop]
        projection = torch.einsum("nd,cd->cn", directions, delta)
        radius_a = torch.sqrt(
            torch.einsum("nd,cde,ne->cn", directions, shape_a[start:stop], directions)
            .clamp_min(1e-12)
        )
        radius_b = torch.sqrt(
            torch.einsum("nd,cde,ne->cn", directions, shape_b[start:stop], directions)
            .clamp_min(1e-12)
        )
        objective = projection - radius_a - radius_b
        best = objective.argmax(dim=-1)
        gaps.append(objective.gather(1, best[:, None]).squeeze(1))
        normals.append(directions[best])
    normal = torch.cat(normals)
    delta = center_b - center_a
    scale = (
        torch.sqrt(torch.diagonal(shape_a, dim1=-2, dim2=-1).sum(dim=-1))
        + torch.sqrt(torch.diagonal(shape_b, dim1=-2, dim2=-1).sum(dim=-1))
    )[:, None].clamp_min(1e-8)
    # Refine the best dense direction with small, decaying Riemannian steps.
    # The dense global search and this local refinement form a substantially
    # stronger reference than either component on its own.
    for iteration in range(192):
        qa_n = torch.einsum("bij,bj->bi", shape_a, normal)
        qb_n = torch.einsum("bij,bj->bi", shape_b, normal)
        radius_a = torch.sqrt((normal * qa_n).sum(dim=-1, keepdim=True).clamp_min(1e-12))
        radius_b = torch.sqrt((normal * qb_n).sum(dim=-1, keepdim=True).clamp_min(1e-12))
        gradient = delta - qa_n / radius_a - qb_n / radius_b
        tangent = gradient - normal * (normal * gradient).sum(dim=-1, keepdim=True)
        denominator = torch.maximum(scale, torch.linalg.vector_norm(tangent, dim=-1, keepdim=True))
        step = 0.12 * (1.0 - iteration / 192) + 0.01
        normal = torch.nn.functional.normalize(normal + step * tangent / denominator, dim=-1)
    qa_n = torch.einsum("bij,bj->bi", shape_a, normal)
    qb_n = torch.einsum("bij,bj->bi", shape_b, normal)
    gap = (
        (normal * delta).sum(dim=-1)
        - torch.sqrt((normal * qa_n).sum(dim=-1).clamp_min(1e-12))
        - torch.sqrt((normal * qb_n).sum(dim=-1).clamp_min(1e-12))
    )
    return gap, normal


def percentile(values: torch.Tensor, levels: list[float]) -> list[float]:
    return torch.quantile(values, torch.tensor(levels, device="cuda")).cpu().tolist()


def subset_metrics(mask: torch.Tensor, gap_error: torch.Tensor, sphere_error: torch.Tensor) -> dict[str, object]:
    selected_gap = gap_error[mask]
    selected_sphere = sphere_error[mask]
    if selected_gap.numel() == 0:
        return {"count": 0}
    return {
        "count": int(selected_gap.numel()),
        "ours_gap_error_m_p50_p95_p99": percentile(selected_gap, [0.5, 0.95, 0.99]),
        "sphere_gap_error_m_p50_p95_p99": percentile(selected_sphere, [0.5, 0.95, 0.99]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=4096)
    parser.add_argument("--directions", type=int, default=65536)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, default=Path("outputs/ellipsoid_benchmark.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The benchmark requires CUDA")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    rotations_a = random_rotations(args.cases, generator=generator)
    rotations_b = random_rotations(args.cases, generator=generator)
    log_min, log_max = math.log(0.025), math.log(0.25)
    scales_a = torch.exp(
        torch.empty((args.cases, 3), device="cuda").uniform_(log_min, log_max, generator=generator)
    )
    scales_b = torch.exp(
        torch.empty((args.cases, 3), device="cuda").uniform_(log_min, log_max, generator=generator)
    )
    shape_a = shape_from_rotation_scale(rotations_a, scales_a)
    shape_b = shape_from_rotation_scale(rotations_b, scales_b)
    center_a = torch.zeros((args.cases, 3), device="cuda")
    direction = torch.randn((args.cases, 3), device="cuda", generator=generator)
    direction = torch.nn.functional.normalize(direction, dim=-1)
    directional_extent = torch.sqrt(
        torch.einsum("bi,bij,bj->b", direction, shape_a, direction)
    ) + torch.sqrt(torch.einsum("bi,bij,bj->b", direction, shape_b, direction))
    distance_factor = torch.empty(args.cases, device="cuda").uniform_(0.25, 2.0, generator=generator)
    center_b = direction * (directional_extent * distance_factor)[:, None]

    reference_directions = fibonacci_sphere(args.directions)
    reference_gap, reference_normal = dense_reference(
        center_a, shape_a, center_b, shape_b, reference_directions
    )

    # Warm up kernels before measuring.
    for _ in range(3):
        ellipsoid_contact(center_a, shape_a, center_b, shape_b, iterations=args.iterations)
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    prediction = ellipsoid_contact(
        center_a, shape_a, center_b, shape_b, iterations=args.iterations
    )
    end.record()
    torch.cuda.synchronize()

    gap_error = (prediction.signed_gap - reference_gap).abs()
    cosine = (prediction.normal * reference_normal).sum(dim=-1).clamp(-1, 1)
    normal_error = torch.rad2deg(torch.acos(cosine))
    equivalent_a = scales_a.prod(dim=-1).pow(1 / 3)
    equivalent_b = scales_b.prod(dim=-1).pow(1 / 3)
    sphere_gap = torch.linalg.vector_norm(center_b - center_a, dim=-1) - equivalent_a - equivalent_b
    sphere_error = (sphere_gap - reference_gap).abs()
    classification = prediction.signed_gap > 0
    reference_classification = reference_gap > 0

    gradient_cases = min(args.cases, 128)
    differentiable_scale = scales_a[:gradient_cases].detach().clone().requires_grad_(True)
    differentiable_shape = shape_from_rotation_scale(
        rotations_a[:gradient_cases], differentiable_scale
    )
    differentiable_gap = ellipsoid_contact(
        center_a[:gradient_cases],
        differentiable_shape,
        center_b[:gradient_cases],
        shape_b[:gradient_cases],
        iterations=args.iterations,
    ).signed_gap
    scale_gradient = torch.autograd.grad(differentiable_gap.sum(), differentiable_scale)[0]
    finite_difference = torch.empty_like(scale_gradient)
    epsilon = 1e-4
    with torch.no_grad():
        for axis in range(3):
            plus = scales_a[:gradient_cases].clone()
            minus = scales_a[:gradient_cases].clone()
            plus[:, axis] += epsilon
            minus[:, axis] -= epsilon
            plus_gap = ellipsoid_contact(
                center_a[:gradient_cases],
                shape_from_rotation_scale(rotations_a[:gradient_cases], plus),
                center_b[:gradient_cases],
                shape_b[:gradient_cases],
                iterations=args.iterations,
            ).signed_gap
            minus_gap = ellipsoid_contact(
                center_a[:gradient_cases],
                shape_from_rotation_scale(rotations_a[:gradient_cases], minus),
                center_b[:gradient_cases],
                shape_b[:gradient_cases],
                iterations=args.iterations,
            ).signed_gap
            finite_difference[:, axis] = (plus_gap - minus_gap) / (2 * epsilon)
    gradient_absolute_error = (scale_gradient - finite_difference).abs()
    gradient_relative_error = gradient_absolute_error / finite_difference.abs().clamp_min(1e-3)
    anisotropy = torch.maximum(
        scales_a.max(dim=-1).values / scales_a.min(dim=-1).values,
        scales_b.max(dim=-1).values / scales_b.min(dim=-1).values,
    )
    strata = {
        "overlap": subset_metrics(reference_gap <= 0, gap_error, sphere_error),
        "separated": subset_metrics(reference_gap > 0, gap_error, sphere_error),
        "anisotropy_1_to_2": subset_metrics(anisotropy < 2, gap_error, sphere_error),
        "anisotropy_2_to_4": subset_metrics(
            (anisotropy >= 2) & (anisotropy < 4), gap_error, sphere_error
        ),
        "anisotropy_4_to_8": subset_metrics(
            (anisotropy >= 4) & (anisotropy < 8), gap_error, sphere_error
        ),
        "anisotropy_8_plus": subset_metrics(anisotropy >= 8, gap_error, sphere_error),
    }

    report = {
        "cases": args.cases,
        "reference_directions": args.directions,
        "solver_iterations": args.iterations,
        "scale_range_m": [0.025, 0.25],
        "gpu": torch.cuda.get_device_name(0),
        "runtime_ms_total": begin.elapsed_time(end),
        "runtime_us_per_pair": begin.elapsed_time(end) * 1000 / args.cases,
        "ours_gap_absolute_error_m_p50_p95_p99_max": percentile(
            gap_error, [0.5, 0.95, 0.99, 1.0]
        ),
        "ours_normal_error_deg_p50_p95_p99_max": percentile(
            normal_error, [0.5, 0.95, 0.99, 1.0]
        ),
        "ours_contact_classification_accuracy": float(
            (classification == reference_classification).float().mean().cpu()
        ),
        "volume_equivalent_sphere_gap_error_m_p50_p95_p99_max": percentile(
            sphere_error, [0.5, 0.95, 0.99, 1.0]
        ),
        "sphere_to_ours_p95_error_ratio": float(
            (torch.quantile(sphere_error, 0.95) / torch.quantile(gap_error, 0.95).clamp_min(1e-12)).cpu()
        ),
        "scale_gradient_finite_difference_cases": gradient_cases,
        "scale_gradient_absolute_error_p50_p95_p99": percentile(
            gradient_absolute_error.flatten(), [0.5, 0.95, 0.99]
        ),
        "scale_gradient_relative_error_p50_p95_p99": percentile(
            gradient_relative_error.flatten(), [0.5, 0.95, 0.99]
        ),
        "stratified_metrics": strata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sample_output = args.output.with_suffix(".pt")
    torch.save(
        {
            "reference_gap_m": reference_gap.cpu(),
            "ours_gap_error_m": gap_error.cpu(),
            "ours_normal_error_deg": normal_error.cpu(),
            "sphere_gap_error_m": sphere_error.cpu(),
            "anisotropy_ratio": anisotropy.cpu(),
            "distance_factor": distance_factor.cpu(),
        },
        sample_output,
    )
    report["sample_output"] = str(sample_output)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
