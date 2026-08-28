from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .ellipsoid import shape_from_rotation_scale
from .object_contact import gaussian_object_contact


def load_object(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cuda", weights_only=True)
    means = data["means_m"].to("cuda")
    scales = data["scales_m"].to("cuda")
    rotations = data["rotations"].to("cuda")
    shapes = shape_from_rotation_scale(rotations, scales)
    equivalent_radius = scales.prod(dim=-1).pow(1 / 3)
    sphere_shapes = torch.diag_embed(equivalent_radius[:, None].expand(-1, 3).square())
    return means, shapes, sphere_shapes


def rotation_z(angle: torch.Tensor) -> torch.Tensor:
    cosine, sine = torch.cos(angle), torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((cosine, -sine, zero)),
            torch.stack((sine, cosine, zero)),
            torch.stack((zero, zero, one)),
        )
    )


def quantiles(values: torch.Tensor) -> list[float]:
    return torch.quantile(
        values, torch.tensor([0.5, 0.95, 0.99, 1.0], device="cuda")
    ).cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object-a",
        type=Path,
        default=Path("data/processed/pin_wm/gaussians/cube_t.pt"),
    )
    parser.add_argument(
        "--object-b",
        type=Path,
        default=Path("data/processed/pin_wm/gaussians/ee.pt"),
    )
    parser.add_argument("--scenes", type=int, default=128)
    parser.add_argument("--reference-pairs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, default=Path("outputs/object_contact.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Object contact benchmark requires CUDA")

    means_a, shapes_a, sphere_shapes_a = load_object(args.object_a)
    means_b, shapes_b, sphere_shapes_b = load_object(args.object_b)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    identity = torch.eye(3, device="cuda")
    zero = torch.zeros(3, device="cuda")
    radius_a = torch.linalg.vector_norm(means_a, dim=-1).max() + torch.sqrt(
        torch.linalg.eigvalsh(shapes_a)[..., -1]
    ).max()
    radius_b = torch.linalg.vector_norm(means_b, dim=-1).max() + torch.sqrt(
        torch.linalg.eigvalsh(shapes_b)[..., -1]
    ).max()
    gaussian_radius_a = torch.sqrt(torch.linalg.eigvalsh(shapes_a)[..., -1])
    gaussian_radius_b = torch.sqrt(torch.linalg.eigvalsh(shapes_b)[..., -1])
    sphere_radius_a = torch.sqrt(torch.linalg.eigvalsh(sphere_shapes_a)[..., -1])
    sphere_radius_b = torch.sqrt(torch.linalg.eigvalsh(sphere_shapes_b)[..., -1])
    total_pairs = means_a.shape[0] * means_b.shape[0]
    methods = [
        count
        for count in (16, 32, 64, 128, 256, 512, 1024, 2048)
        if count <= total_pairs
    ]
    records: dict[str, list[dict[str, float | bool]]] = {
        f"ellipsoid_{count}": [] for count in methods
    }
    records["sphere_512"] = []
    records["ellipsoid_adaptive"] = []
    uncertified_references = 0

    for scene in range(args.scenes):
        angle = torch.rand((), device="cuda", generator=generator) * (2 * math.pi)
        rotation_b = rotation_z(angle)
        direction_angle = torch.rand((), device="cuda", generator=generator) * (2 * math.pi)
        direction = torch.stack((torch.cos(direction_angle), torch.sin(direction_angle), torch.zeros_like(direction_angle)))
        distance_factor = torch.empty((), device="cuda").uniform_(0.45, 1.45, generator=generator)
        translation_b = direction * (radius_a + radius_b) * distance_factor

        reference_pairs = min(args.reference_pairs, total_pairs)
        torch.cuda.synchronize()
        reference_started = time.perf_counter()
        reference = gaussian_object_contact(
            means_a,
            shapes_a,
            identity,
            zero,
            means_b,
            shapes_b,
            rotation_b,
            translation_b,
            broadphase_pairs=reference_pairs,
            bounding_radius_a=gaussian_radius_a,
            bounding_radius_b=gaussian_radius_b,
            compiled_narrowphase=True,
        )
        while not reference.certified_global_minimum.item() and reference_pairs < total_pairs:
            reference_pairs = min(reference_pairs * 2, total_pairs)
            reference = gaussian_object_contact(
                means_a,
                shapes_a,
                identity,
                zero,
                means_b,
                shapes_b,
                rotation_b,
                translation_b,
                broadphase_pairs=reference_pairs,
                bounding_radius_a=gaussian_radius_a,
                bounding_radius_b=gaussian_radius_b,
                compiled_narrowphase=True,
            )
        torch.cuda.synchronize()
        reference_runtime_ms = (time.perf_counter() - reference_started) * 1000
        uncertified_references += int(not reference.certified_global_minimum.item())
        records["ellipsoid_adaptive"].append(
            {
                "gap_error_m": 0.0,
                "normal_error_deg": 0.0,
                "classification_correct": True,
                "certified": bool(reference.certified_global_minimum.cpu()),
                "runtime_ms": reference_runtime_ms,
                "candidate_pairs": reference_pairs,
            }
        )

        for count in methods:
            if scene == 0:
                gaussian_object_contact(
                    means_a,
                    shapes_a,
                    identity,
                    zero,
                    means_b,
                    shapes_b,
                    rotation_b,
                    translation_b,
                    broadphase_pairs=count,
                    bounding_radius_a=gaussian_radius_a,
                    bounding_radius_b=gaussian_radius_b,
                    compiled_narrowphase=True,
                )
                torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = gaussian_object_contact(
                means_a,
                shapes_a,
                identity,
                zero,
                means_b,
                shapes_b,
                rotation_b,
                translation_b,
                broadphase_pairs=count,
                bounding_radius_a=gaussian_radius_a,
                bounding_radius_b=gaussian_radius_b,
                compiled_narrowphase=True,
            )
            end.record()
            torch.cuda.synchronize()
            cosine = (result.normal * reference.normal).sum().clamp(-1, 1)
            records[f"ellipsoid_{count}"].append(
                {
                    "gap_error_m": float((result.signed_gap - reference.signed_gap).abs().cpu()),
                    "normal_error_deg": float(torch.rad2deg(torch.acos(cosine)).cpu()),
                    "classification_correct": bool(((result.signed_gap > 0) == (reference.signed_gap > 0)).cpu()),
                    "certified": bool(result.certified_global_minimum.cpu()),
                    "runtime_ms": start.elapsed_time(end),
                }
            )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        sphere = gaussian_object_contact(
            means_a,
            sphere_shapes_a,
            identity,
            zero,
            means_b,
            sphere_shapes_b,
            rotation_b,
            translation_b,
            broadphase_pairs=512,
            bounding_radius_a=sphere_radius_a,
            bounding_radius_b=sphere_radius_b,
            compiled_narrowphase=True,
        )
        end.record()
        torch.cuda.synchronize()
        cosine = (sphere.normal * reference.normal).sum().clamp(-1, 1)
        records["sphere_512"].append(
            {
                "gap_error_m": float((sphere.signed_gap - reference.signed_gap).abs().cpu()),
                "normal_error_deg": float(torch.rad2deg(torch.acos(cosine)).cpu()),
                "classification_correct": bool(((sphere.signed_gap > 0) == (reference.signed_gap > 0)).cpu()),
                "certified": bool(sphere.certified_global_minimum.cpu()),
                "runtime_ms": start.elapsed_time(end),
            }
        )

    summary = {}
    for method, rows in records.items():
        gap = torch.tensor([row["gap_error_m"] for row in rows], device="cuda")
        normal = torch.tensor([row["normal_error_deg"] for row in rows], device="cuda")
        runtime = torch.tensor([row["runtime_ms"] for row in rows], device="cuda")
        summary[method] = {
            "gap_error_mm_p50_p95_p99_max": [value * 1000 for value in quantiles(gap)],
            "normal_error_deg_p50_p95_p99_max": quantiles(normal),
            "contact_classification_accuracy": sum(bool(row["classification_correct"]) for row in rows) / len(rows),
            "certificate_rate": sum(bool(row["certified"]) for row in rows) / len(rows),
            "mean_runtime_ms": sum(float(row["runtime_ms"]) for row in rows) / len(rows),
            "runtime_ms_p50_p95_p99_max": quantiles(runtime),
        }
        if "candidate_pairs" in rows[0]:
            candidates = torch.tensor(
                [row["candidate_pairs"] for row in rows], device="cuda", dtype=torch.float32
            )
            summary[method]["candidate_pairs_p50_p95_p99_max"] = quantiles(candidates)
    output = {
        "gpu": torch.cuda.get_device_name(0),
        "scenes": args.scenes,
        "gaussians_a": means_a.shape[0],
        "gaussians_b": means_b.shape[0],
        "uncertified_references": uncertified_references,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
