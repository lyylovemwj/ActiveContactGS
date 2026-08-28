from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import trimesh


def sample_mesh_surface_gpu(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    count: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    triangles = vertices[faces]
    cross = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = torch.linalg.vector_norm(cross, dim=-1).clamp_min(1e-12)
    selected = torch.multinomial(double_area, count, replacement=True, generator=generator)
    chosen = triangles[selected]
    normal = torch.nn.functional.normalize(cross[selected], dim=-1)
    random = torch.rand((count, 2), device="cuda", generator=generator)
    root = torch.sqrt(random[:, :1])
    points = (
        (1 - root) * chosen[:, 0]
        + root * (1 - random[:, 1:]) * chosen[:, 1]
        + root * random[:, 1:] * chosen[:, 2]
    )
    return points, normal


@torch.no_grad()
def kmeans_gpu(
    points: torch.Tensor,
    clusters: int,
    iterations: int,
    *,
    generator: torch.Generator,
    chunk: int = 32768,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clusters > points.shape[0]:
        raise ValueError("clusters cannot exceed points")
    centers = torch.empty((clusters, 3), device="cuda", dtype=points.dtype)
    first = torch.randint(points.shape[0], (1,), device="cuda", generator=generator)
    centers[0] = points[first]
    nearest = torch.full((points.shape[0],), torch.inf, device="cuda")
    for index in range(1, clusters):
        distance = (points - centers[index - 1]).square().sum(dim=-1)
        nearest = torch.minimum(nearest, distance)
        # Farthest-point initialization gives deterministic surface coverage.
        centers[index] = points[nearest.argmax()]

    assignment = torch.zeros(points.shape[0], dtype=torch.long, device="cuda")
    for _ in range(iterations):
        for start in range(0, points.shape[0], chunk):
            stop = min(start + chunk, points.shape[0])
            assignment[start:stop] = torch.cdist(points[start:stop], centers).argmin(dim=-1)
        sums = torch.zeros_like(centers).index_add_(0, assignment, points)
        counts = torch.bincount(assignment, minlength=clusters).to(points.dtype).unsqueeze(-1)
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
    return centers, assignment


@torch.no_grad()
def fit_surface_gaussians(
    points: torch.Tensor,
    clusters: int,
    iterations: int,
    *,
    generator: torch.Generator,
    object_diagonal: torch.Tensor,
) -> dict[str, torch.Tensor]:
    centers, assignment = kmeans_gpu(
        points, clusters, iterations, generator=generator
    )
    difference = points - centers[assignment]
    outer = difference.unsqueeze(-1) * difference.unsqueeze(-2)
    covariance_sum = torch.zeros((clusters, 3, 3), device="cuda").index_add_(
        0, assignment, outer
    )
    counts = torch.bincount(assignment, minlength=clusters).to(points.dtype)
    covariance = covariance_sum / counts[:, None, None].clamp_min(1)
    regularizer = (object_diagonal * 2e-4).square()
    covariance = covariance + torch.eye(3, device="cuda") * regularizer
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scales = 2.5 * torch.sqrt(eigenvalues.clamp_min(regularizer))
    min_thickness = object_diagonal * 8e-4
    max_extent = object_diagonal * 0.18
    scales = scales.clamp(min=min_thickness, max=max_extent)
    return {
        "means": centers,
        "rotations": eigenvectors,
        "scales": scales,
        "opacity": torch.ones((clusters, 1), device="cuda"),
        "cluster_counts": counts,
        "assignment": assignment,
    }


def locate_meshes(root: Path) -> list[Path]:
    candidates = sorted(root.rglob("textured.obj"))
    if not candidates:
        candidates = sorted(root.rglob("*.obj"))
    # Exactly one Google 16k textured mesh is expected per selected object.
    return [path for path in candidates if "google_16k" in str(path)] or candidates


def percentile(values: torch.Tensor) -> list[float]:
    return torch.quantile(values, torch.tensor([0.5, 0.95, 0.99], device="cuda")).cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/ycb16k/objects"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/ycb16k/gaussians"),
    )
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--gaussians", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--mesh", type=Path, action="append")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("YCB Gaussian preparation requires CUDA")

    args.output.mkdir(parents=True, exist_ok=True)
    meshes = args.mesh or locate_meshes(args.input)
    if not meshes:
        raise FileNotFoundError(f"No OBJ meshes found below {args.input}")
    summaries = []
    for mesh_index, mesh_path in enumerate(meshes):
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device="cuda")
        faces = torch.as_tensor(mesh.faces, dtype=torch.long, device="cuda")
        raw_diagonal = torch.linalg.vector_norm(vertices.max(dim=0).values - vertices.min(dim=0).values)
        unit_scale = 0.001 if raw_diagonal > 10 else 1.0
        vertices = vertices * unit_scale
        center = (vertices.max(dim=0).values + vertices.min(dim=0).values) / 2
        vertices = vertices - center
        diagonal = torch.linalg.vector_norm(vertices.max(dim=0).values - vertices.min(dim=0).values)
        generator = torch.Generator(device="cuda").manual_seed(args.seed + mesh_index)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        points, normals = sample_mesh_surface_gpu(
            vertices, faces, args.samples, generator=generator
        )
        gaussians = fit_surface_gaussians(
            points,
            args.gaussians,
            args.iterations,
            generator=generator,
            object_diagonal=diagonal,
        )
        end.record()
        torch.cuda.synchronize()

        nearest_center = torch.empty(points.shape[0], device="cuda")
        for chunk_start in range(0, points.shape[0], 32768):
            chunk_end = min(chunk_start + 32768, points.shape[0])
            nearest_center[chunk_start:chunk_end] = torch.cdist(
                points[chunk_start:chunk_end], gaussians["means"]
            ).min(dim=-1).values
        object_name = next(
            (parent.name for parent in mesh_path.parents if parent.name[:3].isdigit()),
            mesh_path.parent.name,
        )
        output_path = args.output / f"{object_name}.pt"
        torch.save(
            {
                "source_mesh": str(mesh_path),
                "source_unit_scale": unit_scale,
                "mesh_center_source_units": center.cpu(),
                "vertices_m": vertices.cpu(),
                "faces": faces.cpu(),
                "surface_sample_m": points[: min(50000, points.shape[0])].cpu(),
                "surface_normal": normals[: min(50000, normals.shape[0])].cpu(),
                "means_m": gaussians["means"].cpu(),
                "rotations": gaussians["rotations"].cpu(),
                "scales_m": gaussians["scales"].cpu(),
                "opacity": gaussians["opacity"].cpu(),
                "cluster_counts": gaussians["cluster_counts"].cpu(),
            },
            output_path,
        )
        summary = {
            "object": object_name,
            "source_mesh": str(mesh_path),
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "unit_scale_to_m": unit_scale,
            "diagonal_m": float(diagonal.cpu()),
            "surface_samples": args.samples,
            "gaussians": args.gaussians,
            "gpu_fit_time_ms": start.elapsed_time(end),
            "nearest_center_error_mm_p50_p95_p99": [
                value * 1000 for value in percentile(nearest_center)
            ],
            "output": str(output_path),
        }
        summaries.append(summary)
        print(json.dumps(summary))

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "objects": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
