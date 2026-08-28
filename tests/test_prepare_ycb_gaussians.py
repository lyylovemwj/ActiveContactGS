import torch

from active_contact_gs.prepare_ycb_gaussians import kmeans_gpu, sample_mesh_surface_gpu


DEVICE = "cuda"


def test_gpu_surface_sampling_stays_on_triangle() -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=DEVICE
    )
    faces = torch.tensor([[0, 1, 2]], device=DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    points, normals = sample_mesh_surface_gpu(vertices, faces, 4096, generator=generator)
    assert torch.allclose(points[:, 2], torch.zeros_like(points[:, 2]))
    assert ((points[:, :2] >= 0).all(dim=-1) & (points[:, :2].sum(dim=-1) <= 1)).all()
    assert torch.allclose(normals, torch.tensor([0.0, 0.0, 1.0], device=DEVICE).expand_as(normals))


def test_gpu_kmeans_separates_two_clouds() -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    points = torch.cat(
        (
            torch.randn((2048, 3), device=DEVICE, generator=generator) * 0.01 - 1,
            torch.randn((2048, 3), device=DEVICE, generator=generator) * 0.01 + 1,
        )
    )
    centers, assignment = kmeans_gpu(points, 2, 8, generator=generator)
    assert centers[:, 0].sort().values[0] < -0.9
    assert centers[:, 0].sort().values[1] > 0.9
    assert assignment.unique().numel() == 2
