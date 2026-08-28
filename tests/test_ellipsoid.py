import torch

from active_contact_gs.ellipsoid import ellipsoid_contact, ellipsoid_time_of_impact


DEVICE = "cuda"


def test_sphere_pair_matches_analytic_gap() -> None:
    center_a = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
    center_b = torch.tensor([3.0, 0.0, 0.0], device=DEVICE)
    shape_a = torch.eye(3, device=DEVICE) * 0.7**2
    shape_b = torch.eye(3, device=DEVICE) * 0.4**2
    result = ellipsoid_contact(center_a, shape_a, center_b, shape_b)
    assert torch.allclose(result.signed_gap, torch.tensor(1.9, device=DEVICE), atol=1e-4)
    assert torch.allclose(result.normal, torch.tensor([1.0, 0.0, 0.0], device=DEVICE), atol=1e-4)


def test_anisotropy_changes_contact_distance() -> None:
    center_a = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
    center_b = torch.tensor([0.0, 0.5, 0.0], device=DEVICE)
    elongated_x = torch.diag(torch.tensor([0.4**2, 0.1**2, 0.1**2], device=DEVICE))
    sphere = torch.eye(3, device=DEVICE) * 0.1**2
    result = ellipsoid_contact(center_a, elongated_x, center_b, sphere)
    assert torch.allclose(result.signed_gap, torch.tensor(0.3, device=DEVICE), atol=1e-3)


def test_gap_backpropagates_to_geometry() -> None:
    center_a = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
    center_b = torch.tensor([1.0, 0.2, 0.0], device=DEVICE, requires_grad=True)
    scale = torch.tensor([0.3, 0.15, 0.1], device=DEVICE, requires_grad=True)
    shape_a = torch.diag(scale.square())
    shape_b = torch.eye(3, device=DEVICE) * 0.2**2
    result = ellipsoid_contact(center_a, shape_a, center_b, shape_b)
    result.signed_gap.backward()
    assert torch.isfinite(center_b.grad).all()
    assert torch.isfinite(scale.grad).all()
    assert center_b.grad.norm() > 0
    assert scale.grad.norm() > 0


def test_sphere_time_of_impact_matches_analytic_solution() -> None:
    shape = (torch.eye(3, device=DEVICE) * 1.0**2).unsqueeze(0)
    result = ellipsoid_time_of_impact(
        torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE),
        torch.zeros((1, 3), device=DEVICE),
        shape,
        torch.tensor([[3.0, 0.0, 0.0]], device=DEVICE),
        torch.tensor([[-1.0, 0.0, 0.0]], device=DEVICE),
        shape,
        max_time=2.0,
    )
    assert result.hit.item()
    assert torch.allclose(
        result.time_of_impact, torch.tensor([1.0], device=DEVICE), atol=2e-5
    )


def test_time_of_impact_rejects_separating_motion() -> None:
    shape = torch.eye(3, device=DEVICE).unsqueeze(0)
    result = ellipsoid_time_of_impact(
        torch.zeros((1, 3), device=DEVICE),
        torch.zeros((1, 3), device=DEVICE),
        shape,
        torch.tensor([[3.0, 0.0, 0.0]], device=DEVICE),
        torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE),
        shape,
        max_time=2.0,
    )
    assert not result.hit.item()
    assert result.time_of_impact.item() == 2.0
