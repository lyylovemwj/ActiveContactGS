"""Small analytic checks that run on standard GitHub-hosted CPU runners."""

import torch

import active_contact_gs
from active_contact_gs.ellipsoid import ellipsoid_contact
from active_contact_gs.physics import PlanarRigidBodySimulator


def test_public_api_imports_on_cpu() -> None:
    assert active_contact_gs.ParticleBelief is not None
    assert active_contact_gs.PlanarRigidBodySimulator is PlanarRigidBodySimulator


def test_analytic_sphere_contact_runs_on_cpu() -> None:
    center_a = torch.tensor([0.0, 0.0, 0.0])
    center_b = torch.tensor([3.0, 0.0, 0.0])
    shape_a = torch.eye(3) * 0.7**2
    shape_b = torch.eye(3) * 0.4**2
    result = ellipsoid_contact(center_a, shape_a, center_b, shape_b)
    assert torch.allclose(result.signed_gap, torch.tensor(1.9), atol=1e-4)
    assert torch.allclose(result.normal, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)
