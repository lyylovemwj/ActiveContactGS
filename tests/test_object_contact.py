import torch

from active_contact_gs.ellipsoid import ellipsoid_contact
from active_contact_gs.object_contact import gaussian_object_contact


DEVICE = "cuda"


def test_single_gaussian_objects_match_pair_contact() -> None:
    means_a = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
    means_b = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
    shape_a = torch.diag(torch.tensor([0.3**2, 0.1**2, 0.1**2], device=DEVICE)).unsqueeze(0)
    shape_b = (torch.eye(3, device=DEVICE) * 0.2**2).unsqueeze(0)
    identity = torch.eye(3, device=DEVICE)
    translation_a = torch.zeros(3, device=DEVICE)
    translation_b = torch.tensor([1.0, 0.2, 0.0], device=DEVICE)
    expected = ellipsoid_contact(
        means_a[0], shape_a[0], means_b[0] + translation_b, shape_b[0]
    )
    result = gaussian_object_contact(
        means_a,
        shape_a,
        identity,
        translation_a,
        means_b,
        shape_b,
        identity,
        translation_b,
    )
    assert torch.allclose(result.signed_gap, expected.signed_gap, atol=1e-6)
    assert result.certified_global_minimum.item()


def test_broadphase_certificate_matches_exhaustive_pairs() -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    means_a = torch.randn((12, 3), device=DEVICE, generator=generator) * 0.2
    means_b = torch.randn((15, 3), device=DEVICE, generator=generator) * 0.2
    scales_a = torch.rand((12, 3), device=DEVICE, generator=generator) * 0.04 + 0.01
    scales_b = torch.rand((15, 3), device=DEVICE, generator=generator) * 0.04 + 0.01
    shapes_a = torch.diag_embed(scales_a.square())
    shapes_b = torch.diag_embed(scales_b.square())
    identity = torch.eye(3, device=DEVICE)
    zero = torch.zeros(3, device=DEVICE)
    exhaustive = gaussian_object_contact(
        means_a, shapes_a, identity, zero, means_b, shapes_b, identity, zero, broadphase_pairs=10000
    )
    bounded = gaussian_object_contact(
        means_a, shapes_a, identity, zero, means_b, shapes_b, identity, zero, broadphase_pairs=64
    )
    assert bounded.certified_global_minimum.item()
    assert torch.allclose(bounded.signed_gap, exhaustive.signed_gap, atol=1e-6)


def test_object_contact_backpropagates_to_pose() -> None:
    means = torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], device=DEVICE)
    shapes = torch.eye(3, device=DEVICE).repeat(2, 1, 1) * 0.04**2
    identity = torch.eye(3, device=DEVICE)
    translation = torch.tensor([0.5, 0.1, 0.0], device=DEVICE, requires_grad=True)
    result = gaussian_object_contact(
        means,
        shapes,
        identity,
        torch.zeros(3, device=DEVICE),
        means,
        shapes,
        identity,
        translation,
        broadphase_pairs=4,
    )
    result.signed_gap.backward()
    assert translation.grad is not None
    assert torch.isfinite(translation.grad).all()
    assert translation.grad.norm() > 0
