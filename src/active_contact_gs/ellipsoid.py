from __future__ import annotations

from dataclasses import dataclass

import torch


_COMPILED_ELLIPSOID_CONTACT = None


@dataclass
class EllipsoidContact:
    signed_gap: torch.Tensor
    normal: torch.Tensor
    point_a: torch.Tensor
    point_b: torch.Tensor


@dataclass
class ContinuousEllipsoidContact:
    time_of_impact: torch.Tensor
    hit: torch.Tensor
    contact: EllipsoidContact


def _support_point(center: torch.Tensor, shape: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Support point for ``(x-c)^T shape^-1 (x-c) <= 1``."""
    shape_n = torch.einsum("...ij,...j->...i", shape, normal)
    denominator = torch.sqrt(
        torch.sum(normal * shape_n, dim=-1, keepdim=True).clamp_min(1e-12)
    )
    return center + shape_n / denominator


def ellipsoid_contact(
    center_a: torch.Tensor,
    shape_a: torch.Tensor,
    center_b: torch.Tensor,
    shape_b: torch.Tensor,
    *,
    iterations: int = 64,
    step_size: float = 0.3,
    multistart: bool = True,
) -> EllipsoidContact:
    """Differentiable signed separation and contact normal for two ellipsoids.

    ``shape`` is the positive-definite matrix whose eigenvalues are squared
    semi-axis lengths. We maximize the separating support-plane gap on the unit
    sphere with fixed unrolled Riemannian gradient steps. Positive values denote
    separation and negative values denote overlap.
    """
    delta = center_b - center_a
    fallback = torch.zeros_like(delta)
    fallback[..., 0] = 1.0
    delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    delta_normal = delta / delta_norm.clamp_min(1e-12)
    delta_normal = torch.where(delta_norm > 1e-9, delta_normal, fallback)

    if multistart:
        # Principal axes cover difficult deep-overlap cases whose optimum can be
        # far from the center-to-center direction. Initializers are detached:
        # they affect basin selection but are not part of the contact gradient.
        axes_a = torch.linalg.eigh(shape_a.detach()).eigenvectors.transpose(-1, -2)
        axes_b = torch.linalg.eigh(shape_b.detach()).eigenvectors.transpose(-1, -2)
        normal = torch.cat(
            (
                delta_normal.unsqueeze(-2),
                axes_a,
                -axes_a,
                axes_b,
                -axes_b,
            ),
            dim=-2,
        )
    else:
        normal = delta_normal.unsqueeze(-2)

    shape_a_expanded = shape_a.unsqueeze(-3)
    shape_b_expanded = shape_b.unsqueeze(-3)
    radius_scale = (
        torch.sqrt(torch.diagonal(shape_a, dim1=-2, dim2=-1).sum(dim=-1))
        + torch.sqrt(torch.diagonal(shape_b, dim1=-2, dim2=-1).sum(dim=-1))
    ).unsqueeze(-1).unsqueeze(-1).clamp_min(1e-8)

    for _ in range(iterations):
        qa_n = torch.matmul(shape_a_expanded, normal.unsqueeze(-1)).squeeze(-1)
        qb_n = torch.matmul(shape_b_expanded, normal.unsqueeze(-1)).squeeze(-1)
        radius_a = torch.sqrt(torch.sum(normal * qa_n, dim=-1, keepdim=True).clamp_min(1e-12))
        radius_b = torch.sqrt(torch.sum(normal * qb_n, dim=-1, keepdim=True).clamp_min(1e-12))
        euclidean_gradient = delta.unsqueeze(-2) - qa_n / radius_a - qb_n / radius_b
        tangent_gradient = euclidean_gradient - normal * torch.sum(
            normal * euclidean_gradient, dim=-1, keepdim=True
        )
        # Cap the angular update to make one step stable across scene scales.
        update_scale = torch.maximum(
            radius_scale,
            torch.linalg.vector_norm(tangent_gradient, dim=-1, keepdim=True),
        )
        normal = torch.nn.functional.normalize(
            normal + step_size * tangent_gradient / update_scale, dim=-1
        )

    qa_n = torch.matmul(shape_a_expanded, normal.unsqueeze(-1)).squeeze(-1)
    qb_n = torch.matmul(shape_b_expanded, normal.unsqueeze(-1)).squeeze(-1)
    radius_a = torch.sqrt(torch.sum(normal * qa_n, dim=-1).clamp_min(1e-12))
    radius_b = torch.sqrt(torch.sum(normal * qb_n, dim=-1).clamp_min(1e-12))
    candidate_gap = torch.sum(normal * delta.unsqueeze(-2), dim=-1) - radius_a - radius_b
    best = candidate_gap.argmax(dim=-1, keepdim=True)
    gather_index = best.unsqueeze(-1).expand(best.shape + (3,))
    normal = torch.gather(normal, dim=-2, index=gather_index).squeeze(-2)
    signed_gap = torch.gather(candidate_gap, dim=-1, index=best).squeeze(-1)

    point_a = _support_point(center_a, shape_a, normal)
    point_b = _support_point(center_b, shape_b, -normal)
    return EllipsoidContact(
        signed_gap=signed_gap,
        normal=normal,
        point_a=point_a,
        point_b=point_b,
    )


def compiled_ellipsoid_contact(*args, **kwargs) -> EllipsoidContact:
    """Inductor-fused contact kernel with CUDA graph capture disabled.

    ``torch.linalg.eigh`` cannot be CUDA-graph captured on the current PyTorch
    stack, but regular Inductor compilation still fuses the unrolled optimizer.
    Compilation is lazy and cached by PyTorch for each input shape.
    """
    global _COMPILED_ELLIPSOID_CONTACT
    if _COMPILED_ELLIPSOID_CONTACT is None:
        _COMPILED_ELLIPSOID_CONTACT = torch.compile(
            ellipsoid_contact, options={"triton.cudagraphs": False}
        )
    return _COMPILED_ELLIPSOID_CONTACT(*args, **kwargs)


def shape_from_rotation_scale(rotation: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Convert 3DGS rotation and scale to an ellipsoid shape matrix."""
    return rotation @ torch.diag_embed(scale.square()) @ rotation.transpose(-1, -2)


def ellipsoid_time_of_impact(
    center_a: torch.Tensor,
    velocity_a: torch.Tensor,
    shape_a: torch.Tensor,
    center_b: torch.Tensor,
    velocity_b: torch.Tensor,
    shape_b: torch.Tensor,
    *,
    max_time: float = 1.0,
    iterations: int = 20,
    contact_iterations: int = 48,
    tolerance: float = 1e-6,
    safety: float = 0.9,
) -> ContinuousEllipsoidContact:
    """Differentiable conservative advancement for translating ellipsoids.

    Orientations are fixed over the query. Rotational motion can be handled by
    subdividing the rigid-body step according to an angular-motion bound.
    """
    time = torch.zeros_like(center_a[..., 0])
    relative_velocity = velocity_b - velocity_a
    active = torch.ones_like(time, dtype=torch.bool)
    hit = torch.zeros_like(time, dtype=torch.bool)

    for _ in range(iterations):
        query_a = center_a + velocity_a * time.unsqueeze(-1)
        query_b = center_b + velocity_b * time.unsqueeze(-1)
        contact = ellipsoid_contact(
            query_a,
            shape_a,
            query_b,
            shape_b,
            iterations=contact_iterations,
        )
        reached = contact.signed_gap <= tolerance
        hit = hit | (active & reached)
        closing_speed = -(contact.normal * relative_velocity).sum(dim=-1)
        can_advance = active & ~reached & (closing_speed > 1e-9)
        delta_time = (
            safety
            * (contact.signed_gap - tolerance).clamp_min(0)
            / closing_speed.clamp_min(1e-9)
        )
        proposed = time + delta_time
        update = can_advance & (proposed <= max_time)
        time = torch.where(update, proposed, time)
        active = update

    query_a = center_a + velocity_a * time.unsqueeze(-1)
    query_b = center_b + velocity_b * time.unsqueeze(-1)
    contact = ellipsoid_contact(
        query_a,
        shape_a,
        query_b,
        shape_b,
        iterations=contact_iterations,
    )
    hit = hit | (active & (contact.signed_gap <= tolerance * 2))
    time_of_impact = torch.where(hit, time, torch.full_like(time, max_time))
    return ContinuousEllipsoidContact(
        time_of_impact=time_of_impact,
        hit=hit,
        contact=contact,
    )
