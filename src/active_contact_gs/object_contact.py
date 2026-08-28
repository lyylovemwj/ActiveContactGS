from __future__ import annotations

from dataclasses import dataclass

import torch

from .ellipsoid import compiled_ellipsoid_contact, ellipsoid_contact


@dataclass
class GaussianObjectContact:
    signed_gap: torch.Tensor
    normal: torch.Tensor
    point_a: torch.Tensor
    point_b: torch.Tensor
    gaussian_a_index: torch.Tensor
    gaussian_b_index: torch.Tensor
    evaluated_pairs: int
    next_lower_bound: torch.Tensor
    certified_global_minimum: torch.Tensor


def transform_gaussian_object(
    means: torch.Tensor,
    shapes: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_means = means @ rotation.transpose(-1, -2) + translation
    world_shapes = rotation @ shapes @ rotation.transpose(-1, -2)
    return world_means, world_shapes


def gaussian_object_contact(
    means_a: torch.Tensor,
    shapes_a: torch.Tensor,
    rotation_a: torch.Tensor,
    translation_a: torch.Tensor,
    means_b: torch.Tensor,
    shapes_b: torch.Tensor,
    rotation_b: torch.Tensor,
    translation_b: torch.Tensor,
    *,
    broadphase_pairs: int = 256,
    contact_iterations: int = 64,
    bounding_radius_a: torch.Tensor | None = None,
    bounding_radius_b: torch.Tensor | None = None,
    compiled_narrowphase: bool = False,
) -> GaussianObjectContact:
    """Closest contact between two rigid objects represented by ellipsoids.

    Bounding spheres provide a conservative lower bound. The returned
    ``certified_global_minimum`` is true when every unevaluated lower bound is
    greater than the best exact ellipsoid gap, making the broad-phase answer
    globally exact for the represented Gaussian unions.
    """
    world_means_a, world_shapes_a = transform_gaussian_object(
        means_a, shapes_a, rotation_a, translation_a
    )
    world_means_b, world_shapes_b = transform_gaussian_object(
        means_b, shapes_b, rotation_b, translation_b
    )
    radius_a = (
        bounding_radius_a
        if bounding_radius_a is not None
        else torch.sqrt(torch.linalg.eigvalsh(shapes_a.detach())[..., -1])
    )
    radius_b = (
        bounding_radius_b
        if bounding_radius_b is not None
        else torch.sqrt(torch.linalg.eigvalsh(shapes_b.detach())[..., -1])
    )
    lower_bound = (
        torch.cdist(world_means_a, world_means_b)
        - radius_a[:, None]
        - radius_b[None, :]
    )
    flattened = lower_bound.flatten()
    total_pairs = flattened.numel()
    evaluated_pairs = min(broadphase_pairs, total_pairs)
    selection_count = min(evaluated_pairs + 1, total_pairs)
    selected_bound, selected = torch.topk(
        flattened, selection_count, largest=False, sorted=True
    )
    next_lower_bound = (
        selected_bound[-1]
        if evaluated_pairs < total_pairs
        else torch.full_like(selected_bound[-1], torch.inf)
    )
    selected = selected[:evaluated_pairs]
    index_a = torch.div(selected, world_means_b.shape[0], rounding_mode="floor")
    index_b = selected.remainder(world_means_b.shape[0])
    contact_function = compiled_ellipsoid_contact if compiled_narrowphase else ellipsoid_contact
    contacts = contact_function(
        world_means_a[index_a],
        world_shapes_a[index_a],
        world_means_b[index_b],
        world_shapes_b[index_b],
        iterations=contact_iterations,
    )
    best_local = contacts.signed_gap.argmin()
    best_gap = contacts.signed_gap[best_local]
    return GaussianObjectContact(
        signed_gap=best_gap,
        normal=contacts.normal[best_local],
        point_a=contacts.point_a[best_local],
        point_b=contacts.point_b[best_local],
        gaussian_a_index=index_a[best_local],
        gaussian_b_index=index_b[best_local],
        evaluated_pairs=evaluated_pairs,
        next_lower_bound=next_lower_bound,
        certified_global_minimum=next_lower_bound >= best_gap,
    )
