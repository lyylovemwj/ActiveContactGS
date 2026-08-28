import torch

from active_contact_gs.amortized_world import (
    AmortizedContactWorld,
    intervention_actions,
    normalize_parameters,
    sample_tasks,
)
from active_contact_gs.identifiability_analysis import finite_difference_jacobian


def test_relaxed_geometry_jacobian_is_finite_on_cuda() -> None:
    generator = torch.Generator(device="cuda").manual_seed(23)
    tasks = sample_tasks(3, device="cuda", generator=generator)
    world = AmortizedContactWorld(steps=20, observation_stride=5)
    actions = intervention_actions(device="cuda")[-3:].expand(3, -1, -1)
    jacobian = finite_difference_jacobian(
        world,
        normalize_parameters(tasks.parameters),
        tasks.hypotheses.float(),
        tasks.geometry,
        actions,
        epsilon=1e-3,
    )
    assert jacobian.shape == (3, 3 * 5 * 4, 5)
    assert torch.isfinite(jacobian).all()
    assert jacobian.abs().sum() > 0
