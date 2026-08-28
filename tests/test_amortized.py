import torch

from active_contact_gs.amortized_model import AmortizedPhysicsPosterior
from active_contact_gs.amortized_world import (
    AmortizedContactWorld,
    intervention_actions,
    sample_tasks,
)


def test_amortized_world_and_posterior_run_on_cuda() -> None:
    assert torch.cuda.is_available()
    generator = torch.Generator(device="cuda").manual_seed(7)
    tasks = sample_tasks(8, device="cuda", generator=generator)
    world = AmortizedContactWorld(steps=20, observation_stride=5)
    actions = intervention_actions(device="cuda")[:3]
    selected = actions[torch.arange(8, device="cuda").remainder(3)]
    observations = world.observe(tasks, selected, noise_std=0.004, generator=generator)
    assert observations.shape == (8, 5, 4)
    model = AmortizedPhysicsPosterior(
        observation_frames=5, width=48, heads=4, layers=2
    ).cuda()
    posterior = model(
        selected[:, None], observations[:, None], tasks.geometry, torch.ones((8, 1), device="cuda", dtype=torch.bool)
    )
    loss, metrics = model.loss(posterior, tasks.parameters, tasks.hypotheses)
    loss.backward()
    assert torch.isfinite(loss)
    assert posterior.parameter_mean.shape == (8, 4)
    assert torch.isfinite(metrics["normalized_mae"])


def test_visual_geometry_is_shared_across_contact_hypotheses() -> None:
    generator = torch.Generator(device="cuda").manual_seed(8)
    tasks = sample_tasks(2, device="cuda", generator=generator)
    parameters = tasks.parameters[:1].expand(2, -1)
    geometry = tasks.geometry[:1].expand(2, -1)
    hypotheses = torch.tensor([0, 1], device="cuda")
    action = intervention_actions(device="cuda")[-1:].expand(2, -1)
    world = AmortizedContactWorld()
    trajectories = world.rollout(parameters, hypotheses, action, geometry)
    assert not torch.allclose(trajectories[0], trajectories[1])
    assert torch.allclose(geometry[0], geometry[1])
