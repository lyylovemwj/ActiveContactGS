import torch

from active_contact_gs.amortized_model import AmortizedPhysicsPosterior
from active_contact_gs.amortized_world import (
    AmortizedContactWorld,
    intervention_actions,
    sample_tasks,
)
from active_contact_gs.evaluate_amortized import (
    acquisition_scores,
    amortized_information_scores,
    downstream_control_metrics,
)


def test_acquisition_scores_are_finite_on_cuda() -> None:
    generator = torch.Generator(device="cuda").manual_seed(11)
    tasks = sample_tasks(3, device="cuda", generator=generator)
    world = AmortizedContactWorld(steps=20, observation_stride=5)
    model = AmortizedPhysicsPosterior(
        observation_frames=5, width=48, heads=4, layers=2
    ).cuda()
    actions = torch.zeros((3, 2, 4), device="cuda")
    observations = torch.zeros((3, 2, 5, 4), device="cuda")
    mask = torch.zeros((3, 2), device="cuda", dtype=torch.bool)
    posterior = model(actions, observations, tasks.geometry, mask)
    scores = acquisition_scores(
        posterior,
        tasks.geometry,
        intervention_actions(device="cuda")[:5],
        world,
        posterior_samples=12,
        generator=generator,
        action_chunk=3,
    )
    assert scores.shape == (3, 5)
    assert torch.isfinite(scores).all()


def test_belief_control_metrics_run_on_cuda() -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)
    tasks = sample_tasks(2, device="cuda", generator=generator)
    world = AmortizedContactWorld(steps=20, observation_stride=5)
    model = AmortizedPhysicsPosterior(
        observation_frames=5, width=48, heads=4, layers=2
    ).cuda().eval()
    actions = torch.zeros((2, 1, 4), device="cuda")
    observations = torch.zeros((2, 1, 5, 4), device="cuda")
    mask = torch.zeros((2, 1), device="cuda", dtype=torch.bool)
    posterior = model(actions, observations, tasks.geometry, mask)
    metrics = downstream_control_metrics(
        posterior,
        tasks,
        world,
        intervention_actions(device="cuda"),
        generator=generator,
        posterior_samples=3,
    )
    assert all(value.shape == (2,) for value in metrics.values())
    assert all(torch.isfinite(value).all() for value in metrics.values())


def test_amortized_eig_scores_are_finite_on_cuda() -> None:
    generator = torch.Generator(device="cuda").manual_seed(13)
    tasks = sample_tasks(2, device="cuda", generator=generator)
    world = AmortizedContactWorld(steps=20, observation_stride=5)
    model = AmortizedPhysicsPosterior(
        observation_frames=5, width=48, heads=4, layers=2
    ).cuda().eval()
    actions = torch.zeros((2, 2, 4), device="cuda")
    observations = torch.zeros((2, 2, 5, 4), device="cuda")
    mask = torch.zeros((2, 2), device="cuda", dtype=torch.bool)
    posterior = model(actions, observations, tasks.geometry, mask)
    scores = amortized_information_scores(
        model,
        posterior,
        actions,
        observations,
        mask,
        tasks.geometry,
        intervention_actions(device="cuda")[:4],
        world,
        fantasy_samples=3,
        noise_std=0.006,
        generator=generator,
        action_chunk=2,
    )
    assert scores.shape == (2, 4)
    assert torch.isfinite(scores).all()
