import torch

from active_contact_gs.amortized_world import (
    AmortizedContactWorld,
    intervention_actions,
    sample_tasks,
)
from active_contact_gs.particle_baseline import particle_inference


def test_particle_baseline_runs_on_cuda() -> None:
    generator = torch.Generator(device="cuda").manual_seed(29)
    tasks = sample_tasks(2, device="cuda", generator=generator)
    world = AmortizedContactWorld(steps=20, observation_stride=5)
    actions = intervention_actions(device="cuda")[-2:].expand(2, -1, -1)
    observations = torch.stack(
        [
            world.observe(
                tasks, actions[:, probe], noise_std=0.006, generator=generator
            )
            for probe in range(2)
        ],
        dim=1,
    )
    estimate, probability, ess = particle_inference(
        tasks,
        actions,
        observations,
        particle_count=128,
        noise_std=0.006,
        generator=generator,
        world=world,
    )
    assert estimate.shape == (2, 4)
    assert probability.shape == (2,)
    assert torch.isfinite(estimate).all()
    assert torch.isfinite(ess).all()
