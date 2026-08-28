import torch

from active_contact_gs.amortized_world import (
    AmortizedContactWorld,
    intervention_actions,
    sample_tasks,
)
from active_contact_gs.video_model import GaussianVideoPosterior, render_gaussian_video


def test_gaussian_video_posterior_runs_on_cuda() -> None:
    generator = torch.Generator(device="cuda").manual_seed(31)
    tasks = sample_tasks(4, device="cuda", generator=generator)
    world = AmortizedContactWorld(steps=20, observation_stride=5)
    selected = intervention_actions(device="cuda")[-4:]
    observations = world.observe(tasks, selected, noise_std=0.004, generator=generator)
    video = render_gaussian_video(
        observations[:, None], tasks.geometry[:, None], resolution=32
    )
    assert video.shape == (4, 1, 6, 3, 32, 32)
    assert video.min() >= 0 and video.max() <= 1
    model = GaussianVideoPosterior(
        observation_frames=5,
        width=48,
        heads=4,
        layers=2,
        video_frames=6,
        image_resolution=32,
    ).cuda()
    posterior = model(
        selected[:, None],
        observations[:, None],
        tasks.geometry,
        torch.ones((4, 1), device="cuda", dtype=torch.bool),
    )
    loss, _ = model.loss(posterior, tasks.parameters, tasks.hypotheses)
    loss.backward()
    assert torch.isfinite(loss)
    assert posterior.parameter_mean.shape == (4, 4)
