import torch

from active_contact_gs.evaluate_iclr import information_scores
from active_contact_gs.iclr_model import ICLRPhysicsPosterior
from active_contact_gs.iclr_world import iclr_actions, sample_iclr_tasks


def test_information_scores_support_batch_one() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(41)
    tasks = sample_iclr_tasks(1, device=device, generator=generator, split="id")
    model = ICLRPhysicsPosterior(width=32, heads=4, layers=1).to(device).eval()
    actions = torch.zeros((1, 1, 4), device=device)
    observations = torch.zeros((1, 1, 19, 4), device=device)
    mask = torch.zeros((1, 1), device=device, dtype=torch.bool)
    posterior = model(actions, observations, tasks.geometry, mask)
    score = information_scores(
        model,
        posterior,
        actions,
        observations,
        mask,
        tasks.geometry,
        iclr_actions(device=device)[:2],
        samples=2,
        noise_std=0.006,
        generator=generator,
        candidate_chunk=2,
    )
    assert score.shape == (1, 2)
    assert torch.isfinite(score).all()
