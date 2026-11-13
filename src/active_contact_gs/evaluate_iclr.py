from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .iclr_model import ICLRPhysicsPosterior, posterior_metrics
from .iclr_world import (
    ICLRContactWorld,
    ICLRTaskBatch,
    action_direction_indices,
    all_factor_tuples,
    denormalize_parameters,
    iclr_actions,
    sample_iclr_tasks,
)


def load_model(path: Path, device: torch.device) -> ICLRPhysicsPosterior:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ICLRPhysicsPosterior(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.inference_mode()
def information_scores(
    model: ICLRPhysicsPosterior,
    posterior,
    actions: torch.Tensor,
    observations: torch.Tensor,
    mask: torch.Tensor,
    geometry: torch.Tensor,
    candidates: torch.Tensor,
    *,
    samples: int,
    noise_std: float,
    generator: torch.Generator,
    candidate_chunk: int = 12,
) -> torch.Tensor:
    batch = actions.shape[0]
    joint = posterior.joint_probabilities.float()
    sampled_worlds = torch.multinomial(joint, samples, replacement=True, generator=generator)
    tuples = all_factor_tuples(device=actions.device)[sampled_worlds]
    row = torch.arange(batch, device=actions.device)[:, None].expand(-1, samples)
    mean = posterior.conditional_parameter_mean[row, sampled_worlds]
    std = posterior.conditional_parameter_std[row, sampled_worlds]
    normalized = (mean + std * torch.randn(mean.shape, device=mean.device, generator=generator)).clamp(0.0, 1.0)
    parameters = denormalize_parameters(normalized)
    world = ICLRContactWorld()
    scores = []
    prior_entropy = posterior.entropy().float()
    for start in range(0, len(candidates), candidate_chunk):
        candidate = candidates[start : start + candidate_chunk]
        count = len(candidate)
        fantasy_actions = candidate[None, :, None, :].expand(batch, -1, samples, -1)
        fantasy_observations = world.rollout(
            parameters[:, None].expand(-1, count, -1, -1),
            tuples[:, None].expand(-1, count, -1, -1),
            fantasy_actions,
            geometry[:, None, None].expand(-1, count, samples, -1),
        )
        fantasy_observations += noise_std * torch.randn(fantasy_observations.shape, device=actions.device, generator=generator)
        repeats = count * samples
        expanded_actions = actions[:, None].expand(-1, repeats, -1, -1).reshape(batch * repeats, actions.shape[1], 4).clone()
        expanded_observations = observations[:, None].expand(-1, repeats, -1, -1, -1).reshape(batch * repeats, observations.shape[1], world.observation_frames, 4).clone()
        expanded_mask = mask[:, None].expand(-1, repeats, -1).reshape(batch * repeats, mask.shape[1]).clone()
        available_slot = expanded_mask.long().sum(dim=-1)
        flat_index = torch.arange(batch * repeats, device=actions.device)
        expanded_actions[flat_index, available_slot] = fantasy_actions.reshape(-1, 4)
        expanded_observations[flat_index, available_slot] = fantasy_observations.reshape(-1, world.observation_frames, 4)
        expanded_mask[flat_index, available_slot] = True
        future = model(
            expanded_actions,
            expanded_observations,
            geometry[:, None].expand(-1, repeats, -1).reshape(batch * repeats, 4),
            expanded_mask,
        ).entropy().reshape(batch, count, samples).float().mean(dim=-1)
        energy = candidate[:, :2].square().sum(dim=-1) / 6.5**2
        scores.append(prior_entropy[:, None] - future - 0.004 * energy[None])
    return torch.cat(scores, dim=-1)


@torch.inference_mode()
def evaluate_strategy(
    model: ICLRPhysicsPosterior,
    tasks: ICLRTaskBatch,
    *,
    strategy: str,
    probes: int,
    posterior_samples: int,
    noise_std: float,
    mismatch: float,
    seed: int,
) -> dict[str, object]:
    device = tasks.parameters.device
    generator = torch.Generator(device=device).manual_seed(seed)
    bank = iclr_actions(device=device)
    directions = action_direction_indices(bank)
    batch = len(tasks.parameters)
    observations = torch.zeros((batch, probes + 1, 19, 4), device=device)
    actions = torch.zeros((batch, probes + 1, 4), device=device)
    mask = torch.zeros((batch, probes + 1), device=device, dtype=torch.bool)
    used = torch.zeros((batch, len(bank)), device=device, dtype=torch.bool)
    direction_counts = torch.zeros((batch, 12), device=device, dtype=torch.long)
    selected_history = []
    histories = []
    truth_world = ICLRContactWorld(mismatch=mismatch)
    started = time.perf_counter()
    fixed = torch.linspace(len(bank) - 1, len(bank) // 2, probes, device=device).long()
    balanced_directions = torch.tensor((0, 3, 6, 9, 1, 4, 7, 10), device=device)
    for probe in range(probes):
        posterior = model(actions, observations, tasks.geometry, mask)
        if strategy == "active":
            score = information_scores(model, posterior, actions, observations, mask, tasks.geometry, bank, samples=posterior_samples, noise_std=noise_std, generator=generator)
            selected = score.masked_fill(used, -torch.inf).argmax(dim=-1)
        elif strategy == "eig_matched":
            direction = int(balanced_directions[probe % len(balanced_directions)])
            subset = torch.where(directions == direction)[0]
            score = information_scores(model, posterior, actions, observations, mask, tasks.geometry, bank[subset], samples=posterior_samples, noise_std=noise_std, generator=generator)
            selected = subset[score.argmax(dim=-1)]
        elif strategy == "balanced_fixed":
            direction = int(balanced_directions[probe % len(balanced_directions)])
            selected = torch.full((batch,), direction * 12 + 7, device=device, dtype=torch.long)
        elif strategy == "adaptive_coverage":
            target_direction = direction_counts.argmin(dim=-1)
            selected = target_direction * 12 + 7
        elif strategy == "random":
            selected = torch.rand((batch, len(bank)), device=device, generator=generator).masked_fill(used, -torch.inf).argmax(dim=-1)
        elif strategy == "fixed":
            selected = fixed[probe].expand(batch)
        else:
            raise ValueError(strategy)
        used.scatter_(1, selected[:, None], True)
        chosen_direction = directions[selected]
        direction_counts.scatter_add_(1, chosen_direction[:, None], torch.ones((batch, 1), device=device, dtype=torch.long))
        chosen = bank[selected]
        observation = truth_world.observe(tasks, chosen, noise_std=noise_std, generator=generator)
        actions[:, probe] = chosen
        observations[:, probe] = observation
        mask[:, probe] = True
        selected_history.append(selected)
        metrics = posterior_metrics(model(actions, observations, tasks.geometry, mask), tasks.parameters, tasks.factors)
        histories.append({"probe": probe + 1, **{key: float(value.mean().cpu()) for key, value in metrics.items()}})
    final = posterior_metrics(model(actions, observations, tasks.geometry, mask), tasks.parameters, tasks.factors)
    torch.cuda.synchronize()
    return {
        "strategy": strategy,
        "history": histories,
        "summary": {key: float(value.mean().cpu()) for key, value in final.items()},
        "per_task": {key: value.cpu().tolist() for key, value in final.items()},
        "selected_action_indices": torch.stack(selected_history, dim=1).cpu().tolist(),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--probes", type=int, default=6)
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.006)
    parser.add_argument("--mismatch", type=float, default=0.0)
    parser.add_argument("--splits", nargs="+", default=("id", "composition_ood", "geometry_ood"))
    parser.add_argument("--strategies", nargs="+", default=("active", "eig_matched", "balanced_fixed", "adaptive_coverage", "random", "fixed"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ICLR evaluation requires CUDA")
    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    results = {}
    for split_index, split in enumerate(args.splits):
        generator = torch.Generator(device=device).manual_seed(8701 + split_index)
        tasks = sample_iclr_tasks(args.tasks, device=device, generator=generator, split=split)
        results[split] = {}
        for strategy_index, strategy in enumerate(args.strategies):
            print(f"evaluating {split}/{strategy}", flush=True)
            results[split][strategy] = evaluate_strategy(
                model,
                tasks,
                strategy=strategy,
                probes=args.probes,
                posterior_samples=args.posterior_samples,
                noise_std=args.noise_std,
                mismatch=args.mismatch,
                seed=9701 + split_index * 100 + strategy_index,
            )
            print(json.dumps(results[split][strategy]["summary"]), flush=True)
    payload = {
        "config": vars(args) | {"checkpoint": str(args.checkpoint), "output": str(args.output)},
        "results": results,
        "gpu": torch.cuda.get_device_name(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
