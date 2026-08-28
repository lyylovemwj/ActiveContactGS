from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .amortized_world import (
    AmortizedContactWorld,
    denormalize_parameters,
    intervention_actions,
    normalize_parameters,
    sample_tasks,
)


@torch.inference_mode()
def finite_difference_jacobian(
    world: AmortizedContactWorld,
    normalized_parameters: torch.Tensor,
    relaxed_hypotheses: torch.Tensor,
    geometry: torch.Tensor,
    actions: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    batch, probes = actions.shape[:2]
    latent = torch.cat(
        (normalized_parameters, relaxed_hypotheses[:, None]), dim=-1
    )
    derivatives = []
    for dimension in range(5):
        plus = latent.clone()
        minus = latent.clone()
        plus[:, dimension] = (plus[:, dimension] + epsilon).clamp(0.001, 0.999)
        minus[:, dimension] = (minus[:, dimension] - epsilon).clamp(0.001, 0.999)
        denominator = (plus[:, dimension] - minus[:, dimension]).clamp_min(1e-7)

        def simulate(value: torch.Tensor) -> torch.Tensor:
            parameters = denormalize_parameters(value[:, :4])
            hypothesis = value[:, 4]
            trajectory = world.rollout(
                parameters[:, None].expand(-1, probes, -1).reshape(-1, 4),
                hypothesis[:, None].expand(-1, probes).reshape(-1),
                actions.reshape(-1, 4),
                geometry[:, None].expand(-1, probes, -1).reshape(-1, 4),
            ).reshape(batch, probes, world.observation_frames, 4)
            trajectory[..., :2] = (
                trajectory[..., :2] / geometry[:, None, None, 3:4]
            )
            return trajectory.flatten(1)

        derivatives.append(
            (simulate(plus) - simulate(minus)) / denominator[:, None]
        )
    return torch.stack(derivatives, dim=-1)


def bootstrap_difference(
    active: torch.Tensor, baseline: torch.Tensor, *, seed: int
) -> dict[str, float]:
    difference = (active - baseline).double().cpu().numpy()
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(difference), (20_000, len(difference)))
    samples = difference[indices].mean(axis=1)
    low, median, high = np.quantile(samples, (0.025, 0.5, 0.975))
    return {
        "active_mean": float(active.mean()),
        "baseline_mean": float(baseline.mean()),
        "paired_difference": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_median": float(median),
        "ci95_high": float(high),
        "probability_active_higher": float((samples > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--ridge", type=float, default=1e-5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Identifiability analysis requires CUDA")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    world = AmortizedContactWorld()
    action_bank = intervention_actions(device="cuda")
    results: dict[str, dict[str, dict[str, object]]] = {}
    comparisons: dict[str, dict[str, dict[str, object]]] = {}
    for split_index, split in enumerate(("id", "ood_thin", "ood_round")):
        task_count = evaluation["config"]["tasks"]
        generator = torch.Generator(device="cuda").manual_seed(8701 + split_index)
        tasks = sample_tasks(
            task_count, device="cuda", generator=generator, split=split
        )
        normalized = normalize_parameters(tasks.parameters)
        relaxed = tasks.hypotheses.float()
        results[split] = {}
        strategy_tensors: dict[str, dict[str, torch.Tensor]] = {}
        for strategy, row in evaluation["results"][split].items():
            indices = torch.tensor(
                row["selected_action_indices"], device="cuda", dtype=torch.long
            )
            actions = action_bank[indices]
            jacobian = finite_difference_jacobian(
                world,
                normalized,
                relaxed,
                tasks.geometry,
                actions,
                epsilon=args.epsilon,
            )
            fisher = jacobian.transpose(-1, -2) @ jacobian
            diagonal_scale = fisher.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-8)
            fisher = fisher / diagonal_scale[:, None, None]
            eigenvalues = torch.linalg.eigvalsh(
                fisher + args.ridge * torch.eye(5, device="cuda")
            )
            logdet = eigenvalues.log().sum(dim=-1)
            minimum = eigenvalues[:, 0]
            rank = (
                eigenvalues > 1e-3 * eigenvalues[:, -1:].clamp_min(1e-9)
            ).sum(dim=-1).float()
            strategy_tensors[strategy] = {
                "normalized_logdet": logdet,
                "minimum_eigenvalue": minimum,
                "identifiable_rank": rank,
            }
            results[split][strategy] = {
                key: {
                    "mean": float(value.mean()),
                    "median": float(value.median()),
                    "per_task": value.cpu().tolist(),
                }
                for key, value in strategy_tensors[strategy].items()
            }
        comparisons[split] = {}
        for strategy, metrics in strategy_tensors.items():
            if strategy == "active":
                continue
            comparisons[split][strategy] = {
                key: bootstrap_difference(
                    strategy_tensors["active"][key], value, seed=1201 + split_index
                )
                for key, value in metrics.items()
            }
    payload = {
        "evaluation": str(args.evaluation),
        "epsilon": args.epsilon,
        "ridge": args.ridge,
        "results": results,
        "comparisons": comparisons,
        "gpu": torch.cuda.get_device_name(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(comparisons, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
