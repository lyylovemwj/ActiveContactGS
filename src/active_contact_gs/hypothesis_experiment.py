from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .belief import PRIOR_HIGH, PRIOR_LOW
from .hypothesis_belief import ContactHypothesisBelief
from .hypothesis_physics import ContactHypothesisSimulator, hypothesis_actions


def run_hypothesis_trial(
    *, strategy: str, seed: int, probes: int, particles: int, noise_std: float
) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    simulator = ContactHypothesisSimulator()
    actions = hypothesis_actions().to("cuda")
    low = torch.tensor(PRIOR_LOW, device="cuda")
    high = torch.tensor(PRIOR_HIGH, device="cuda")
    truth = low + (high - low) * (
        0.05 + 0.9 * torch.rand(4, device="cuda", generator=generator)
    )
    truth_hypothesis = torch.randint(2, (), device="cuda", generator=generator)
    belief = ContactHypothesisBelief.from_prior(
        particles, device="cuda", generator=generator
    )
    used: set[int] = set()
    history = []
    for probe in range(probes):
        if strategy == "active":
            scores = belief.action_scores(simulator, actions, noise_std=noise_std)
            if used:
                scores[torch.tensor(sorted(used), device="cuda")] = -torch.inf
            action_index = int(scores.argmax())
        elif strategy == "random":
            available = torch.tensor(
                [index for index in range(len(actions)) if index not in used], device="cuda"
            )
            sampled = torch.randint(len(available), (), device="cuda", generator=generator)
            action_index = int(available[sampled])
        elif strategy == "fixed":
            action_index = len(actions) // 2
        else:
            raise ValueError(strategy)
        used.add(action_index)
        action = actions[action_index]
        observation = simulator.observe(
            truth,
            truth_hypothesis,
            action,
            noise_std=noise_std,
            generator=generator,
        )
        belief.update(simulator, action, observation, noise_std=noise_std)
        sphere_probability = belief.sphere_probability()
        true_probability = torch.where(
            truth_hypothesis == 1, sphere_probability, 1 - sphere_probability
        )
        estimate = belief.parameter_mean()
        history.append(
            {
                "probe": probe + 1,
                "action_index": action_index,
                "true_model_probability": float(true_probability.cpu()),
                "sphere_probability": float(sphere_probability.cpu()),
                "parameter_absolute_error": (estimate - truth).abs().cpu().tolist(),
                "effective_sample_size": float(belief.effective_sample_size().cpu()),
            }
        )
        belief.resample_if_needed(generator=generator)
    return {
        "strategy": strategy,
        "seed": seed,
        "truth": truth.cpu().tolist(),
        "truth_hypothesis": int(truth_hypothesis.cpu()),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--probes", type=int, default=6)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--noise-std", type=float, default=0.008)
    parser.add_argument("--output", type=Path, default=Path("outputs/hypothesis_id.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Hypothesis experiment requires CUDA")
    rows = [
        run_hypothesis_trial(
            strategy=strategy,
            seed=seed,
            probes=args.probes,
            particles=args.particles,
            noise_std=args.noise_std,
        )
        for strategy in ("active", "random", "fixed")
        for seed in range(args.seeds)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    span = torch.tensor(PRIOR_HIGH) - torch.tensor(PRIOR_LOW)
    for strategy in ("active", "random", "fixed"):
        selected = [row for row in rows if row["strategy"] == strategy]
        probability = torch.tensor([row["history"][-1]["true_model_probability"] for row in selected])
        error = torch.tensor([row["history"][-1]["parameter_absolute_error"] for row in selected])
        print(
            f"{strategy}: true_model_probability={probability.mean():.4f}; "
            f"model_accuracy={(probability > 0.5).float().mean():.4f}; "
            f"normalized_parameter_error={(error / span).mean():.4f}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
