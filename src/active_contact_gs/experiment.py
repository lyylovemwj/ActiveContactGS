from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .belief import PRIOR_HIGH, PRIOR_LOW, ParticleBelief
from .physics import PlanarRigidBodySimulator, default_actions


PARAMETER_NAMES = ("mass", "friction", "restitution", "inertia_scale")


def run_trial(
    *,
    strategy: str,
    seed: int,
    probes: int,
    particle_count: int,
    noise_std: float,
    device: str,
) -> dict:
    generator = torch.Generator(device=device).manual_seed(seed)
    simulator = PlanarRigidBodySimulator()
    actions = default_actions().to(device)
    prior_low = torch.tensor(PRIOR_LOW, device=device)
    prior_high = torch.tensor(PRIOR_HIGH, device=device)
    prior_span = prior_high - prior_low
    # Keep targets away from hard prior boundaries while covering the full range.
    truth = prior_low + prior_span * (0.05 + 0.90 * torch.rand(4, device=device, generator=generator))
    belief = ParticleBelief.from_uniform_prior(
        particle_count, device=device, generator=generator
    )

    history = []
    used_action_indices: set[int] = set()
    for probe in range(probes):
        if strategy == "active":
            scores = belief.action_scores(simulator, actions, noise_std=noise_std)
            if used_action_indices:
                used = torch.tensor(sorted(used_action_indices), device=device)
                scores[used] = -torch.inf
            action_index = int(torch.argmax(scores))
        elif strategy == "random":
            available = torch.tensor(
                [index for index in range(len(actions)) if index not in used_action_indices],
                device=device,
            )
            sampled = torch.randint(
                len(available), (1,), generator=generator, device=device
            )
            action_index = int(available[sampled])
        elif strategy == "fixed":
            action_index = len(actions) // 2
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        action = actions[action_index]
        used_action_indices.add(action_index)
        observation = simulator.observe(
            truth, action, noise_std=noise_std, generator=generator
        )
        belief.update(simulator, action, observation, noise_std=noise_std)
        estimate = belief.mean()
        history.append(
            {
                "probe": probe + 1,
                "action_index": action_index,
                "estimate": estimate.detach().cpu().tolist(),
                "absolute_error": torch.abs(estimate - truth).detach().cpu().tolist(),
                "posterior_std": belief.std().detach().cpu().tolist(),
                "effective_sample_size": float(belief.effective_sample_size().cpu()),
            }
        )
        belief.resample_if_needed(generator=generator)

    return {
        "strategy": strategy,
        "seed": seed,
        "truth": truth.cpu().tolist(),
        "parameter_names": PARAMETER_NAMES,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--probes", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--noise-std", type=float, default=0.008)
    parser.add_argument("--output", type=Path, default=Path("outputs/active_id.json"))
    args = parser.parse_args()

    if args.device != "cuda":
        raise ValueError("Project experiments are configured to run on the RTX 5090; use --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    results = []
    for strategy in ("active", "random", "fixed"):
        for seed in range(args.seeds):
            results.append(
                run_trial(
                    strategy=strategy,
                    seed=seed,
                    probes=args.probes,
                    particle_count=args.particles,
                    noise_std=args.noise_std,
                    device=args.device,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for strategy in ("active", "random", "fixed"):
        final = torch.tensor(
            [r["history"][-1]["absolute_error"] for r in results if r["strategy"] == strategy]
        )
        span = torch.tensor(PRIOR_HIGH) - torch.tensor(PRIOR_LOW)
        summary = ", ".join(
            f"{name}={value:.4f}" for name, value in zip(PARAMETER_NAMES, final.mean(dim=0))
        )
        normalized = (final / span).mean()
        print(f"{strategy:>6}: normalized={normalized:.4f}; {summary}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
