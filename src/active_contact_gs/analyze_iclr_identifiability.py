from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluate_iclr import evaluate_strategy, load_model
from .iclr_world import ICLRContactWorld, ICLRTaskBatch, iclr_actions, sample_iclr_tasks


@torch.inference_mode()
def trajectory_separation(aspect: float, *, device: torch.device) -> dict[str, float]:
    world = ICLRContactWorld()
    bank = iclr_actions(device=device)
    parameters = torch.tensor((1.0, 0.25, 0.65, 1.0), device=device)
    geometry = torch.tensor((0.14, 0.14 * aspect, 0.25, 0.60), device=device)
    pairs = {"native_rounded": (0, 1), "native_compliant": (0, 2), "rounded_compliant": (1, 2)}
    result = {}
    for name, (left, right) in pairs.items():
        left_factors = torch.zeros((len(bank), 5), device=device, dtype=torch.long)
        right_factors = left_factors.clone()
        right_factors[:, 0] = right
        left_factors[:, 0] = left
        left_rollout = world.rollout(parameters, left_factors, bank, geometry)
        right_rollout = world.rollout(parameters, right_factors, bank, geometry)
        distance = (left_rollout - right_rollout).square().mean(dim=(-1, -2))
        result[f"{name}_max_mse"] = float(distance.max().cpu())
        result[f"{name}_mean_mse"] = float(distance.mean().cpu())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--probes", type=int, default=6)
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--aspects", nargs="+", type=float, default=(0.3, 0.5, 0.7, 0.9, 0.95, 1.0))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Identifiability analysis requires CUDA")
    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    rows = []
    for index, aspect in enumerate(args.aspects):
        generator = torch.Generator(device=device).manual_seed(15001 + index)
        sampled = sample_iclr_tasks(args.tasks, device=device, generator=generator, split="id")
        semi_major = torch.full((args.tasks,), 0.14, device=device)
        geometry = torch.stack((semi_major, semi_major * aspect, sampled.geometry[:, 2], sampled.geometry[:, 3]), dim=-1)
        tasks = ICLRTaskBatch(sampled.parameters, sampled.factors, geometry)
        evaluation = evaluate_strategy(
            model,
            tasks,
            strategy="active",
            probes=args.probes,
            posterior_samples=args.posterior_samples,
            noise_std=0.006,
            mismatch=0.0,
            seed=16001 + index,
        )
        row = {"aspect_ratio": aspect, **trajectory_separation(aspect, device=device), **evaluation["summary"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    config = vars(args) | {
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
    }
    payload = {"checkpoint": str(args.checkpoint), "config": config, "rows": rows, "gpu": torch.cuda.get_device_name()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
