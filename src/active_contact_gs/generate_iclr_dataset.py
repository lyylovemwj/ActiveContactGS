from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .iclr_world import ICLRContactWorld, iclr_actions, sample_iclr_tasks


@torch.inference_mode()
def generate_split(
    output: Path,
    split: str,
    count: int,
    *,
    probes: int,
    shard_size: int,
    noise_std: float,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    world = ICLRContactWorld()
    bank = iclr_actions(device="cuda")
    paths = []
    started = time.perf_counter()
    for start in range(0, count, shard_size):
        size = min(shard_size, count - start)
        tasks = sample_iclr_tasks(size, device="cuda", generator=generator, split=split)
        action_indices = torch.randint(len(bank), (size, probes), device="cuda", generator=generator)
        actions = bank[action_indices]
        observations = world.rollout(
            tasks.parameters[:, None].expand(-1, probes, -1),
            tasks.factors[:, None].expand(-1, probes, -1),
            actions,
            tasks.geometry[:, None].expand(-1, probes, -1),
        )
        observations += noise_std * torch.randn(observations.shape, device="cuda", generator=generator)
        payload = {
            "parameters": tasks.parameters.cpu(),
            "factors": tasks.factors.to(torch.int8).cpu(),
            "geometry": tasks.geometry.cpu(),
            "action_indices": action_indices.to(torch.int16).cpu(),
            "observations": observations.to(torch.float16).cpu(),
        }
        path = output / f"{split}-{start // shard_size:04d}.pt"
        torch.save(payload, path)
        paths.append(str(path))
        print(f"{split}: {start + size}/{count}", flush=True)
    return {"split": split, "tasks": count, "seed": seed, "seconds": time.perf_counter() - started, "shards": paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tasks", type=int, default=262_144)
    parser.add_argument("--validation-tasks", type=int, default=16_384)
    parser.add_argument("--test-tasks", type=int, default=16_384)
    parser.add_argument("--probes", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=8_192)
    parser.add_argument("--noise-std", type=float, default=0.006)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ICLR dataset generation requires CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, count, seed in (
        ("train", args.train_tasks, 31001),
        ("id", args.validation_tasks, 41001),
        ("composition_ood", args.test_tasks, 51001),
        ("geometry_ood", args.test_tasks, 61001),
    ):
        manifest.append(generate_split(args.output, split, count, probes=args.probes, shard_size=args.shard_size, noise_std=args.noise_std, seed=seed))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
