from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .amortized_world import (
    AmortizedContactWorld,
    intervention_actions,
    sample_tasks,
)


@torch.inference_mode()
def generate_split(
    *,
    output_directory: Path,
    split: str,
    tasks_count: int,
    shard_size: int,
    probes: int,
    noise_std: float,
    seed: int,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    world = AmortizedContactWorld()
    action_bank = intervention_actions(device="cuda")
    started = time.perf_counter()
    shard_paths: list[str] = []
    for start in range(0, tasks_count, shard_size):
        count = min(shard_size, tasks_count - start)
        tasks = sample_tasks(
            count, device="cuda", generator=generator, split=split
        )
        action_indices = torch.randint(
            len(action_bank),
            (count, probes),
            device="cuda",
            generator=generator,
        )
        actions = action_bank[action_indices]
        flat_parameters = tasks.parameters[:, None].expand(-1, probes, -1).reshape(-1, 4)
        flat_hypotheses = tasks.hypotheses[:, None].expand(-1, probes).reshape(-1)
        flat_geometry = tasks.geometry[:, None].expand(-1, probes, -1).reshape(-1, 4)
        clean = world.rollout(
            flat_parameters,
            flat_hypotheses,
            actions.reshape(-1, 4),
            flat_geometry,
        ).reshape(count, probes, world.observation_frames, 4)
        observations = clean + noise_std * torch.randn(
            clean.shape, device="cuda", generator=generator
        )
        shard = {
            "parameters": tasks.parameters.cpu(),
            "hypotheses": tasks.hypotheses.to(torch.int8).cpu(),
            "geometry": tasks.geometry.cpu(),
            "action_indices": action_indices.to(torch.int16).cpu(),
            "observations": observations.to(torch.float16).cpu(),
        }
        path = output_directory / f"{split}-{start // shard_size:04d}.pt"
        torch.save(shard, path)
        shard_paths.append(str(path))
        elapsed = time.perf_counter() - started
        print(
            f"{split}: {start + count}/{tasks_count} tasks; "
            f"{(start + count) / elapsed:.1f} tasks/s",
            flush=True,
        )
    return {
        "split": split,
        "tasks": tasks_count,
        "shard_size": shard_size,
        "probes": probes,
        "noise_std": noise_std,
        "seed": seed,
        "seconds": time.perf_counter() - started,
        "shards": shard_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tasks", type=int, default=131_072)
    parser.add_argument("--validation-tasks", type=int, default=16_384)
    parser.add_argument("--test-tasks", type=int, default=16_384)
    parser.add_argument("--shard-size", type=int, default=8_192)
    parser.add_argument("--probes", type=int, default=6)
    parser.add_argument("--noise-std", type=float, default=0.006)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Dataset generation requires CUDA")
    torch.set_float32_matmul_precision("high")
    manifests = []
    for split, count, seed in (
        ("train", args.train_tasks, 31001),
        ("id", args.validation_tasks, 41001),
        ("ood_thin", args.test_tasks, 51001),
        ("ood_round", args.test_tasks, 61001),
    ):
        manifests.append(
            generate_split(
                output_directory=args.output,
                split=split,
                tasks_count=count,
                shard_size=args.shard_size,
                probes=args.probes,
                noise_std=args.noise_std,
                seed=seed,
            )
        )
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
