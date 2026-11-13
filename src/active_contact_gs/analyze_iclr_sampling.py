from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .evaluate_iclr import information_scores, load_model
from .iclr_world import iclr_actions, sample_iclr_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--samples", nargs="+", type=int, default=(2, 4, 8, 16, 32, 64))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling analysis requires CUDA")
    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(17001)
    tasks = sample_iclr_tasks(args.tasks, device=device, generator=generator, split="id")
    bank = iclr_actions(device=device)
    actions = torch.zeros((args.tasks, 1, 4), device=device)
    observations = torch.zeros((args.tasks, 1, 19, 4), device=device)
    mask = torch.zeros((args.tasks, 1), device=device, dtype=torch.bool)
    posterior = model(actions, observations, tasks.geometry, mask)
    raw = {}
    for samples in args.samples:
        scores = []
        seconds = []
        for repeat in range(args.repeats):
            repeat_generator = torch.Generator(device=device).manual_seed(18001 + samples * 101 + repeat)
            torch.cuda.synchronize()
            started = time.perf_counter()
            score = information_scores(
                model,
                posterior,
                actions,
                observations,
                mask,
                tasks.geometry,
                bank,
                samples=samples,
                noise_std=0.006,
                generator=repeat_generator,
            )
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
            scores.append(score.cpu())
        raw[samples] = torch.stack(scores)
        print(f"samples={samples}: {sum(seconds) / len(seconds):.3f}s/acquisition", flush=True)
        raw[(samples, "seconds")] = seconds
    reference_samples = max(args.samples)
    reference = raw[reference_samples].mean(dim=0).argmax(dim=-1)
    rows = []
    for samples in args.samples:
        score = raw[samples]
        selections = score.argmax(dim=-1)
        agreement = (selections == reference[None]).float()
        rows.append(
            {
                "samples": samples,
                "mean_score_variance": float(score.var(dim=0, unbiased=True).mean()),
                "top1_reference_agreement": float(agreement.mean()),
                "seconds_per_acquisition": float(sum(raw[(samples, "seconds")]) / args.repeats),
                "repeats": args.repeats,
                "tasks": args.tasks,
            }
        )
    payload = {
        "checkpoint": str(args.checkpoint),
        "reference_samples": reference_samples,
        "rows": rows,
        "gpu": torch.cuda.get_device_name(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
