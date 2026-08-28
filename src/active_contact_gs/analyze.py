from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .belief import PRIOR_HIGH, PRIOR_LOW


def _bootstrap_interval(samples: torch.Tensor) -> dict[str, float]:
    """Return a labelled percentile interval instead of an ambiguous list."""
    quantiles = torch.quantile(
        samples,
        torch.tensor([0.025, 0.5, 0.975], device=samples.device, dtype=samples.dtype),
    )
    return {
        "lower": float(quantiles[0].cpu()),
        "median": float(quantiles[1].cpu()),
        "upper": float(quantiles[2].cpu()),
    }


def paired_comparison(
    active: torch.Tensor,
    baseline: torch.Tensor,
    *,
    bootstrap: int,
    generator: torch.Generator,
) -> dict[str, object]:
    """Paired metrics robust to individual errors close to zero.

    The previous per-instance relative gain divided every pair by its baseline
    error. A single nearly-perfect baseline could therefore dominate the mean.
    We report the ratio of aggregate means and bootstrap that statistic directly.
    """
    if active.shape != baseline.shape or active.ndim != 1:
        raise ValueError("active and baseline must be paired one-dimensional tensors")
    count = active.numel()
    difference = baseline - active
    indices = torch.randint(
        count, (bootstrap, count), device=active.device, generator=generator
    )
    active_boot = active[indices].mean(dim=1)
    baseline_boot = baseline[indices].mean(dim=1)
    difference_boot = difference[indices].mean(dim=1)
    relative_boot = difference_boot / baseline_boot.clamp_min(1e-12)
    difference_std = difference.std(unbiased=True).clamp_min(1e-12)
    return {
        "active_mean": float(active.mean().cpu()),
        "baseline_mean": float(baseline.mean().cpu()),
        "mean_paired_absolute_improvement": float(difference.mean().cpu()),
        "absolute_improvement_95_ci": _bootstrap_interval(difference_boot),
        "ratio_of_means_relative_improvement": float(
            (difference.mean() / baseline.mean().clamp_min(1e-12)).cpu()
        ),
        "relative_improvement_95_ci": _bootstrap_interval(relative_boot),
        "paired_effect_size_cohen_dz": float((difference.mean() / difference_std).cpu()),
        "bootstrap_probability_active_better": float((difference_boot > 0).float().mean().cpu()),
        "active_wins": int((active < baseline).sum().cpu()),
        "ties": int((active == baseline).sum().cpu()),
        "pairs": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Analysis is configured for the RTX 5090 and requires CUDA")
    rows = json.loads(args.result.read_text(encoding="utf-8"))
    span = (torch.tensor(PRIOR_HIGH, device="cuda") - torch.tensor(PRIOR_LOW, device="cuda"))
    strategies = sorted({row["strategy"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    probes = len(rows[0]["history"])

    curves = {}
    by_strategy = {}
    final_error = {}
    for strategy in strategies:
        selected = sorted((r for r in rows if r["strategy"] == strategy), key=lambda r: r["seed"])
        error = torch.tensor(
            [[h["absolute_error"] for h in r["history"]] for r in selected], device="cuda"
        )
        normalized = (error / span).mean(dim=-1)
        by_strategy[strategy] = normalized
        curves[strategy] = normalized.mean(dim=0).cpu().tolist()
        final = normalized[:, -1]
        final_error[strategy] = {
            "mean": float(final.mean().cpu()),
            "std": float(final.std(unbiased=True).cpu()),
            "median": float(final.median().cpu()),
        }

    generator = torch.Generator(device="cuda").manual_seed(20260826)
    active = by_strategy["active"][:, -1]
    comparisons = {}
    for baseline_name in ("random", "fixed"):
        baseline = by_strategy[baseline_name][:, -1]
        comparisons[baseline_name] = paired_comparison(
            active, baseline, bootstrap=args.bootstrap, generator=generator
        )

    summary = {
        "source": str(args.result),
        "seeds": len(seeds),
        "probes": probes,
        "mean_normalized_error_curve": curves,
        "final_normalized_error": final_error,
        "comparisons": comparisons,
        "gpu": torch.cuda.get_device_name(0),
    }
    output = args.output or args.result.with_name(args.result.stem + "_summary.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
