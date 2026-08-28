"""Aggregate 1--6 probe curves across independently trained models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPLITS = ("id", "ood_thin", "ood_round")
TITLES = {"id": "Held-out ID", "ood_thin": "OOD: thin", "ood_round": "OOD: near-round"}
STRATEGIES = ("active", "fixed", "random")
LABELS = {"active": "Active", "fixed": "Fixed", "random": "Random"}
COLORS = {"active": "#008B87", "fixed": "#7F7F7F", "random": "#EA8C1B"}


def hierarchical_ci(differences: np.ndarray, seed: int) -> dict[str, float]:
    """Bootstrap model seeds, then paired tasks within each sampled model."""
    generator = np.random.default_rng(seed)
    n_bootstrap = 20_000
    n_models, n_tasks = differences.shape
    model_indices = generator.integers(0, n_models, size=(n_bootstrap, n_models))
    task_indices = generator.integers(0, n_tasks, size=(n_bootstrap, n_models, n_tasks))
    selected = differences[model_indices[:, :, None], task_indices]
    samples = selected.mean(axis=(1, 2))
    return {
        "mean_improvement": float(differences.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_active_better": float(np.mean(samples > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.run]
    args.output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "sources": [str(path) for path in args.run],
        "model_seeds": len(payloads),
        "results": {},
        "comparisons": {},
    }
    rows: list[tuple[object, ...]] = []
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.75), sharey=True, constrained_layout=True)
    for split_index, (split, axis) in enumerate(zip(SPLITS, axes)):
        summary["results"][split] = {}
        summary["comparisons"][split] = {}
        probe_count = len(payloads[0]["results"][split]["active"]["history"])
        probes = np.arange(1, probe_count + 1)
        for strategy in STRATEGIES:
            values = np.asarray(
                [
                    [row["normalized_parameter_mae"] for row in payload["results"][split][strategy]["history"]]
                    for payload in payloads
                ],
                dtype=np.float64,
            )
            means = values.mean(axis=0)
            stds = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(means)
            summary["results"][split][strategy] = {
                "mean": means.tolist(),
                "model_seed_std": stds.tolist(),
            }
            for probe, mean, std in zip(probes, means, stds):
                rows.append((split, strategy, int(probe), float(mean), float(std)))
            axis.plot(probes, 100 * means, marker="o", color=COLORS[strategy], label=LABELS[strategy])
            axis.fill_between(probes, 100 * (means - stds), 100 * (means + stds), color=COLORS[strategy], alpha=0.13)

        for baseline_index, baseline in enumerate(("fixed", "random")):
            summary["comparisons"][split][baseline] = {}
            for probe_index, probe in enumerate(probes):
                active = np.asarray(
                    [payload["results"][split]["active"]["history"][probe_index]["per_task"]["normalized_parameter_mae"] for payload in payloads]
                )
                other = np.asarray(
                    [payload["results"][split][baseline]["history"][probe_index]["per_task"]["normalized_parameter_mae"] for payload in payloads]
                )
                summary["comparisons"][split][baseline][str(probe)] = hierarchical_ci(
                    other - active,
                    8200 + 100 * split_index + 10 * baseline_index + probe_index,
                )
        axis.set_title(TITLES[split], fontweight="bold")
        axis.set_xticks(probes)
        axis.set_xlabel("Physical interactions")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Normalized parameter error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    figure.suptitle("Active probing improves identification in the limited-interaction regime", fontweight="bold", y=1.17)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(args.output / f"probe_budget_curve.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)

    (args.output / "probe_budget_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "probe_budget_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "strategy", "probes", "mean", "model_seed_std"))
        writer.writerows(rows)
    print(json.dumps(summary["comparisons"], indent=2))


if __name__ == "__main__":
    main()
