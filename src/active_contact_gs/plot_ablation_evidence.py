from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("full", "isotropic", "no_shape")
LABELS = {"full": "Full anisotropic", "isotropic": "Isotropic", "no_shape": "No shape"}
COLORS = {"full": "#008B87", "isotropic": "#6C4EA3", "no_shape": "#EA8C1B"}
SPLITS = ("id", "ood_thin", "ood_round")
SPLIT_LABELS = ("ID", "OOD thin", "OOD round")


def bootstrap_difference(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, float]:
    difference = right - left
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(difference), size=(20_000, len(difference)))
    samples = difference[indices].mean(axis=1)
    return {
        "mean_improvement": float(difference.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_full_better": float((samples > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--isotropic", type=Path, required=True)
    parser.add_argument("--no-shape", type=Path, required=True)
    parser.add_argument("--factorized", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"full": args.full, "isotropic": args.isotropic, "no_shape": args.no_shape}
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    metrics = ("normalized_parameter_mae", "inertia_error", "predictive_rollout_rmse")
    summary: dict[str, object] = {"sources": {name: str(path) for name, path in paths.items()}, "results": {}, "comparisons": {}}
    rows: list[tuple[object, ...]] = []
    for split_index, split in enumerate(SPLITS):
        summary["results"][split] = {}
        summary["comparisons"][split] = {}
        for model in MODELS:
            values = payloads[model]["results"][split]["active"]["final_per_task"]
            summary["results"][split][model] = {}
            for metric in metrics:
                array = np.asarray(values[metric], dtype=np.float64)
                summary["results"][split][model][metric] = float(array.mean())
                rows.append((split, model, metric, float(array.mean()), float(array.std(ddof=1))))
        for model_index, model in enumerate(("isotropic", "no_shape")):
            summary["comparisons"][split][model] = {}
            for metric_index, metric in enumerate(metrics):
                full_values = np.asarray(payloads["full"]["results"][split]["active"]["final_per_task"][metric])
                ablated_values = np.asarray(payloads[model]["results"][split]["active"]["final_per_task"][metric])
                summary["comparisons"][split][model][metric] = bootstrap_difference(
                    full_values, ablated_values, 3100 + 100 * split_index + 10 * model_index + metric_index
                )
    if args.factorized:
        factorized = json.loads(args.factorized.read_text(encoding="utf-8"))
        summary["factorized_source"] = str(args.factorized)
        summary["factorized"] = {}
        for split in SPLITS:
            summary["factorized"][split] = {
                metric: float(np.mean(factorized["results"][split]["active"]["final_per_task"][metric]))
                for metric in metrics
            }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "unified_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "unified_ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "model", "metric", "mean", "task_std"))
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.85), constrained_layout=True)
    titles = ("Parameter error", "Inertia error", "Predictive rollout")
    x = np.arange(len(SPLITS))
    width = 0.24
    for axis, metric, title in zip(axes, metrics, titles):
        for index, model in enumerate(MODELS):
            values = [summary["results"][split][model][metric] for split in SPLITS]
            axis.bar(x + (index - 1) * width, values, width, color=COLORS[model], label=LABELS[model])
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, SPLIT_LABELS)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Error after three active probes")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    figure.suptitle(
        "Anisotropic geometry improves physical identification under distribution shift",
        fontweight="bold",
        y=1.18,
    )
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(args.output / f"geometry_ablation.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary["comparisons"], indent=2))


if __name__ == "__main__":
    main()
