"""Quantify the interaction between geometry fidelity and active probing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPLITS = ("id", "ood_thin", "ood_round")
TITLES = {"id": "Held-out ID", "ood_thin": "OOD: thin", "ood_round": "OOD: near-round"}
MODELS = ("full", "isotropic", "no_shape")
LABELS = {"full": "Full anisotropic", "isotropic": "Isotropic", "no_shape": "No shape"}
COLORS = {"active": "#008B87", "random": "#EA8C1B"}


def bootstrap(values: np.ndarray, seed: int) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(20_000, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(samples > 0)),
    }


def task_values(payload: dict, split: str, strategy: str) -> np.ndarray:
    return np.asarray(
        payload["results"][split][strategy]["final_per_task"]["normalized_parameter_mae"],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--isotropic", type=Path, required=True)
    parser.add_argument("--no-shape", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"full": args.full, "isotropic": args.isotropic, "no_shape": args.no_shape}
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    args.output.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {"sources": {key: str(value) for key, value in paths.items()}, "results": {}, "interaction": {}}
    rows: list[tuple[object, ...]] = []
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.85), sharey=True, constrained_layout=True)
    x = np.arange(len(MODELS))
    width = 0.34
    for split_index, (split, axis) in enumerate(zip(SPLITS, axes)):
        summary["results"][split] = {}
        summary["interaction"][split] = {}
        for strategy_index, strategy in enumerate(("active", "random")):
            means = []
            for model in MODELS:
                values = task_values(payloads[model], split, strategy)
                mean = float(values.mean())
                means.append(mean)
                rows.append((split, model, strategy, mean, float(values.std(ddof=1))))
                summary["results"][split].setdefault(model, {})[strategy] = mean
            axis.bar(x + (strategy_index - 0.5) * width, 100 * np.asarray(means), width, color=COLORS[strategy], label=strategy.title())

        full_gain = np.log(task_values(payloads["full"], split, "random")) - np.log(
            task_values(payloads["full"], split, "active")
        )
        for model_index, model in enumerate(("isotropic", "no_shape")):
            ablated_gain = np.log(task_values(payloads[model], split, "random")) - np.log(
                task_values(payloads[model], split, "active")
            )
            summary["interaction"][split][model] = bootstrap(
                full_gain - ablated_gain,
                9100 + 10 * split_index + model_index,
            )
        axis.set_title(TITLES[split], fontweight="bold")
        axis.set_xticks(x, ("Full", "Isotropic", "No shape"), rotation=12)
        axis.set_xlabel("Geometry representation")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Parameter error after three probes (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    figure.suptitle("Geometry fidelity and active probing jointly improve identification", fontweight="bold", y=1.17)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(args.output / f"geometry_active_interaction.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)

    (args.output / "geometry_active_interaction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output / "geometry_active_interaction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "geometry", "strategy", "mean", "task_std"))
        writer.writerows(rows)
    print(json.dumps(summary["interaction"], indent=2))


if __name__ == "__main__":
    main()
