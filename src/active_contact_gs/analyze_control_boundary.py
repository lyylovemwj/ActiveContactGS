from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"active": "#0072B2", "fixed": "#D55E00", "random": "#777777"}
LABELS = {"active": "Counterfactual EIG", "fixed": "Fixed diverse", "random": "Random"}


def bootstrap_mean_difference(
    left: np.ndarray, right: np.ndarray, *, seed: int = 20260826, draws: int = 20_000
) -> dict[str, float]:
    generator = np.random.default_rng(seed)
    difference = left - right
    indices = generator.integers(0, len(difference), size=(draws, len(difference)))
    samples = difference[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_difference": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_positive": float((samples > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distribution", default="ood_thin")
    parser.add_argument("--threshold", type=float, default=0.08)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = payload["results"][args.distribution]
    strategies = ("active", "fixed", "random")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    summary: dict[str, object] = {
        "source": str(args.input),
        "distribution": args.distribution,
        "success_threshold": args.threshold,
        "strategies": {},
        "paired_comparisons": {},
    }
    for strategy in strategies:
        per_task = results[strategy]["final_per_task"]
        rollout = np.asarray(per_task["predictive_rollout_rmse"], dtype=np.float64)
        control = np.asarray(per_task["belief_control_error"], dtype=np.float64)
        parameter = np.asarray(per_task["normalized_parameter_mae"], dtype=np.float64)
        arrays[strategy] = {"rollout": rollout, "control": control, "parameter": parameter}
        summary["strategies"][strategy] = {
            "tasks": int(len(control)),
            "parameter_error_mean": float(parameter.mean()),
            "rollout_error_mean": float(rollout.mean()),
            "control_error_mean": float(control.mean()),
            "control_error_median": float(np.median(control)),
            "control_error_q25": float(np.quantile(control, 0.25)),
            "control_error_q75": float(np.quantile(control, 0.75)),
            "control_success_rate": float((control < args.threshold).mean()),
            "rollout_control_correlation": float(np.corrcoef(rollout, control)[0, 1]),
            "parameter_control_correlation": float(np.corrcoef(parameter, control)[0, 1]),
        }

    for baseline in ("fixed", "random"):
        summary["paired_comparisons"][f"{baseline}_minus_active_control_error"] = (
            bootstrap_mean_difference(
                arrays[baseline]["control"], arrays["active"]["control"]
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ood_thin_control_boundary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output / "ood_thin_control_per_task.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("strategy", "task", "parameter_error", "rollout_error", "control_error"))
        for strategy in strategies:
            for task, values in enumerate(
                zip(
                    arrays[strategy]["parameter"],
                    arrays[strategy]["rollout"],
                    arrays[strategy]["control"],
                )
            ):
                writer.writerow((strategy, task, *values))

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.65), constrained_layout=True)
    for strategy in strategies:
        rollout = arrays[strategy]["rollout"]
        control = arrays[strategy]["control"]
        axes[0].scatter(
            rollout,
            control,
            s=11,
            alpha=0.42,
            color=COLORS[strategy],
            edgecolors="none",
            label=LABELS[strategy],
        )
        order = np.argsort(control)
        axes[1].step(
            control[order],
            (np.arange(len(control)) + 1) / len(control),
            where="post",
            color=COLORS[strategy],
            linewidth=1.8,
            label=LABELS[strategy],
        )
    axes[0].axhline(args.threshold, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("A. Continuous errors")
    axes[0].set_xlabel("Predictive rollout RMSE")
    axes[0].set_ylabel("Belief-control error")
    axes[0].grid(alpha=0.2)

    axes[1].axvline(args.threshold, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("B. Control-error ECDF")
    axes[1].set_xlabel("Belief-control error")
    axes[1].set_ylabel("Fraction of tasks")
    axes[1].set_xlim(left=0)
    axes[1].grid(alpha=0.2)

    thresholds = np.linspace(0.0, 0.6, 241)
    for strategy in strategies:
        success = (arrays[strategy]["control"][:, None] < thresholds[None]).mean(axis=0)
        axes[2].plot(
            thresholds,
            success,
            color=COLORS[strategy],
            linewidth=1.8,
            label=LABELS[strategy],
        )
    axes[2].axvline(args.threshold, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_title("C. Threshold sensitivity")
    axes[2].set_xlabel("Success threshold")
    axes[2].set_ylabel("Success rate")
    axes[2].grid(alpha=0.2)
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")

    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            args.output / f"ood_thin_control_boundary.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
