from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    "normalized_parameter_mae",
    "structure_correct",
    "predictive_rollout_rmse",
    "belief_control_error",
    "belief_control_success",
)


def task_bootstrap(
    values: np.ndarray, *, seed: int, draws: int = 20_000
) -> tuple[float, float]:
    """Bootstrap tasks after averaging matched tasks over model seeds."""
    task_values = values.mean(axis=0)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(task_values), size=(draws, len(task_values)))
    samples = task_values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, (0.025, 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        raise ValueError("at least two independently trained model evaluations are required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    distributions = tuple(payloads[0]["results"])
    strategies = tuple(payloads[0]["results"][distributions[0]])
    output: dict[str, object] = {
        "inputs": [str(path) for path in args.inputs],
        "model_seeds": len(payloads),
        "results": {},
        "comparisons": {},
    }
    csv_rows: list[tuple[object, ...]] = []
    for distribution_index, distribution in enumerate(distributions):
        output["results"][distribution] = {}
        output["comparisons"][distribution] = {}
        for strategy_index, strategy in enumerate(strategies):
            output["results"][distribution][strategy] = {}
            for metric_index, metric in enumerate(METRICS):
                values = np.asarray(
                    [
                        payload["results"][distribution][strategy]["final_per_task"][metric]
                        for payload in payloads
                    ],
                    dtype=np.float64,
                )
                seed_means = values.mean(axis=1)
                low, high = task_bootstrap(
                    values,
                    seed=20260826 + 100 * distribution_index + 10 * strategy_index + metric_index,
                )
                row = {
                    "mean": float(seed_means.mean()),
                    "model_seed_std": float(seed_means.std(ddof=1)),
                    "task_bootstrap_ci95_low": low,
                    "task_bootstrap_ci95_high": high,
                    "per_model_means": seed_means.tolist(),
                }
                output["results"][distribution][strategy][metric] = row
                for model_index, model_mean in enumerate(seed_means):
                    csv_rows.append(
                        (distribution, strategy, metric, model_index, float(model_mean))
                    )

        active = output["results"][distribution]["active"]
        for baseline in ("random", "fixed"):
            output["comparisons"][distribution][baseline] = {}
            for metric in METRICS:
                active_values = np.asarray(
                    [
                        payload["results"][distribution]["active"]["final_per_task"][metric]
                        for payload in payloads
                    ],
                    dtype=np.float64,
                )
                baseline_values = np.asarray(
                    [
                        payload["results"][distribution][baseline]["final_per_task"][metric]
                        for payload in payloads
                    ],
                    dtype=np.float64,
                )
                lower_is_better = metric not in {"structure_correct", "belief_control_success"}
                difference = (
                    baseline_values - active_values
                    if lower_is_better
                    else active_values - baseline_values
                )
                task_difference = difference.mean(axis=0)
                generator = np.random.default_rng(991 + distribution_index)
                indices = generator.integers(
                    0, len(task_difference), size=(20_000, len(task_difference))
                )
                bootstrap = task_difference[indices].mean(axis=1)
                output["comparisons"][distribution][baseline][metric] = {
                    "favorable_difference": float(task_difference.mean()),
                    "ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "probability_active_better": float((bootstrap > 0).mean()),
                }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "model_seed_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    with (args.output / "model_seed_means.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("distribution", "strategy", "metric", "model_index", "mean"))
        writer.writerows(csv_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
