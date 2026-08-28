from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--identifiability", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    summary_rows: list[tuple[object, ...]] = []
    task_rows: list[tuple[object, ...]] = []
    comparison_rows: list[tuple[object, ...]] = []
    for distribution, strategies in evaluation["results"].items():
        for strategy, result in strategies.items():
            per_task = result["final_per_task"]
            metric_names = tuple(per_task)
            task_count = len(per_task[metric_names[0]])
            for metric in metric_names:
                values = np.asarray(per_task[metric], dtype=np.float64)
                summary_rows.append(
                    (
                        distribution,
                        strategy,
                        metric,
                        len(values),
                        float(values.mean()),
                        float(values.std(ddof=1)),
                        float(np.median(values)),
                        float(np.quantile(values, 0.05)),
                        float(np.quantile(values, 0.95)),
                    )
                )
            for task in range(task_count):
                task_rows.append(
                    (distribution, strategy, task, *(per_task[name][task] for name in metric_names))
                )
        for baseline, metrics in evaluation["comparisons"][distribution].items():
            for metric, values in metrics.items():
                comparison_rows.append(
                    (
                        distribution,
                        baseline,
                        metric,
                        values["active_mean"],
                        values["baseline_mean"],
                        values["relative_improvement"],
                        values["paired_absolute_improvement"],
                        values["ci95_low"],
                        values["ci95_high"],
                        values["probability_active_better"],
                        values["active_wins"],
                        values["pairs"],
                    )
                )
    first = next(iter(next(iter(evaluation["results"].values())).values()))
    task_metric_names = tuple(first["final_per_task"])
    write_csv(
        args.output / "evaluation_summary.csv",
        ("distribution", "strategy", "metric", "tasks", "mean", "std", "median", "q05", "q95"),
        summary_rows,
    )
    write_csv(
        args.output / "evaluation_per_task.csv",
        ("distribution", "strategy", "task", *task_metric_names),
        task_rows,
    )
    write_csv(
        args.output / "evaluation_comparisons.csv",
        (
            "distribution",
            "baseline",
            "metric",
            "active_mean",
            "baseline_mean",
            "relative_improvement",
            "paired_absolute_improvement",
            "ci95_low",
            "ci95_high",
            "probability_active_better",
            "active_wins",
            "pairs",
        ),
        comparison_rows,
    )

    input_paths = [args.evaluation]
    if args.identifiability:
        identifiability = json.loads(args.identifiability.read_text(encoding="utf-8"))
        ident_summary: list[tuple[object, ...]] = []
        ident_comparisons: list[tuple[object, ...]] = []
        for distribution, strategies in identifiability["results"].items():
            for strategy, metrics in strategies.items():
                for metric, values in metrics.items():
                    ident_summary.append(
                        (distribution, strategy, metric, values["mean"], values["median"])
                    )
        for distribution, baselines in identifiability["comparisons"].items():
            for baseline, metrics in baselines.items():
                for metric, values in metrics.items():
                    ident_comparisons.append(
                        (
                            distribution,
                            baseline,
                            metric,
                            values.get("mean_difference"),
                            values.get("ci95_low"),
                            values.get("ci95_high"),
                            values.get("probability_active_better"),
                        )
                    )
        write_csv(
            args.output / "identifiability_summary.csv",
            ("distribution", "strategy", "metric", "mean", "median"),
            ident_summary,
        )
        write_csv(
            args.output / "identifiability_comparisons.csv",
            (
                "distribution",
                "baseline",
                "metric",
                "mean_difference",
                "ci95_low",
                "ci95_high",
                "probability_active_better",
            ),
            ident_comparisons,
        )
        input_paths.append(args.identifiability)

    outputs = sorted(args.output.glob("*.csv"))
    manifest = {
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in input_paths],
        "outputs": [{"path": str(path), "sha256": sha256(path)} for path in outputs],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
