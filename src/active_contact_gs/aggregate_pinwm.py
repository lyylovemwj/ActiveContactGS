"""Aggregate official PIN-WM native Push-T reproductions across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TARGET = {
    "mass": 1.0,
    "friction_coefficient": 0.03,
    "restitution": 0.0,
    "inertia": np.asarray([0.0020617075423587554, 0.0012130596914547934, 0.002740123401651869]),
}


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def mean_std(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "seed_std": float(values.std(ddof=1))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summary]
    trajectories = [read_csv(path.with_name(path.stem + "_trajectory.csv")) for path in args.summary]
    metrics = [read_csv(path.with_name(path.stem + "_metrics.csv")) for path in args.summary]
    final = [summary["final"] for summary in summaries]

    fields = ("mass", "friction_coefficient", "restitution", "inertia_x", "inertia_y", "inertia_z")
    aggregate = {field: mean_std(np.asarray([row[field] for row in final])) for field in fields}
    error_fields = ("mass", "friction_coefficient", "restitution", "inertia_mean")
    errors = {
        field: mean_std(np.asarray([row["absolute_error"][field] for row in final]))
        for field in error_fields
    }
    final_loss = mean_std(np.asarray([rows[-1]["loss"] for rows in metrics]))
    final_psnr = mean_std(np.asarray([rows[-1]["psnr"] for rows in metrics]))
    payload = {
        "protocol": "official PIN-WM native Push-T reproduction; headless launcher only",
        "official_commit": "99d3fde5d233aeffabfa287f94831cf7c7afee64",
        "seeds": len(summaries),
        "target_from_pybullet_getDynamicsInfo": {
            "mass": TARGET["mass"],
            "friction_coefficient": TARGET["friction_coefficient"],
            "restitution": TARGET["restitution"],
            "inertia": TARGET["inertia"].tolist(),
        },
        "final_parameter_mean_and_seed_std": aggregate,
        "absolute_error_mean_and_seed_std": errors,
        "final_image_loss": final_loss,
        "final_psnr_db": final_psnr,
        "source_summaries": [str(path) for path in args.summary],
        "limitation": "Native-domain availability/provenance result, not a common-task head-to-head comparison.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("quantity", "mean", "seed_std"))
        for field, values in aggregate.items():
            writer.writerow((field, values["mean"], values["seed_std"]))
        for field, values in errors.items():
            writer.writerow((f"absolute_error_{field}", values["mean"], values["seed_std"]))
        writer.writerow(("final_image_loss", final_loss["mean"], final_loss["seed_std"]))
        writer.writerow(("final_psnr_db", final_psnr["mean"], final_psnr["seed_std"]))

    iterations = np.asarray([row["iteration"] for row in trajectories[0]])
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6), constrained_layout=True)
    specs = (
        (axes[0, 0], "mass", TARGET["mass"], "Mass"),
        (axes[0, 1], "friction_coefficient", TARGET["friction_coefficient"], "Friction"),
        (axes[1, 0], "restitution", TARGET["restitution"], "Restitution"),
    )
    for axis, field, target, title in specs:
        values = np.asarray([[row[field] for row in trajectory] for trajectory in trajectories])
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        axis.plot(iterations, mean, marker="o", color="#008B87", label="PIN-WM mean")
        axis.fill_between(iterations, mean - std, mean + std, color="#008B87", alpha=0.18, label="seed std")
        axis.axhline(target, color="#333333", linestyle="--", label="PyBullet target")
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Optimization iteration")
        axis.grid(alpha=0.2)
    inertia_errors = []
    for trajectory in trajectories:
        inertia = np.asarray([[row["inertia_x"], row["inertia_y"], row["inertia_z"]] for row in trajectory])
        inertia_errors.append(np.abs(inertia - TARGET["inertia"][None]).mean(axis=1))
    inertia_errors = np.asarray(inertia_errors)
    axes[1, 1].plot(iterations, inertia_errors.mean(axis=0), marker="o", color="#6C4EA3")
    axes[1, 1].fill_between(
        iterations,
        inertia_errors.mean(axis=0) - inertia_errors.std(axis=0, ddof=1),
        inertia_errors.mean(axis=0) + inertia_errors.std(axis=0, ddof=1),
        color="#6C4EA3",
        alpha=0.18,
    )
    axes[1, 1].set_title("Inertia mean absolute error", fontweight="bold")
    axes[1, 1].set_xlabel("Optimization iteration")
    axes[1, 1].grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Official PIN-WM native Push-T reproduction (3 seeds)", fontweight="bold", y=1.09)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(args.output.with_suffix(f".{suffix}"), dpi=260 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
