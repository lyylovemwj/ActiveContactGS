"""Summarize an official PIN-WM native Push-T identification run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUND_TRUTH = {
    "mass": 1.0,
    "friction_coefficient": 0.03,
    "restitution": 0.0,
    # Values returned by pybullet.getDynamicsInfo after loading the official
    # cube_t_target.urdf.  PyBullet computes the local diagonal from the mesh;
    # the raw inertia text in the URDF is therefore not the runtime truth.
    "inertia": [0.0020617075423587554, 0.0012130596914547934, 0.002740123401651869],
}


def load_trajectory(run_dir: Path) -> list[dict]:
    rows = []
    for path in run_dir.glob("iteration_*/physical_parameters_iter.json"):
        match = re.fullmatch(r"iteration_(\d+)", path.parent.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        bodies = {int(item["body_id"]): item for item in payload["activate"]}
        obj = bodies[1]
        inertia = np.diag(np.asarray(obj["inertia"], dtype=float)).tolist()
        rows.append(
            {
                "iteration": int(match.group(1)),
                "mass": float(obj["mass"]),
                "friction_coefficient": float(obj["friction_coefficient"]),
                "restitution": float(obj["restitution"]),
                "inertia_x": inertia[0],
                "inertia_y": inertia[1],
                "inertia_z": inertia[2],
            }
        )
    return sorted(rows, key=lambda row: row["iteration"])


def parse_training_log(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    matches = re.findall(r"loss=([0-9.eE+-]+), psnr=([0-9.eE+-]+)", text)
    values = [(float(loss), float(psnr)) for loss, psnr in matches]
    deduplicated = [value for index, value in enumerate(values) if index == 0 or value != values[index - 1]]
    return [
        {"optimizer_step": index, "loss": loss, "psnr": psnr}
        for index, (loss, psnr) in enumerate(deduplicated)
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], metrics: list[dict], output: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))
    it = [row["iteration"] for row in rows]
    if metrics:
        steps = [row["optimizer_step"] for row in metrics]
        axes[0, 0].plot(steps, [row["loss"] for row in metrics], color="#008c87")
        axes[0, 0].set_title("Official image-space objective")
        axes[0, 0].set_ylabel("Loss")
        twin = axes[0, 0].twinx()
        twin.plot(steps, [row["psnr"] for row in metrics], color="#6e4ca5", alpha=0.8)
        twin.set_ylabel("PSNR (dB)")

    axes[0, 1].plot(it, [row["friction_coefficient"] for row in rows], "o-", label="friction")
    axes[0, 1].axhline(GROUND_TRUTH["friction_coefficient"], ls="--", color="black", label="target")
    axes[0, 1].set_title("Object friction")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(it, [row["mass"] for row in rows], "o-", color="#ed8b00", label="mass")
    axes[1, 0].axhline(GROUND_TRUTH["mass"], ls="--", color="black", label="target")
    axes[1, 0].set_title("Object mass")
    axes[1, 0].set_xlabel("Optimization iteration")
    axes[1, 0].legend(frameon=False)

    for key, label in [("inertia_x", "Ixx"), ("inertia_y", "Iyy"), ("inertia_z", "Izz")]:
        axes[1, 1].plot(it, [row[key] for row in rows], "o-", label=label)
    for index, target in enumerate(GROUND_TRUTH["inertia"]):
        axes[1, 1].axhline(
            target,
            ls="--",
            color=("#444444", "#777777", "#AAAAAA")[index],
            linewidth=1.0,
            label="PyBullet target" if index == 0 else None,
        )
    axes[1, 1].set_title("Object inertia diagonal")
    axes[1, 1].set_xlabel("Optimization iteration")
    axes[1, 1].legend(frameon=False, ncol=2)

    fig.suptitle("PIN-WM official native Push-T reproduction", fontweight="bold")
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_trajectory(args.run_dir)
    if not rows:
        raise RuntimeError(f"No saved PIN-WM iterations under {args.run_dir}")
    metrics = parse_training_log(args.log_file)
    final = dict(rows[-1])
    final["absolute_error"] = {
        "mass": abs(final["mass"] - GROUND_TRUTH["mass"]),
        "friction_coefficient": abs(
            final["friction_coefficient"] - GROUND_TRUTH["friction_coefficient"]
        ),
        "restitution": abs(final["restitution"] - GROUND_TRUTH["restitution"]),
        "inertia_mean": float(np.mean([
            abs(final[key] - target)
            for key, target in zip(("inertia_x", "inertia_y", "inertia_z"), GROUND_TRUTH["inertia"])
        ])),
    }
    summary = {
        "protocol": "official PIN-WM native Push-T; headless launcher only",
        "ground_truth_from_pybullet_getDynamicsInfo": GROUND_TRUTH,
        "saved_iterations": len(rows),
        "logged_optimizer_updates": len(metrics),
        "final": final,
        "warning": "Native-domain reproduction; not a common-task head-to-head comparison.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.output.with_name(args.output.name + "_trajectory.csv"), rows)
    write_csv(args.output.with_name(args.output.name + "_metrics.csv"), metrics)
    plot(rows, metrics, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
