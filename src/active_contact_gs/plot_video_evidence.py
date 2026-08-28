from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"active": "#008B87", "fixed": "#7F7F7F", "random": "#EA8C1B"}
LABELS = {"active": "Active video", "fixed": "Fixed video", "random": "Random video"}
TITLES = {"id": "Held-out ID", "ood_thin": "OOD: thin", "ood_round": "OOD: near-round"}


def series(payload: dict, split: str, strategy: str, metric: str) -> list[float]:
    return [float(row[metric]) for row in payload["results"][split][strategy]["history"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    video = json.loads(args.video.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    figure, axes = plt.subplots(2, 3, figsize=(9.2, 4.5), sharex=True, constrained_layout=True)
    probes = range(1, len(video["results"]["id"]["active"]["history"]) + 1)
    for column, split in enumerate(("id", "ood_thin", "ood_round")):
        for strategy in ("active", "fixed", "random"):
            axes[0, column].plot(
                probes,
                [100 * value for value in series(video, split, strategy, "normalized_parameter_mae")],
                color=COLORS[strategy],
                linewidth=2.0 if strategy == "active" else 1.5,
                marker="o",
                label=LABELS[strategy],
            )
            axes[1, column].plot(
                probes,
                [100 * value for value in series(video, split, strategy, "structure_correct")],
                color=COLORS[strategy],
                linewidth=2.0 if strategy == "active" else 1.5,
                marker="o",
                label=LABELS[strategy],
            )
        axes[0, column].plot(
            probes,
            [100 * value for value in series(state, split, "active", "normalized_parameter_mae")],
            color="#6C4EA3",
            linestyle="--",
            linewidth=1.4,
            marker="s",
            label="Active state reference",
        )
        axes[1, column].plot(
            probes,
            [100 * value for value in series(state, split, "active", "structure_correct")],
            color="#6C4EA3",
            linestyle="--",
            linewidth=1.4,
            marker="s",
            label="Active state reference",
        )
        axes[0, column].set_title(TITLES[split], fontweight="bold")
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
        axes[1, column].set_ylim(45, 102)
        axes[1, column].set_xticks(tuple(probes))
        axes[1, column].set_xlabel("Physical interactions")
    axes[0, 0].set_ylabel("Normalized parameter error (%)")
    axes[1, 0].set_ylabel("Structure accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Active identification remains effective from Gaussian-video observations",
        fontweight="bold",
        y=1.11,
    )
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            args.output / f"video_active_evidence.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    # Cleaner main-paper version: the identification objective is primary;
    # structure accuracy and the state/video gap remain visible in the full
    # supplementary figure above.
    main_figure, main_axes = plt.subplots(1, 3, figsize=(9.2, 2.7), sharey=True, constrained_layout=True)
    for column, split in enumerate(("id", "ood_thin", "ood_round")):
        for strategy in ("active", "fixed", "random"):
            main_axes[column].plot(
                probes,
                [100 * value for value in series(video, split, strategy, "normalized_parameter_mae")],
                color=COLORS[strategy],
                linewidth=2.0 if strategy == "active" else 1.5,
                marker="o",
                label=LABELS[strategy],
            )
        main_axes[column].plot(
            probes,
            [100 * value for value in series(state, split, "active", "normalized_parameter_mae")],
            color="#6C4EA3",
            linestyle="--",
            linewidth=1.4,
            marker="s",
            label="Active state reference",
        )
        main_axes[column].set_title(TITLES[split], fontweight="bold")
        main_axes[column].set_xticks(tuple(probes))
        main_axes[column].set_xlabel("Physical interactions")
        main_axes[column].grid(alpha=0.2)
    main_axes[0].set_ylabel("Normalized parameter error (%)")
    handles, labels = main_axes[0].get_legend_handles_labels()
    main_figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False)
    main_figure.suptitle(
        "Active probing remains effective with Gaussian-video observations",
        fontweight="bold",
        y=1.18,
    )
    for suffix in ("pdf", "svg", "png"):
        main_figure.savefig(
            args.output / f"video_active_evidence_main.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(main_figure)
    manifest = {
        "state": str(args.state),
        "video": str(args.video),
        "video_tasks": video["config"]["tasks"],
        "state_tasks": state["config"]["tasks"],
        "main_figure": "parameter error only",
        "supplementary_figure": "parameter error and structure accuracy",
    }
    (args.output / "video_active_evidence.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
