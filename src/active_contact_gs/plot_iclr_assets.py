from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch


COLORS = {
    "active": "#007F7B",
    "covariance": "#6A51A3",
    "variance": "#9E9AC8",
    "random": "#E58B25",
    "high_energy_random": "#D95F0E",
    "fixed": "#7A7A7A",
    "sphere": "#B7355B",
    "dark": "#1D2A35",
    "light": "#E8F3F2",
}
LABELS = {
    "active": "Ours: counterfactual EIG",
    "covariance": "Gaussian EIG",
    "variance": "Predictive variance",
    "random": "Random",
    "high_energy_random": "High-energy random",
    "fixed": "Fixed diverse",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.65,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(figure: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            output / f"{name}.{suffix}",
            bbox_inches="tight",
            dpi=320 if suffix == "png" else None,
        )
    plt.close(figure)


def plot_method(evaluation: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.15), sharey=True)
    strategies = ("active", "fixed", "covariance", "random")
    split_titles = {"id": "Held-out ID", "ood_thin": "OOD: thin", "ood_round": "OOD: near-round"}
    for axis, (split, title) in zip(axes, split_titles.items()):
        for strategy in strategies:
            history = evaluation["results"][split][strategy]["history"]
            x = [row["probe"] for row in history]
            y = [100 * row["normalized_parameter_mae"] for row in history]
            axis.plot(
                x,
                y,
                marker="o",
                markersize=3.2,
                linewidth=2.1 if strategy == "active" else 1.35,
                color=COLORS[strategy],
                label=LABELS[strategy],
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Physical interactions")
        axis.set_xticks((1, 2, 3))
        axis.set_ylim(0, 11)
    axes[0].set_ylabel("Normalized parameter error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=7.2)
    figure.tight_layout(w_pad=1.2)
    save(figure, output, "active_ood_curves")


def plot_downstream(evaluation: dict, output: Path) -> None:
    splits = ("id", "ood_thin", "ood_round")
    strategies = ("active", "fixed", "random", "covariance")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))
    width = 0.19
    x = np.arange(len(splits))
    for index, strategy in enumerate(strategies):
        rollout = []
        success = []
        for split in splits:
            final = evaluation["results"][split][strategy]["final_per_task"]
            rollout.append(np.mean(final["predictive_rollout_rmse"]))
            success.append(100 * np.mean(final["belief_control_success"]))
        offset = (index - (len(strategies) - 1) / 2) * width
        axes[0].bar(x + offset, rollout, width, color=COLORS[strategy], label=LABELS[strategy])
        axes[1].bar(x + offset, success, width, color=COLORS[strategy], label=LABELS[strategy])
    labels = ("ID", "OOD thin", "OOD round")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Long-horizon rollout RMSE")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Belief-control success (%)", ylim=(0, 80))
    axes[0].set_title("Prediction after three probes", loc="left", fontweight="bold")
    axes[1].set_title("Uncertainty-aware downstream control", loc="left", fontweight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        fontsize=7.0,
    )
    figure.tight_layout(w_pad=1.6)
    save(figure, output, "downstream_results")


def plot_identifiability(analysis: dict, output: Path) -> None:
    strategies = ("active", "fixed", "random", "covariance")
    splits = ("id", "ood_thin", "ood_round")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.25))
    width = 0.19
    x = np.arange(len(splits))
    for index, strategy in enumerate(strategies):
        logdet = [analysis["results"][split][strategy]["normalized_logdet"]["mean"] for split in splits]
        rank = [analysis["results"][split][strategy]["identifiable_rank"]["mean"] for split in splits]
        offset = (index - 1.5) * width
        axes[0].bar(x + offset, logdet, width, color=COLORS[strategy], label=LABELS[strategy])
        axes[1].bar(x + offset, rank, width, color=COLORS[strategy])
    axes[0].set(xticks=x, xticklabels=("ID", "Thin", "Round"), ylabel="Normalized FIM log determinant")
    axes[1].set(xticks=x, xticklabels=("ID", "Thin", "Round"), ylabel="Locally identifiable rank", ylim=(2.5, 3.8))
    axes[0].set_title("Active probes increase information", loc="left", fontweight="bold")
    axes[1].set_title("More latent directions become observable", loc="left", fontweight="bold")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        fontsize=7.0,
    )
    figure.tight_layout(w_pad=1.8)
    save(figure, output, "identifiability")


def plot_ycb(ycb_paths: list[Path], output: Path) -> None:
    ellipsoid_normal: list[float] = []
    sphere_normal: list[float] = []
    ellipsoid_gap: list[float] = []
    sphere_gap: list[float] = []
    adaptive_candidates: list[float] = []
    for path in ycb_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        method = "ellipsoid_2048" if "ellipsoid_2048" in data["records"] else "ellipsoid_512"
        ellipsoid_normal.extend(row["normal_error_deg"] for row in data["records"][method])
        sphere_normal.extend(row["normal_error_deg"] for row in data["records"]["sphere_512"])
        ellipsoid_gap.extend(1000 * row["gap_error_m"] for row in data["records"][method])
        sphere_gap.extend(1000 * row["gap_error_m"] for row in data["records"]["sphere_512"])
        adaptive_candidates.extend(row["candidate_pairs"] for row in data["records"].get("ellipsoid_adaptive", []))
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.15))
    for axis, ours, sphere, label in (
        (axes[0], ellipsoid_normal, sphere_normal, "Contact-normal error (degrees)"),
        (axes[1], ellipsoid_gap, sphere_gap, "Signed-gap error (mm)"),
    ):
        for values, name, color in (
            (ours, "Native anisotropic", COLORS["active"]),
            (sphere, "Area-matched spheres", COLORS["sphere"]),
        ):
            sorted_values = np.sort(np.maximum(values, 1e-5))
            probability = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            axis.plot(sorted_values, probability, linewidth=2, color=color, label=name)
        axis.set_xscale("log")
        axis.set(xlabel=label, ylabel="Empirical CDF", ylim=(0, 1.01))
    axes[0].legend(fontsize=6.8, loc="lower right")
    if adaptive_candidates:
        values, counts = np.unique(adaptive_candidates, return_counts=True)
        axes[2].bar(np.arange(len(values)), 100 * counts / counts.sum(), color=COLORS["active"])
        axes[2].set(xticks=np.arange(len(values)), xticklabels=[f"{int(v/1000)}k" for v in values], xlabel="Candidates until certificate", ylabel="Scenes (%)")
    axes[0].set_title("YCB contact normals", loc="left", fontweight="bold")
    axes[1].set_title("YCB separation distance", loc="left", fontweight="bold")
    axes[2].set_title("Adaptive global certificate", loc="left", fontweight="bold")
    figure.tight_layout(w_pad=1.5)
    save(figure, output, "ycb_contact_accuracy")


def plot_calibration(checkpoint_path: Path, output: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validation = checkpoint["validation"]
    probes = [0, 1, 2, 3, 6]
    accuracy = [100 * validation[str(p)]["structure_accuracy"] for p in probes]
    error = [100 * np.mean(validation[str(p)]["normalized_mae"]) for p in probes]
    coverage = [100 * np.mean(validation[str(p)]["coverage_90"]) for p in probes]
    figure, axis = plt.subplots(figsize=(3.45, 2.25))
    axis.plot(probes, accuracy, marker="o", linewidth=2, color=COLORS["active"], label="Structure accuracy")
    axis.plot(probes, coverage, marker="s", linewidth=1.8, color=COLORS["covariance"], label="90% posterior coverage")
    axis.plot(probes, error, marker="^", linewidth=1.8, color=COLORS["sphere"], label="Parameter error")
    axis.axhline(90, linestyle="--", color="#999999", linewidth=1)
    axis.set(xlabel="Random training-time probes", ylabel="Percent", xticks=probes, ylim=(0, 102))
    axis.set_title("Posterior confidence and coverage", loc="left", fontweight="bold")
    axis.legend(fontsize=7.0, loc="center right")
    save(figure, output, "posterior_calibration")


def plot_pipeline(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 2.15))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 3.2)
    axis.axis("off")
    boxes = (
        (0.1, "Gaussian inputs", "sparse video +\nanisotropic geometry"),
        (2.5, "Hybrid posterior", r"$q(h)q(\theta\mid h)$"),
        (4.9, "Counterfactuals", "simulate every\nsafe intervention"),
        (7.3, "Expected gain", "minimize future\njoint entropy"),
        (9.7, "Physical control", "belief-aware\nplanning"),
    )
    for index, (x, title, subtitle) in enumerate(boxes):
        color = COLORS["active"] if index in (1, 3) else COLORS["dark"]
        axis.add_patch(FancyBboxPatch((x, 1.25), 2.05, 1.05, boxstyle="round,pad=0.04,rounding_size=0.09", facecolor="white", edgecolor=color, linewidth=1.7))
        axis.text(x + 1.025, 1.92, title, ha="center", va="center", fontsize=7.2, fontweight="bold", color=color)
        axis.text(x + 1.025, 1.51, subtitle, ha="center", va="center", fontsize=6.2, color="#52606B")
        if index < len(boxes) - 1:
            axis.add_patch(FancyArrowPatch((x + 2.05, 1.78), (boxes[index + 1][0], 1.78), arrowstyle="-|>", mutation_scale=10, linewidth=1.3, color="#52606B"))
    axis.add_patch(FancyArrowPatch((10.7, 1.10), (3.5, 1.10), connectionstyle="arc3,rad=-0.20", arrowstyle="-|>", mutation_scale=11, linewidth=1.7, color=COLORS["sphere"]))
    axis.text(7.1, 0.08, "intervene  -  observe contact  -  update structure and parameters", ha="center", fontsize=7.6, color=COLORS["sphere"], fontweight="bold")
    axis.text(0.1, 2.92, "Contact Geometry Is a Latent Variable", fontsize=13, fontweight="bold", color=COLORS["dark"])
    save(figure, output, "iclr_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--identifiability", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ycb", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("paper_assets/iclr_figures"))
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    identifiability = json.loads(args.identifiability.read_text(encoding="utf-8"))
    configure()
    plot_pipeline(args.output)
    plot_method(evaluation, args.output)
    plot_downstream(evaluation, args.output)
    plot_identifiability(identifiability, args.output)
    plot_calibration(args.checkpoint, args.output)
    if args.ycb:
        plot_ycb(args.ycb, args.output)
    manifest = {
        "evaluation": str(args.evaluation),
        "identifiability": str(args.identifiability),
        "checkpoint": str(args.checkpoint),
        "ycb": [str(path) for path in args.ycb],
        "figures": sorted(path.name for path in args.output.glob("*")),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
