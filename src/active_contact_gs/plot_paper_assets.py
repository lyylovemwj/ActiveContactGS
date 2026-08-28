from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

from .belief import PRIOR_HIGH, PRIOR_LOW
from .ellipsoid import ellipsoid_contact, shape_from_rotation_scale


COLORS = {
    "active": "#007F7B",
    "random": "#E58B25",
    "fixed": "#7A7A7A",
    "accent": "#B7355B",
    "dark": "#1D2A35",
    "light": "#E8F3F2",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(
            directory / f"{stem}.{suffix}",
            bbox_inches="tight",
            dpi=320 if suffix == "png" else None,
            transparent=False,
        )
    plt.close(figure)


def plot_active_identification(result: Path, directory: Path) -> None:
    rows = json.loads(result.read_text(encoding="utf-8"))
    span = torch.tensor(PRIOR_HIGH, device="cuda") - torch.tensor(PRIOR_LOW, device="cuda")
    figure, axis = plt.subplots(figsize=(3.45, 2.35))
    labels = {"active": "Ours: information gain", "random": "Random probes", "fixed": "Fixed probe"}
    generator = torch.Generator(device="cuda").manual_seed(20260826)
    for strategy in ("fixed", "random", "active"):
        selected = sorted((row for row in rows if row["strategy"] == strategy), key=lambda x: x["seed"])
        error = torch.tensor(
            [[item["absolute_error"] for item in row["history"]] for row in selected],
            device="cuda",
        )
        normalized = (error / span).mean(dim=-1)
        indices = torch.randint(
            normalized.shape[0],
            (30000, normalized.shape[0]),
            device="cuda",
            generator=generator,
        )
        bootstrap_mean = normalized[indices].mean(dim=1)
        interval = torch.quantile(
            bootstrap_mean,
            torch.tensor([0.025, 0.975], device="cuda"),
            dim=0,
        ).cpu().numpy()
        mean = normalized.mean(dim=0).cpu().numpy()
        x = np.arange(1, mean.size + 1)
        axis.fill_between(x, interval[0], interval[1], color=COLORS[strategy], alpha=0.15, linewidth=0)
        axis.plot(x, mean, marker="o", markersize=3.7, linewidth=2, color=COLORS[strategy], label=labels[strategy])
    axis.set(xlabel="Number of physical probes", ylabel="Normalized parameter error", xticks=np.arange(1, 7))
    axis.set_ylim(bottom=0)
    axis.legend(loc="upper right", fontsize=7.7)
    axis.set_title("Active probes identify physical twins faster", loc="left", fontweight="bold")
    save_figure(figure, directory, "active_identification")


def z_rotation(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device="cuda")


def plot_contact_geometry(directory: Path) -> None:
    center_a = torch.tensor([-0.55, -0.04, 0.0], device="cuda")
    center_b = torch.tensor([0.48, 0.14, 0.0], device="cuda")
    scale_a = torch.tensor([0.58, 0.20, 0.12], device="cuda")
    scale_b = torch.tensor([0.48, 0.16, 0.11], device="cuda")
    angle_a, angle_b = math.radians(24), math.radians(-31)
    shape_a = shape_from_rotation_scale(z_rotation(angle_a), scale_a)
    shape_b = shape_from_rotation_scale(z_rotation(angle_b), scale_b)
    contact = ellipsoid_contact(center_a, shape_a, center_b, shape_b)
    pa, pb, normal = [tensor.detach().cpu().numpy()[:2] for tensor in (contact.point_a, contact.point_b, contact.normal)]

    figure, axis = plt.subplots(figsize=(3.45, 2.35))
    for center, scale, angle, color in (
        (center_a, scale_a, angle_a, COLORS["active"]),
        (center_b, scale_b, angle_b, COLORS["accent"]),
    ):
        xy = center.detach().cpu().numpy()[:2]
        values = scale.detach().cpu().numpy()
        for level, alpha in ((2.0, 0.05), (1.5, 0.08), (1.0, 0.23)):
            axis.add_patch(
                Ellipse(
                    xy,
                    width=2 * values[0] * level,
                    height=2 * values[1] * level,
                    angle=math.degrees(angle),
                    facecolor=color,
                    edgecolor=color if level == 1 else "none",
                    linewidth=1.7 if level == 1 else 0,
                    alpha=alpha if level != 1 else 0.25,
                )
            )
        radius = float(np.prod(values) ** (1 / 3))
        axis.add_patch(Circle(xy, radius, fill=False, linestyle="--", linewidth=1.1, edgecolor=color, alpha=0.75))
    axis.plot([pa[0], pb[0]], [pa[1], pb[1]], color=COLORS["dark"], linewidth=2.2)
    axis.scatter([pa[0], pb[0]], [pa[1], pb[1]], s=25, color=COLORS["dark"], zorder=4)
    midpoint = (pa + pb) / 2
    axis.arrow(midpoint[0], midpoint[1], normal[0] * 0.22, normal[1] * 0.22, width=0.006, head_width=0.05, color=COLORS["dark"], length_includes_head=True)
    axis.text(midpoint[0] + 0.02, midpoint[1] + 0.11, r"signed gap $g$, normal $\mathbf{n}$", fontsize=8)
    axis.text(-0.92, 0.46, "native anisotropic Gaussian", color=COLORS["active"], fontsize=8, fontweight="bold")
    axis.text(0.36, -0.37, "sphere proxy", color=COLORS["accent"], fontsize=8)
    axis.set_aspect("equal")
    axis.set_xlim(-1.25, 1.18)
    axis.set_ylim(-0.62, 0.68)
    axis.axis("off")
    axis.set_title("Rendering geometry is collision geometry", loc="left", fontweight="bold")
    save_figure(figure, directory, "anisotropic_contact")


def plot_collision_accuracy(samples_path: Path, directory: Path) -> None:
    samples = torch.load(samples_path, map_location="cuda", weights_only=True)
    ours_mm = samples["ours_gap_error_m"].to("cuda") * 1000
    sphere_mm = samples["sphere_gap_error_m"].to("cuda") * 1000
    normal_deg = samples["ours_normal_error_deg"].to("cuda")
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    for values, label, color in (
        (ours_mm, "Ours: ellipsoid contact", COLORS["active"]),
        (sphere_mm, "Volume-equivalent sphere", COLORS["accent"]),
    ):
        sorted_values = values.sort().values.clamp_min(1e-6).cpu().numpy()
        probability = np.arange(1, sorted_values.size + 1) / sorted_values.size
        axes[0].plot(sorted_values, probability, linewidth=2, color=color, label=label)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Absolute signed-gap error (mm)", ylabel="Empirical CDF", ylim=(0, 1.01))
    axes[0].legend(loc="lower right", fontsize=7.7)
    axes[0].set_title("Geometry-faithful contact", loc="left", fontweight="bold")

    anisotropy = samples["anisotropy_ratio"].to("cuda")
    boundaries = [(1, 2), (2, 4), (4, 8), (8, 11)]
    centers, median, low, high = [], [], [], []
    for lower, upper in boundaries:
        selected = normal_deg[(anisotropy >= lower) & (anisotropy < upper)]
        q = torch.quantile(selected, torch.tensor([0.5, 0.95, 0.99], device="cuda"))
        centers.append(f"{lower}–{upper if upper < 11 else '10'}")
        median.append(float(q[0].cpu()))
        low.append(float(q[1].cpu()))
        high.append(float(q[2].cpu()))
    x = np.arange(len(centers))
    axes[1].plot(x, median, marker="o", linewidth=2, color=COLORS["active"], label="Median")
    axes[1].plot(x, low, marker="s", linewidth=1.7, color=COLORS["random"], label="P95")
    axes[1].plot(x, high, marker="^", linewidth=1.4, color=COLORS["accent"], label="P99")
    axes[1].set(xticks=x, xticklabels=centers, xlabel="Maximum axis ratio", ylabel="Normal error (degrees)")
    axes[1].legend(fontsize=7.7)
    axes[1].set_title("Stable under anisotropy", loc="left", fontweight="bold")
    figure.tight_layout(w_pad=2.0)
    save_figure(figure, directory, "collision_accuracy")


def plot_pipeline(directory: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 2.35))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3.25)
    axis.axis("off")
    boxes = [
        (0.10, 1.38, 1.72, 1.02, "RGB video\n& masks", "object-centric\nreconstruction"),
        (2.10, 1.38, 1.72, 1.02, "Gaussian\nellipsoids", "rendering =\ncontact geometry"),
        (4.10, 1.38, 1.72, 1.02, "Physical\nhypotheses", "geometry +\nparameters"),
        (6.10, 1.38, 1.72, 1.02, "Safe active\nprobe", "conditional\ninformation gain"),
        (8.10, 1.38, 1.72, 1.02, "Belief-aware\nMPC", "robust\nmanipulation"),
    ]
    for index, (x, y, width, height, title, subtitle) in enumerate(boxes):
        color = COLORS["active"] if index in (1, 3) else COLORS["dark"]
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor="white", edgecolor=color, linewidth=1.6)
        axis.add_patch(patch)
        axis.text(x + width / 2, y + 0.68, title, ha="center", va="center", fontsize=7.8, fontweight="bold", color=color, linespacing=0.92)
        axis.text(x + width / 2, y + 0.26, subtitle, ha="center", va="center", fontsize=6.4, color="#52606B", linespacing=0.95)
        if index < len(boxes) - 1:
            axis.add_patch(FancyArrowPatch((x + width, y + height / 2), (boxes[index + 1][0], y + height / 2), arrowstyle="-|>", mutation_scale=10, linewidth=1.3, color="#52606B"))
    axis.add_patch(FancyArrowPatch((8.95, 1.28), (4.95, 1.28), connectionstyle="arc3,rad=-0.42", arrowstyle="-|>", mutation_scale=11, linewidth=1.6, color=COLORS["accent"]))
    axis.text(6.95, 0.23, "observe motion  →  update posterior  →  eliminate physical twins", ha="center", color=COLORS["accent"], fontsize=7.8, fontweight="bold")
    axis.text(0.10, 3.02, "ContactSplat", fontsize=13.5, fontweight="bold", color=COLORS["dark"], va="top")
    axis.text(0.10, 2.68, "Geometry-faithful and identifiability-aware Gaussian physical twins", fontsize=8.4, color="#52606B", va="top")
    save_figure(figure, directory, "pipeline_overview")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identification", type=Path, default=Path("outputs/scale64.json"))
    parser.add_argument("--collision-samples", type=Path, default=Path("outputs/ellipsoid_assets.pt"))
    parser.add_argument("--output", type=Path, default=Path("paper_assets/figures"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Paper statistics and geometry generation require CUDA")
    configure_style()
    plot_pipeline(args.output)
    plot_active_identification(args.identification, args.output)
    plot_contact_geometry(args.output)
    plot_collision_accuracy(args.collision_samples, args.output)
    manifest = {
        "generated_on": torch.cuda.get_device_name(0),
        "source_identification": str(args.identification),
        "source_collision": str(args.collision_samples),
        "figures": sorted(path.name for path in args.output.glob("*")),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
