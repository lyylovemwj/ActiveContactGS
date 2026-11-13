from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "active": "#0072B2",
    "eig_matched": "#009E73",
    "balanced_fixed": "#E69F00",
    "adaptive_coverage": "#CC79A7",
    "random": "#6B7280",
    "fixed": "#D55E00",
}

REPRESENTATION_COLORS = {
    "flat": "#7A5195",
    "independent": "#00A6A6",
    "autoregressive": "#F28E2B",
    "structured": "#4E79A7",
}

REPRESENTATION_LABELS = {
    "flat": "Flat 243-way",
    "independent": "Independent",
    "autoregressive": "Autoregressive",
    "structured": "Structured joint",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_main(results: Path) -> tuple[list[dict[str, object]], dict]:
    rows = []
    payloads = {}
    pattern = re.compile(r"(flat|independent|autoregressive|structured)-seed(\d+)")
    for path in results.glob("*-seed*/evaluation.json"):
        match = pattern.fullmatch(path.parent.name)
        if not match:
            continue
        representation, seed = match.groups()
        payload = load(path)
        payloads[(representation, int(seed))] = payload
        for split, strategies in payload["results"].items():
            for strategy, result in strategies.items():
                rows.append(
                    {
                        "representation": representation,
                        "seed": int(seed),
                        "split": split,
                        "strategy": strategy,
                        **result["summary"],
                        "seconds": result["seconds"],
                    }
                )
    return rows, payloads


def representation_figure(rows: list[dict[str, object]], output: Path) -> None:
    representations = ("flat", "independent", "autoregressive", "structured")
    splits = ("id", "composition_ood", "geometry_ood")
    matrix = np.full((len(representations), len(splits)), np.nan)
    for i, representation in enumerate(representations):
        for j, split in enumerate(splits):
            values = [float(row["exact_accuracy"]) for row in rows if row["representation"] == representation and row["split"] == split and row["strategy"] == "active"]
            if values:
                matrix[i, j] = np.mean(values) * 100
    if np.isnan(matrix).all():
        return
    figure, axis = plt.subplots(figsize=(6.8, 3.5), layout="constrained")
    image = axis.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
    axis.set_xticks(range(len(splits)), ("ID", "Composition-\nOOD", "Geometry-\nOOD"))
    axis.set_yticks(range(len(representations)), ("Flat 243-way", "Independent", "Autoregressive", "Structured joint"))
    for i in range(len(representations)):
        for j in range(len(splits)):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white" if matrix[i, j] < 55 else "black", fontweight="bold")
    axis.set_title("Representation comparison under Active EIG", loc="left", fontweight="bold")
    figure.colorbar(image, ax=axis, label="Exact-world accuracy (%)", shrink=0.85)
    save(figure, output, "representation_comparison")


def policy_figure(rows: list[dict[str, object]], output: Path) -> None:
    strategies = ("active", "eig_matched", "balanced_fixed", "adaptive_coverage", "random", "fixed")
    splits = ("id", "composition_ood", "geometry_ood")
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True, layout="constrained")
    present = False
    for axis, split in zip(axes, splits):
        labels, means, errors, colors = [], [], [], []
        for strategy in strategies:
            values = [float(row["exact_accuracy"]) * 100 for row in rows if row["representation"] == "independent" and row["split"] == split and row["strategy"] == strategy]
            if values:
                present = True
                labels.append(strategy.replace("_", " "))
                means.append(np.mean(values))
                errors.append(np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0)
                colors.append(COLORS[strategy])
        axis.bar(range(len(means)), means, yerr=errors, color=colors, capsize=2)
        axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        axis.set_title(split.replace("_", "-").title(), fontweight="bold")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Exact-world accuracy (%)")
    figure.suptitle("Directional coverage controls", fontweight="bold")
    if present:
        save(figure, output, "coverage_controls")
    else:
        plt.close(figure)


def mismatch_figure(results: Path, output: Path) -> list[dict[str, object]]:
    rows = []
    for path in results.glob("mismatch-*.json"):
        payload = load(path)
        level = float(payload["config"]["mismatch"])
        for strategy, result in payload["results"]["id"].items():
            rows.append({"mismatch": level, "strategy": strategy, **result["summary"]})
    if not rows:
        return rows
    figure, axis = plt.subplots(figsize=(4.6, 3.0))
    for strategy in sorted({row["strategy"] for row in rows}):
        selected = sorted((row for row in rows if row["strategy"] == strategy), key=lambda row: row["mismatch"])
        axis.plot([100 * row["mismatch"] for row in selected], [100 * row["exact_accuracy"] for row in selected], marker="o", label=strategy.replace("_", " "), color=COLORS.get(strategy))
    axis.set(xlabel="Forward-model mismatch (%)", ylabel="Exact-world accuracy (%)", ylim=(0, 100))
    axis.set_title("Robustness to counterfactual mismatch", loc="left", fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    save(figure, output, "model_mismatch")
    return rows


def sample_figure(results: Path, output: Path) -> list[dict[str, object]]:
    rows = []
    for path in results.glob("samples-*.json"):
        payload = load(path)
        result = payload["results"]["id"]["active"]
        rows.append({"samples": int(payload["config"]["posterior_samples"]), **result["summary"], "seconds": result["seconds"]})
    rows.sort(key=lambda row: row["samples"])
    diagnostics_path = results / "sampling-diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = {row["samples"]: row for row in load(diagnostics_path)["rows"]}
        for row in rows:
            row.update(diagnostics.get(row["samples"], {}))
    if not rows:
        return rows
    figure, (left, diagnostic_axis) = plt.subplots(1, 2, figsize=(8.2, 3.0), layout="constrained")
    right = left.twinx()
    left.plot([row["samples"] for row in rows], [100 * row["exact_accuracy"] for row in rows], marker="o", color="#0072B2", label="Accuracy")
    right.plot([row["samples"] for row in rows], [row["seconds"] for row in rows], marker="s", color="#D55E00", label="Runtime")
    left.set(xlabel="Posterior worlds per candidate", ylabel="Exact-world accuracy (%)", xscale="log")
    left.set_xticks([row["samples"] for row in rows], [row["samples"] for row in rows])
    right.set_ylabel("Evaluation runtime (s)")
    left.grid(alpha=0.25)
    left.set_title("EIG Monte Carlo sensitivity", loc="left", fontweight="bold")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    if diagnostics_path.exists():
        diagnostic_axis.plot([row["samples"] for row in rows], [100 * row["top1_reference_agreement"] for row in rows], marker="o", color="#009E73", label="Top-1 agreement")
        variance_axis = diagnostic_axis.twinx()
        variance_axis.plot([row["samples"] for row in rows], [row["mean_score_variance"] for row in rows], marker="s", color="#CC79A7", label="EIG variance")
        diagnostic_axis.set(xlabel="Posterior worlds per candidate", ylabel="Agreement with 64-sample action (%)", xscale="log")
        diagnostic_axis.set_xticks([row["samples"] for row in rows], [row["samples"] for row in rows])
        variance_axis.set_ylabel("Mean EIG score variance")
        diagnostic_axis.grid(alpha=0.25)
        lines = diagnostic_axis.lines + variance_axis.lines
        diagnostic_axis.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    save(figure, output, "eig_sample_sensitivity")
    return rows


def identifiability_figure(results: Path, output: Path) -> list[dict[str, object]]:
    path = results / "identifiability.json"
    if not path.exists():
        return []
    rows = load(path)["rows"]
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), layout="constrained")
    x = [row["aspect_ratio"] for row in rows]
    entropy_axis = axes[0].twinx()
    axes[0].plot(x, [100 * row["exact_accuracy"] for row in rows], marker="o", label="Exact accuracy", color="#0072B2")
    entropy_axis.plot(x, [row["posterior_entropy"] for row in rows], marker="s", label="Posterior entropy", color="#E69F00")
    axes[0].set(xlabel="Aspect ratio b/a", ylabel="Exact accuracy (%)")
    entropy_axis.set_ylabel("Posterior entropy")
    lines = axes[0].lines + entropy_axis.lines
    axes[0].legend(lines, [line.get_label() for line in lines], frameon=False)
    for key, label in (("native_rounded_max_mse", "native-rounded"), ("native_compliant_max_mse", "native-compliant"), ("rounded_compliant_max_mse", "rounded-compliant")):
        axes[1].plot(x, [row[key] for row in rows], marker="o", label=label)
    trajectory_max = max(row[key] for row in rows for key in ("native_rounded_max_mse", "native_compliant_max_mse", "rounded_compliant_max_mse"))
    axes[1].set(xlabel="Aspect ratio b/a", ylabel="Max trajectory MSE", ylim=(0.0, 1.08 * trajectory_max))
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Near-isotropic identifiability boundary", fontweight="bold")
    save(figure, output, "identifiability_boundary")
    return rows


def action_selection_figure(payloads: dict, output: Path) -> list[dict[str, object]]:
    strategies = ("active", "eig_matched", "balanced_fixed", "random", "fixed")
    direction_rows = []
    figure = plt.figure(figsize=(9.4, 2.8), layout="constrained")
    theta = np.arange(12) * 2 * np.pi / 12
    for panel, strategy in enumerate(strategies, start=1):
        selected = []
        for (representation, _), payload in payloads.items():
            if representation != "autoregressive":
                continue
            result = payload["results"]["composition_ood"].get(strategy)
            if result:
                selected.extend(np.asarray(result["selected_action_indices"], dtype=int).ravel().tolist())
        axis = figure.add_subplot(1, len(strategies), panel, projection="polar")
        if selected:
            directions = np.asarray(selected) // 12
            counts = np.bincount(directions, minlength=12)
            frequency = 100 * counts / counts.sum()
            axis.bar(theta, frequency, width=2 * np.pi / 12 * 0.86, color=COLORS[strategy], alpha=0.88)
            for direction, (count, percent) in enumerate(zip(counts, frequency)):
                direction_rows.append(
                    {
                        "representation": "autoregressive",
                        "split": "composition_ood",
                        "strategy": strategy,
                        "direction_index": direction,
                        "count": int(count),
                        "percent": float(percent),
                    }
                )
        axis.set_theta_zero_location("E")
        axis.set_theta_direction(1)
        axis.set_xticks(np.arange(4) * np.pi / 2, ("+x", "+y", "-x", "-y"))
        axis.set_yticklabels([])
        axis.grid(alpha=0.25)
        axis.set_title(strategy.replace("_", " ").title(), fontsize=9, pad=10)
    figure.suptitle("Action-selection fingerprints on composition OOD", fontweight="bold")
    save(figure, output, "action_selection_fingerprints")
    return direction_rows


def per_task_distribution_figure(payloads: dict, output: Path) -> list[dict[str, object]]:
    representations = ("flat", "independent", "autoregressive", "structured")
    task_rows = []
    factor_values = []
    parameter_values = []
    for representation in representations:
        representation_factor = []
        representation_parameter = []
        for (name, seed), payload in payloads.items():
            if name != representation:
                continue
            result = payload["results"]["composition_ood"]["active"]["per_task"]
            count = len(result["factor_accuracy"])
            for index in range(count):
                row = {
                    "representation": representation,
                    "seed": seed,
                    "task_index": index,
                    "exact_accuracy": float(result["exact_accuracy"][index]),
                    "factor_accuracy": float(result["factor_accuracy"][index]),
                    "normalized_parameter_mae": float(result["normalized_parameter_mae"][index]),
                    "posterior_entropy": float(result["posterior_entropy"][index]),
                }
                task_rows.append(row)
                representation_factor.append(100 * row["factor_accuracy"])
                representation_parameter.append(100 * row["normalized_parameter_mae"])
        factor_values.append(representation_factor)
        parameter_values.append(representation_parameter)
    if not any(factor_values):
        return task_rows
    labels = [REPRESENTATION_LABELS[name] for name in representations]
    colors = [REPRESENTATION_COLORS[name] for name in representations]
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), layout="constrained")
    for axis, values, ylabel, title in (
        (axes[0], factor_values, "Correct factors per task (%)", "Factor recovery"),
        (axes[1], parameter_values, "Normalized parameter MAE (%)", "Continuous parameters"),
    ):
        violins = axis.violinplot(values, showmeans=False, showmedians=True, showextrema=False)
        for body, color in zip(violins["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("white")
            body.set_alpha(0.75)
        violins["cmedians"].set_color("black")
        axis.boxplot(values, widths=0.13, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "alpha": 0.8}, medianprops={"color": "black"})
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Per-task composition-OOD distributions under Active EIG", fontweight="bold")
    save(figure, output, "per_task_distributions")
    return task_rows


def uncertainty_failure_figure(task_rows: list[dict[str, object]], output: Path) -> None:
    if not task_rows:
        return
    representations = ("flat", "independent", "autoregressive", "structured")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, sharey=True, layout="constrained")
    last_hexbin = None
    for axis, representation in zip(axes.ravel(), representations):
        selected = [row for row in task_rows if row["representation"] == representation]
        entropy = np.asarray([row["posterior_entropy"] for row in selected])
        error = 100 * np.asarray([row["normalized_parameter_mae"] for row in selected])
        correct = np.asarray([row["exact_accuracy"] for row in selected], dtype=bool)
        last_hexbin = axis.hexbin(entropy, error, gridsize=18, mincnt=1, cmap="Blues", linewidths=0.2)
        axis.scatter(entropy[~correct], error[~correct], marker="x", s=12, color="#D55E00", alpha=0.45, label="wrong world")
        axis.set_title(f"{REPRESENTATION_LABELS[representation]}  |  {100 * correct.mean():.1f}%", fontweight="bold")
        axis.grid(alpha=0.12)
    for axis in axes[-1]:
        axis.set_xlabel("Posterior entropy")
    for axis in axes[:, 0]:
        axis.set_ylabel("Normalized parameter MAE (%)")
    axes[0, 0].legend(frameon=False, loc="upper right")
    if last_hexbin is not None:
        figure.colorbar(last_hexbin, ax=axes, label="Tasks per hexagon", shrink=0.78)
    figure.suptitle("Uncertainty-error landscape on held-out compositions", fontweight="bold")
    save(figure, output, "uncertainty_failure_landscape")


def physical_response_figure(output: Path) -> list[dict[str, object]]:
    import torch

    from .iclr_world import ICLRContactWorld, iclr_actions

    world = ICLRContactWorld()
    parameters = torch.tensor([1.05, 0.22, 0.72, 1.0], dtype=torch.float32)
    geometry = torch.tensor([0.14, 0.07, 0.35, 0.62], dtype=torch.float32)
    bank = iclr_actions()
    modes = {
        "native": torch.tensor([0, 0, 0, 0, 0]),
        "rounded": torch.tensor([1, 1, 1, 1, 0]),
        "compliant": torch.tensor([2, 2, 2, 2, 0]),
        "x-grain": torch.tensor([0, 0, 0, 0, 1]),
        "y-grain": torch.tensor([0, 0, 0, 0, 2]),
    }
    mode_colors = dict(zip(modes, ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#6B7280")))
    direction_indices = (1, 4, 7, 10)
    figure, axes = plt.subplots(1, 4, figsize=(9.2, 2.7), sharex=True, sharey=True, layout="constrained")
    rows = []
    for axis, direction in zip(axes, direction_indices):
        action = bank[direction * 12 + 11]
        for label, factors in modes.items():
            trajectory = world.rollout(parameters, factors, action, geometry).numpy()
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=mode_colors[label], linewidth=1.8, label=label)
            axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color=mode_colors[label], s=14, zorder=3)
            for frame, state in enumerate(trajectory):
                rows.append({"action_direction": direction, "mode": label, "frame": frame, "x": float(state[0]), "y": float(state[1]), "sin_angle": float(state[2]), "cos_angle": float(state[3])})
        arena = float(geometry[-1])
        axis.add_patch(plt.Rectangle((-arena, -arena), 2 * arena, 2 * arena, fill=False, color="#111827", linewidth=0.9))
        angle = direction * 30
        axis.set_title(f"Push {angle} deg", fontweight="bold")
        axis.set_aspect("equal")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("y position (m)")
    for axis in axes:
        axis.set_xlabel("x position (m)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle("Simulator response atlas across physical factor states", fontweight="bold")
    save(figure, output, "physical_response_atlas")
    return rows


def write_latex_tables(rows: list[dict[str, object]], output: Path) -> None:
    representations = ("flat", "independent", "autoregressive", "structured")
    splits = ("id", "composition_ood", "geometry_ood")
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Representation & ID & Composition-OOD & Geometry-OOD \\",
        r"\midrule",
    ]
    for representation in representations:
        cells = []
        for split in splits:
            values = np.asarray([100 * float(row["exact_accuracy"]) for row in rows if row["representation"] == representation and row["split"] == split and row["strategy"] == "active"])
            cells.append(f"{values.mean():.1f} $\\pm$ {values.std(ddof=1):.1f}" if len(values) > 1 else "--")
        lines.append(f"{REPRESENTATION_LABELS[representation]} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "main_results_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
    args.output.mkdir(parents=True, exist_ok=True)
    main_rows, payloads = collect_main(args.results)
    representation_figure(main_rows, args.output)
    policy_figure(main_rows, args.output)
    mismatch_rows = mismatch_figure(args.results, args.output)
    sample_rows = sample_figure(args.results, args.output)
    identifiability_rows = identifiability_figure(args.results, args.output)
    action_rows = action_selection_figure(payloads, args.output)
    task_rows = per_task_distribution_figure(payloads, args.output)
    uncertainty_failure_figure(task_rows, args.output)
    trajectory_rows = physical_response_figure(args.output)
    write_latex_tables(main_rows, args.output)
    write_csv(args.output / "representation_policy_results.csv", main_rows)
    write_csv(args.output / "model_mismatch_results.csv", mismatch_rows)
    write_csv(args.output / "eig_sample_results.csv", sample_rows)
    write_csv(args.output / "identifiability_results.csv", identifiability_rows)
    write_csv(args.output / "action_selection_results.csv", action_rows)
    write_csv(args.output / "per_task_results.csv", task_rows)
    write_csv(args.output / "physical_response_trajectories.csv", trajectory_rows)
    generated = sorted(
        path.name
        for path in args.output.iterdir()
        if path.is_file() and path.name not in {"README.md", "manifest.json"}
    )
    manifest = {
        "results_root": args.results.as_posix(),
        "main_runs": len(payloads),
        "generated": generated,
    }
    if (args.output / "README.md").is_file():
        manifest["documentation"] = "README.md"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
