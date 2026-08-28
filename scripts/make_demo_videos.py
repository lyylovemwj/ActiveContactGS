"""Generate qualitative trajectory GIFs: Active vs Random vs Fixed probing.

For a small number of tasks, run the three probing strategies (3 interactions
each), record the true physical trajectories and the posterior evolution, then
render one GIF per task showing the object motion in the arena.

Usage (run from the repo root, with the environment that can import the package):

    python scripts/make_demo_videos.py \
        --checkpoint outputs/amortized-mixture-v2/checkpoint-final.pt \
        --output outputs/demo-videos --tasks 2 --probes 3 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable regardless of how this script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from active_contact_gs.amortized_world import (
    AmortizedContactWorld,
    intervention_actions,
    sample_tasks,
)
from active_contact_gs.evaluate_amortized import (
    amortized_information_scores,
    load_model,
    task_metrics,
)

STRATEGY_COLORS = {"active": "#008B87", "random": "#EA8C1B", "fixed": "#7F7F7F"}


@torch.inference_mode()
def run_protocol(
    model: torch.nn.Module,
    tasks,
    strategy: str,
    probes: int,
    noise_std: float,
    posterior_samples: int,
    seed: int,
    device: torch.device,
    action_bank: torch.Tensor,
    world: AmortizedContactWorld,
) -> dict:
    """One task × one strategy. Mirrors evaluate_amortized.evaluate_strategy."""
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = len(tasks.parameters)
    actions_sequence = torch.zeros((batch, probes, 4), device=device)
    observations = torch.zeros(
        (batch, probes, world.observation_frames, 4), device=device
    )
    mask = torch.zeros((batch, probes), device=device, dtype=torch.bool)
    used = torch.zeros((batch, len(action_bank)), device=device, dtype=torch.bool)
    fixed_order = torch.linspace(
        len(action_bank) - 1, len(action_bank) // 2, probes, device=device
    ).long()
    trajectories: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    param_errors: list[torch.Tensor] = []
    selected_indices: list[torch.Tensor] = []
    for probe in range(probes):
        posterior = model(actions_sequence, observations, tasks.geometry, mask)
        if strategy == "active":
            scores = amortized_information_scores(
                model,
                posterior,
                actions_sequence,
                observations,
                mask,
                tasks.geometry,
                action_bank,
                world,
                fantasy_samples=min(posterior_samples, 8),
                noise_std=noise_std,
                generator=generator,
                action_chunk=8,
            )
            scores = scores.masked_fill(used, -torch.inf)
            selected = scores.argmax(dim=-1)
        elif strategy == "random":
            scores = torch.rand((batch, len(action_bank)), device=device, generator=generator)
            scores = scores.masked_fill(used, -torch.inf)
            selected = scores.argmax(dim=-1)
        else:  # fixed
            selected = fixed_order[probe].expand(batch)
        used.scatter_(1, selected[:, None], True)
        selected_indices.append(selected)
        chosen = action_bank[selected]
        observation = world.observe(tasks, chosen, noise_std=noise_std, generator=generator)
        actions_sequence[:, probe] = chosen
        observations[:, probe] = observation
        mask[:, probe] = True
        updated = model(actions_sequence, observations, tasks.geometry, mask)
        traj = world.rollout(tasks.parameters, tasks.hypotheses, chosen, tasks.geometry)
        trajectories.append(traj)
        probs.append(updated.hypothesis_probabilities)
        param_errors.append(task_metrics(updated, tasks)["normalized_parameter_mae"])
    return {
        "trajectories": trajectories,
        "probs": probs,
        "param_errors": param_errors,
        "selected_indices": selected_indices,
    }


def render_gif(
    task_index: int,
    geometry: torch.Tensor,
    hypotheses: torch.Tensor,
    protocols: dict[str, dict],
    probes: int,
    out_path: Path,
) -> None:
    """One GIF: three stacked panels (Active / Random / Fixed), object motion."""
    semi_major = geometry[task_index, 0].item()
    semi_minor = geometry[task_index, 1].item()
    arena = geometry[task_index, 3].item()
    truth_h = int(hypotheses[task_index].item())

    # Concatenate the observation frames of every interaction for each strategy.
    frames_per_interaction = protocols["active"]["trajectories"][0].shape[1]
    frame_schedule = []  # list of (strategy, interaction, frame)
    for strategy in ("active", "random", "fixed"):
        for interaction in range(probes):
            for frame in range(frames_per_interaction):
                frame_schedule.append((strategy, interaction, frame))
    total_frames = len(frame_schedule)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.6), constrained_layout=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    ellipses, trails, labels, titles = {}, {}, {}, {}

    for ax, strategy in zip(axes, ("active", "random", "fixed")):
        ax.set_xlim(-arena * 1.18, arena * 1.18)
        ax.set_ylim(-arena * 1.18, arena * 1.18)
        ax.set_aspect("equal")
        ax.add_patch(
            patches.Rectangle(
                (-arena, -arena), 2 * arena, 2 * arena,
                fill=False, edgecolor="#9aa7b4", linestyle="--", linewidth=1.0,
            )
        )
        ax.set_title(
            f"{strategy.capitalize()} (structure truth: {'sphere' if truth_h else 'ellipse'})",
            color=STRATEGY_COLORS[strategy], fontsize=11, fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ellipses[strategy] = patches.Ellipse(
            (0.0, 0.0), 2 * semi_major, 2 * semi_minor,
            fill=True, facecolor=STRATEGY_COLORS[strategy], alpha=0.55,
            edgecolor="black", linewidth=0.8,
        )
        ax.add_patch(ellipses[strategy])
        (trails[strategy],) = ax.plot([], [], color=STRATEGY_COLORS[strategy], linewidth=1.2, alpha=0.75)
        (labels[strategy],) = ax.plot([], [], alpha=0)
        titles[strategy] = ax.text(
            0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
            va="top", ha="left", color="#1c2733",
        )

    def init() -> list:
        for strategy in ("active", "random", "fixed"):
            trails[strategy].set_data([], [])
            labels[strategy].set_data([], [])
            titles[strategy].set_text("")
        return [trails[s] for s in trails] + [titles[s] for s in titles]

    def update(frame_index: int) -> list:
        strategy, interaction, frame = frame_schedule[frame_index]
        # Each protocol only contains a single task (batch == 1).
        traj = protocols[strategy]["trajectories"][interaction][0]  # [obs_frames, 4]
        x, y = traj[frame, 0].item(), traj[frame, 1].item()
        angle = float(np.arctan2(traj[frame, 2].item(), traj[frame, 3].item()))
        ellipses[strategy].set_center((x, y))
        ellipses[strategy].angle = np.degrees(angle)
        # trail up to current frame
        trail_x = traj[: frame + 1, 0].cpu().numpy()
        trail_y = traj[: frame + 1, 1].cpu().numpy()
        trails[strategy].set_data(trail_x, trail_y)
        prob = protocols[strategy]["probs"][interaction][0].cpu().numpy()
        perr = protocols[strategy]["param_errors"][interaction][0].item()
        titles[strategy].set_text(
            f"interaction {interaction + 1}/{probes}  ·  P(sphere)={prob[0]:.2f}  ·  param err={perr:.4f}"
        )
        return [trails[s] for s in trails] + [titles[s] for s in titles]

    anim = FuncAnimation(
        fig, update, frames=total_frames, init_func=init, interval=90, blit=False
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=12)
    anim.save(out_path, writer=writer, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo-videos"))
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--probes", type=int, default=3)
    parser.add_argument("--posterior-samples", type=int, default=12)
    parser.add_argument("--noise-std", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    model = load_model(args.checkpoint, device)
    world = AmortizedContactWorld()
    action_bank = intervention_actions(device=device)

    tasks = sample_tasks(args.tasks, device=device, generator=torch.Generator(device=device).manual_seed(args.seed), split="id")
    manifest = []
    for task_index in range(args.tasks):
        single = type(tasks)(
            tasks.parameters[task_index : task_index + 1],
            tasks.hypotheses[task_index : task_index + 1],
            tasks.geometry[task_index : task_index + 1],
        )
        protocols = {}
        for strategy in ("active", "random", "fixed"):
            seed = args.seed + task_index * 10 + {"active": 1, "random": 2, "fixed": 3}[strategy]
            protocols[strategy] = run_protocol(
                model, single, strategy, args.probes, args.noise_std,
                args.posterior_samples, seed, device, action_bank, world,
            )
            print(f"task {task_index} / {strategy}: done")
        out_path = args.output / f"task{task_index}_active_random_fixed.gif"
        render_gif(
            task_index, tasks.geometry, tasks.hypotheses, protocols, args.probes, out_path
        )
        manifest.append(
            {
                "task": task_index,
                "geometry": tasks.geometry[task_index].tolist(),
                "hypothesis": int(tasks.hypotheses[task_index].item()),
                "gif": str(out_path),
                "active_actions": [a.tolist() for a in protocols["active"]["selected_indices"]],
                "random_actions": [a.tolist() for a in protocols["random"]["selected_indices"]],
                "fixed_actions": [a.tolist() for a in protocols["fixed"]["selected_indices"]],
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
