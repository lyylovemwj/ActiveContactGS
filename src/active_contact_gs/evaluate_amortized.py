from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .amortized_model import AmortizedPhysicsPosterior, PosteriorOutput
from .amortized_world import (
    AmortizedContactWorld,
    TaskBatch,
    denormalize_parameters,
    intervention_actions,
    normalize_parameters,
    sample_tasks,
)
from .video_model import GaussianVideoPosterior


def trajectory_features(trajectory: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
    """Low-dimensional video statistic used only by the acquisition objective."""
    frame_count = trajectory.shape[-2]
    indices = torch.linspace(
        0, frame_count - 1, 5, device=trajectory.device
    ).round().long()
    selected = trajectory.index_select(-2, indices).clone()
    selected[..., :2] = selected[..., :2] / geometry[..., None, 3:4]
    max_position = (
        trajectory[..., :2].abs().amax(dim=-2) / geometry[..., 3:4]
    )
    return torch.cat((selected.flatten(-2), max_position), dim=-1)


def joint_posterior_entropy(posterior: PosteriorOutput) -> torch.Tensor:
    """Entropy of the categorical/conditional-diagonal-Gaussian posterior."""
    probabilities = posterior.hypothesis_probabilities
    categorical = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(dim=-1)
    conditional = posterior.conditional_parameter_std.log().sum(dim=-1)
    return categorical + (probabilities * conditional).sum(dim=-1)


@torch.inference_mode()
def amortized_information_scores(
    model: AmortizedPhysicsPosterior,
    posterior: PosteriorOutput,
    actions_sequence: torch.Tensor,
    observations: torch.Tensor,
    probe_mask: torch.Tensor,
    geometry: torch.Tensor,
    action_bank: torch.Tensor,
    world: AmortizedContactWorld,
    *,
    fantasy_samples: int,
    noise_std: float,
    generator: torch.Generator,
    action_chunk: int = 16,
) -> torch.Tensor:
    """One-step Bayesian experimental design through the learned posterior.

    For each action, worlds are sampled from q(h)q(theta|h), observations are
    hallucinated with the differentiable simulator, and the same amortized
    inference network predicts the resulting posterior.  The score is the Monte
    Carlo reduction in joint structure/parameter entropy.
    """
    batch, probes = probe_mask.shape
    probe_index = int(probe_mask[0].sum())
    probabilities = posterior.hypothesis_probabilities
    sampled_hypotheses = torch.multinomial(
        probabilities, fantasy_samples, replacement=True, generator=generator
    )
    conditional_mean, conditional_std = posterior.parameters_for_hypotheses(
        sampled_hypotheses
    )
    normalized = conditional_mean + conditional_std * torch.randn(
        (batch, fantasy_samples, 4), device=geometry.device, generator=generator
    )
    sampled_parameters = denormalize_parameters(normalized.clamp(0.01, 0.99))
    current_entropy = joint_posterior_entropy(posterior)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(action_bank), action_chunk):
        candidate_actions = action_bank[start : start + action_chunk]
        candidates = len(candidate_actions)
        expanded_parameters = sampled_parameters[:, None].expand(
            -1, candidates, -1, -1
        )
        expanded_hypotheses = sampled_hypotheses[:, None].expand(
            -1, candidates, -1
        )
        expanded_geometry = geometry[:, None, None].expand(
            -1, candidates, fantasy_samples, -1
        )
        expanded_actions = candidate_actions[None, :, None].expand(
            batch, -1, fantasy_samples, -1
        )
        clean = world.rollout(
            expanded_parameters.reshape(-1, 4),
            expanded_hypotheses.reshape(-1),
            expanded_actions.reshape(-1, 4),
            expanded_geometry.reshape(-1, 4),
        ).reshape(batch, candidates, fantasy_samples, world.observation_frames, 4)
        fantasy_observations = clean + noise_std * torch.randn(
            clean.shape, device=clean.device, generator=generator
        )
        expanded_sequence = actions_sequence[:, None, None].expand(
            -1, candidates, fantasy_samples, -1, -1
        ).clone()
        expanded_observations = observations[:, None, None].expand(
            -1, candidates, fantasy_samples, -1, -1, -1
        ).clone()
        expanded_mask = probe_mask[:, None, None].expand(
            -1, candidates, fantasy_samples, -1
        ).clone()
        expanded_sequence[..., probe_index, :] = expanded_actions
        expanded_observations[..., probe_index, :, :] = fantasy_observations
        expanded_mask[..., probe_index] = True
        future = model(
            expanded_sequence.reshape(-1, probes, 4),
            expanded_observations.reshape(
                -1, probes, world.observation_frames, 4
            ),
            expanded_geometry.reshape(-1, 4),
            expanded_mask.reshape(-1, probes),
        )
        future_entropy = joint_posterior_entropy(future).reshape(
            batch, candidates, fantasy_samples
        ).mean(dim=-1)
        energy = candidate_actions[:, :2].square().sum(dim=-1) / (6.5**2)
        chunks.append(current_entropy[:, None] - future_entropy - 0.004 * energy[None])
    return torch.cat(chunks, dim=1)


@torch.inference_mode()
def acquisition_scores(
    posterior: PosteriorOutput,
    geometry: torch.Tensor,
    action_bank: torch.Tensor,
    world: AmortizedContactWorld,
    *,
    posterior_samples: int,
    generator: torch.Generator,
    score_kind: str = "information",
    action_chunk: int = 24,
) -> torch.Tensor:
    """Expected conditional entropy reduction under the amortized posterior."""
    batch = geometry.shape[0]
    probabilities = posterior.hypothesis_logits.softmax(dim=-1)
    hypotheses = torch.multinomial(
        probabilities, posterior_samples, replacement=True, generator=generator
    )
    conditional_mean, conditional_std = posterior.parameters_for_hypotheses(hypotheses)
    normalized = conditional_mean + conditional_std * torch.randn(
        (batch, posterior_samples, 4), device=geometry.device, generator=generator
    )
    normalized = normalized.clamp(0.01, 0.99)
    parameters = denormalize_parameters(normalized)
    latent = torch.cat((normalized, hypotheses[..., None].float()), dim=-1)
    latent_centered = latent - latent.mean(dim=1, keepdim=True)
    denominator = max(posterior_samples - 1, 1)
    prior_covariance = torch.einsum(
        "bsi,bsj->bij", latent_centered, latent_centered
    ) / denominator
    latent_eye = torch.eye(5, device=geometry.device, dtype=geometry.dtype)
    _, prior_logdet = torch.linalg.slogdet(prior_covariance + 1e-5 * latent_eye)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(action_bank), action_chunk):
        actions = action_bank[start : start + action_chunk]
        candidates = len(actions)
        parameters_expanded = parameters[:, None].expand(-1, candidates, -1, -1)
        hypotheses_expanded = hypotheses[:, None].expand(-1, candidates, -1)
        geometry_expanded = geometry[:, None, None].expand(
            -1, candidates, posterior_samples, -1
        )
        actions_expanded = actions[None, :, None].expand(
            batch, -1, posterior_samples, -1
        )
        trajectory = world.rollout(
            parameters_expanded.reshape(-1, 4),
            hypotheses_expanded.reshape(-1),
            actions_expanded.reshape(-1, 4),
            geometry_expanded.reshape(-1, 4),
        ).reshape(batch, candidates, posterior_samples, world.observation_frames, 4)
        features = trajectory_features(trajectory, geometry_expanded)
        centered = features - features.mean(dim=2, keepdim=True)
        observation_covariance = torch.einsum(
            "bcsi,bcsj->bcij", centered, centered
        ) / denominator
        if score_kind == "variance":
            score = observation_covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        elif score_kind == "information":
            cross_covariance = torch.einsum(
                "bsi,bcsj->bcij", latent_centered, centered
            ) / denominator
            observation_dimension = features.shape[-1]
            observation_eye = torch.eye(
                observation_dimension, device=geometry.device, dtype=geometry.dtype
            )
            innovation = observation_covariance + 2e-4 * observation_eye
            conditional = prior_covariance[:, None] - cross_covariance @ torch.linalg.solve(
                innovation, cross_covariance.transpose(-1, -2)
            )
            conditional = 0.5 * (conditional + conditional.transpose(-1, -2))
            _, conditional_logdet = torch.linalg.slogdet(
                conditional + 1e-5 * latent_eye
            )
            score = 0.5 * (prior_logdet[:, None] - conditional_logdet)
        else:
            raise ValueError(score_kind)
        energy = actions[:, :2].square().sum(dim=-1) / (6.5**2)
        chunks.append(score - 0.004 * energy[None])
    return torch.cat(chunks, dim=1)


def load_model(checkpoint_path: Path, device: torch.device) -> AmortizedPhysicsPosterior:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model_class = (
        GaussianVideoPosterior
        if config.get("observation_mode", "pose") == "gaussian_video"
        else AmortizedPhysicsPosterior
    )
    model_options = dict(
        observation_frames=config["observation_frames"],
        width=config["width"],
        heads=config["heads"],
        layers=config["layers"],
        geometry_mode=config.get("geometry_mode", "full"),
    )
    if model_class is GaussianVideoPosterior:
        model_options.update(
            image_resolution=config.get("image_resolution", 16),
            video_frames=config.get("video_frames", 6),
        )
    model = model_class(**model_options).to(device)
    state = checkpoint["model"]
    # Convert the earlier factorized q(h)q(theta) ablation into two identical
    # conditional branches so it can be evaluated with the same acquisition code.
    if state["parameter_head.weight"].shape[0] == 8:
        old_weight = state["parameter_head.weight"]
        old_bias = state["parameter_head.bias"]
        state["parameter_head.weight"] = torch.cat(
            (
                old_weight[:4].repeat(2, 1),
                old_weight[4:].repeat(2, 1),
            ),
            dim=0,
        )
        state["parameter_head.bias"] = torch.cat(
            (old_bias[:4].repeat(2), old_bias[4:].repeat(2)), dim=0
        )
    model.load_state_dict(state)
    model.eval()
    return model


def task_metrics(
    posterior: PosteriorOutput, tasks: TaskBatch
) -> dict[str, torch.Tensor]:
    target = normalize_parameters(tasks.parameters)
    error = (posterior.parameter_mean - target).abs()
    probabilities = posterior.hypothesis_logits.softmax(dim=-1)
    return {
        "normalized_parameter_mae": error.mean(dim=-1),
        "mass_error": error[:, 0],
        "friction_error": error[:, 1],
        "restitution_error": error[:, 2],
        "inertia_error": error[:, 3],
        "structure_correct": (
            probabilities.argmax(dim=-1) == tasks.hypotheses
        ).float(),
        "true_structure_probability": probabilities.gather(
            1, tasks.hypotheses[:, None]
        ).squeeze(1),
    }


@torch.inference_mode()
def predictive_rollout_error(
    posterior: PosteriorOutput,
    tasks: TaskBatch,
    world: AmortizedContactWorld,
    action_bank: torch.Tensor,
) -> torch.Tensor:
    evaluation_actions = action_bank[
        torch.linspace(0, len(action_bank) - 1, 12, device=action_bank.device).long()
    ]
    batch, actions_count = len(tasks.parameters), len(evaluation_actions)
    predicted_hypotheses = posterior.hypothesis_logits.argmax(dim=-1)
    row = torch.arange(batch, device=action_bank.device)
    predicted_parameters = denormalize_parameters(
        posterior.conditional_parameter_mean[row, predicted_hypotheses]
    )
    actions = evaluation_actions[None].expand(batch, -1, -1)
    geometry = tasks.geometry[:, None].expand(-1, actions_count, -1)
    predicted = world.rollout(
        predicted_parameters[:, None].expand(-1, actions_count, -1).reshape(-1, 4),
        predicted_hypotheses[:, None].expand(-1, actions_count).reshape(-1),
        actions.reshape(-1, 4),
        geometry.reshape(-1, 4),
    ).reshape(batch, actions_count, world.observation_frames, 4)
    truth = world.rollout(
        tasks.parameters[:, None].expand(-1, actions_count, -1).reshape(-1, 4),
        tasks.hypotheses[:, None].expand(-1, actions_count).reshape(-1),
        actions.reshape(-1, 4),
        geometry.reshape(-1, 4),
    ).reshape_as(predicted)
    residual = predicted - truth
    residual[..., :2] = residual[..., :2] / tasks.geometry[:, None, None, 3:4]
    return residual.square().flatten(1).mean(dim=-1).sqrt()


@torch.inference_mode()
def downstream_control_metrics(
    posterior: PosteriorOutput,
    tasks: TaskBatch,
    world: AmortizedContactWorld,
    action_bank: torch.Tensor,
    *,
    generator: torch.Generator,
    posterior_samples: int = 24,
) -> dict[str, torch.Tensor]:
    """One-step belief-aware control toward a reachable target pose."""
    batch = len(tasks.parameters)
    magnitude = action_bank[:, :2].square().sum(dim=-1).sqrt()
    candidate_indices = torch.where(magnitude >= torch.quantile(magnitude, 0.35))[0]
    candidates = action_bank[candidate_indices]
    target_choice = torch.randint(
        len(candidates), (batch,), device=action_bank.device, generator=generator
    )
    target_actions = candidates[target_choice]
    target_trajectory = world.rollout(
        tasks.parameters, tasks.hypotheses, target_actions, tasks.geometry
    )
    target = target_trajectory[:, -1].clone()
    target[:, :2] = target[:, :2] / tasks.geometry[:, 3:4]

    probabilities = posterior.hypothesis_probabilities
    sampled_hypotheses = torch.multinomial(
        probabilities, posterior_samples, replacement=True, generator=generator
    )
    conditional_mean, conditional_std = posterior.parameters_for_hypotheses(
        sampled_hypotheses
    )
    normalized = conditional_mean + conditional_std * torch.randn(
        conditional_mean.shape, device=action_bank.device, generator=generator
    )
    sampled_parameters = denormalize_parameters(normalized.clamp(0.01, 0.99))
    action_count = len(candidates)
    parameters_expanded = sampled_parameters[:, None].expand(
        -1, action_count, -1, -1
    )
    hypotheses_expanded = sampled_hypotheses[:, None].expand(
        -1, action_count, -1
    )
    geometry_expanded = tasks.geometry[:, None, None].expand(
        -1, action_count, posterior_samples, -1
    )
    actions_expanded = candidates[None, :, None].expand(
        batch, -1, posterior_samples, -1
    )
    predicted = world.rollout(
        parameters_expanded.reshape(-1, 4),
        hypotheses_expanded.reshape(-1),
        actions_expanded.reshape(-1, 4),
        geometry_expanded.reshape(-1, 4),
    ).reshape(batch, action_count, posterior_samples, world.observation_frames, 4)
    predicted_final = predicted[..., -1, :].clone()
    predicted_final[..., :2] = (
        predicted_final[..., :2] / geometry_expanded[..., 3:4]
    )
    sample_cost = (predicted_final - target[:, None, None]).square().mean(dim=-1)
    belief_objective = sample_cost.mean(dim=-1) + 0.25 * sample_cost.std(dim=-1)
    belief_choice = belief_objective.argmin(dim=-1)

    map_hypothesis = posterior.hypothesis_logits.argmax(dim=-1)
    row = torch.arange(batch, device=action_bank.device)
    map_parameters = denormalize_parameters(
        posterior.conditional_parameter_mean[row, map_hypothesis]
    )
    point_predicted = world.rollout(
        map_parameters[:, None].expand(-1, action_count, -1).reshape(-1, 4),
        map_hypothesis[:, None].expand(-1, action_count).reshape(-1),
        candidates[None].expand(batch, -1, -1).reshape(-1, 4),
        tasks.geometry[:, None].expand(-1, action_count, -1).reshape(-1, 4),
    ).reshape(batch, action_count, world.observation_frames, 4)
    point_final = point_predicted[..., -1, :].clone()
    point_final[..., :2] = point_final[..., :2] / tasks.geometry[:, None, 3:4]
    point_choice = (point_final - target[:, None]).square().mean(dim=-1).argmin(dim=-1)

    def execute(choice: torch.Tensor) -> torch.Tensor:
        actual = world.rollout(
            tasks.parameters,
            tasks.hypotheses,
            candidates[choice],
            tasks.geometry,
        )[:, -1]
        actual[:, :2] = actual[:, :2] / tasks.geometry[:, 3:4]
        return (actual - target).square().mean(dim=-1).sqrt()

    belief_error = execute(belief_choice)
    point_error = execute(point_choice)
    return {
        "belief_control_error": belief_error,
        "belief_control_success": (belief_error < 0.08).float(),
        "point_control_error": point_error,
        "point_control_success": (point_error < 0.08).float(),
    }


@torch.inference_mode()
def evaluate_strategy(
    model: AmortizedPhysicsPosterior,
    tasks: TaskBatch,
    *,
    strategy: str,
    probes: int,
    noise_std: float,
    posterior_samples: int,
    seed: int,
) -> dict[str, object]:
    device = tasks.parameters.device
    generator = torch.Generator(device=device).manual_seed(seed)
    world = AmortizedContactWorld()
    action_bank = intervention_actions(device=device)
    batch = len(tasks.parameters)
    actions_sequence = torch.zeros((batch, probes, 4), device=device)
    observations = torch.zeros(
        (batch, probes, world.observation_frames, 4), device=device
    )
    mask = torch.zeros((batch, probes), device=device, dtype=torch.bool)
    used = torch.zeros((batch, len(action_bank)), device=device, dtype=torch.bool)
    selected_history: list[torch.Tensor] = []
    histories: list[dict[str, object]] = []
    fixed_order = torch.linspace(
        len(action_bank) - 1, len(action_bank) // 2, probes, device=device
    ).long()
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
                fantasy_samples=min(posterior_samples, 16),
                noise_std=noise_std,
                generator=generator,
                # Pixel-space fantasies are substantially larger than pose
                # tokens; single-action chunks bound memory without changing
                # the acquisition score or candidate set.
                action_chunk=1 if isinstance(model, GaussianVideoPosterior) else 16,
            )
            scores = scores.masked_fill(used, -torch.inf)
            selected = scores.argmax(dim=-1)
        elif strategy in {"covariance", "variance"}:
            scores = acquisition_scores(
                posterior,
                tasks.geometry,
                action_bank,
                world,
                posterior_samples=posterior_samples,
                generator=generator,
                score_kind="information" if strategy == "covariance" else "variance",
            )
            scores = scores.masked_fill(used, -torch.inf)
            selected = scores.argmax(dim=-1)
        elif strategy in {"random", "high_energy_random"}:
            scores = torch.rand(
                (batch, len(action_bank)), device=device, generator=generator
            )
            if strategy == "high_energy_random":
                magnitude = action_bank[:, :2].square().sum(dim=-1).sqrt()
                scores[:, magnitude < torch.quantile(magnitude, 0.75)] = -torch.inf
            scores = scores.masked_fill(used, -torch.inf)
            selected = scores.argmax(dim=-1)
        elif strategy == "fixed":
            selected = fixed_order[probe].expand(batch)
        else:
            raise ValueError(strategy)
        used.scatter_(1, selected[:, None], True)
        selected_history.append(selected)
        chosen = action_bank[selected]
        observation = world.observe(
            tasks, chosen, noise_std=noise_std, generator=generator
        )
        actions_sequence[:, probe] = chosen
        observations[:, probe] = observation
        mask[:, probe] = True
        updated = model(actions_sequence, observations, tasks.geometry, mask)
        metrics = task_metrics(updated, tasks)
        histories.append(
            {
                "probe": probe + 1,
                **{
                    key: float(value.mean().cpu()) for key, value in metrics.items()
                },
                "per_task": {
                    key: value.float().cpu().tolist() for key, value in metrics.items()
                },
            }
        )
    final_posterior = model(actions_sequence, observations, tasks.geometry, mask)
    final = task_metrics(final_posterior, tasks)
    final["predictive_rollout_rmse"] = predictive_rollout_error(
        final_posterior, tasks, world, action_bank
    )
    final.update(
        downstream_control_metrics(
            final_posterior,
            tasks,
            world,
            action_bank,
            generator=generator,
        )
    )
    return {
        "strategy": strategy,
        "history": histories,
        "final_per_task": {
            key: value.float().cpu().tolist() for key, value in final.items()
        },
        "selected_action_indices": torch.stack(selected_history, dim=1).cpu().tolist(),
    }


def paired_bootstrap(
    active: list[float], baseline: list[float], *, seed: int = 991
) -> dict[str, float]:
    active_array = np.asarray(active, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    difference = baseline_array - active_array
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(difference), size=(20_000, len(difference)))
    sampled = difference[indices].mean(axis=1)
    low, median, high = np.quantile(sampled, (0.025, 0.5, 0.975))
    return {
        "active_mean": float(active_array.mean()),
        "baseline_mean": float(baseline_array.mean()),
        "relative_improvement": float(
            1.0 - active_array.mean() / max(baseline_array.mean(), 1e-12)
        ),
        "paired_absolute_improvement": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_median": float(median),
        "ci95_high": float(high),
        "probability_active_better": float((sampled > 0).mean()),
        "active_wins": int((difference > 0).sum()),
        "pairs": len(difference),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=128)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.006)
    parser.add_argument("--posterior-samples", type=int, default=48)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["active", "covariance", "variance", "random", "high_energy_random", "fixed"],
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA")
    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    rows: dict[str, dict[str, dict[str, object]]] = {}
    for split_index, split in enumerate(("id", "ood_thin", "ood_round")):
        generator = torch.Generator(device=device).manual_seed(8701 + split_index)
        tasks = sample_tasks(
            args.tasks, device=device, generator=generator, split=split
        )
        rows[split] = {}
        for strategy_index, strategy in enumerate(args.strategies):
            print(f"evaluating {split}/{strategy}", flush=True)
            rows[split][strategy] = evaluate_strategy(
                model,
                tasks,
                strategy=strategy,
                probes=args.probes,
                noise_std=args.noise_std,
                posterior_samples=args.posterior_samples,
                seed=9701 + split_index * 100 + strategy_index,
            )
    comparisons: dict[str, dict[str, dict[str, object]]] = {}
    for split, strategies in rows.items():
        comparisons[split] = {}
        active = strategies["active"]["final_per_task"]
        for strategy, result in strategies.items():
            if strategy == "active":
                continue
            comparisons[split][strategy] = {
                metric: paired_bootstrap(
                    active[metric], result["final_per_task"][metric]
                )
                for metric in (
                    "normalized_parameter_mae",
                    "predictive_rollout_rmse",
                    "belief_control_error",
                )
            }
    payload = {
        "config": vars(args) | {"checkpoint": str(args.checkpoint), "output": str(args.output)},
        "results": rows,
        "comparisons": comparisons,
        "gpu": torch.cuda.get_device_name(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(comparisons, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
