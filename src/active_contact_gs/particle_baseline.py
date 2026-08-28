from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .amortized_world import (
    AmortizedContactWorld,
    TaskBatch,
    denormalize_parameters,
    intervention_actions,
    normalize_parameters,
    sample_tasks,
)
from .belief import PRIOR_HIGH, PRIOR_LOW
from .evaluate_amortized import load_model


@torch.inference_mode()
def particle_inference(
    tasks: TaskBatch,
    actions: torch.Tensor,
    observations: torch.Tensor,
    *,
    particle_count: int,
    noise_std: float,
    generator: torch.Generator,
    world: AmortizedContactWorld | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, probes = actions.shape[:2]
    world = world or AmortizedContactWorld()
    low = torch.tensor(PRIOR_LOW, device="cuda")
    high = torch.tensor(PRIOR_HIGH, device="cuda")
    particles = low + (high - low) * torch.rand(
        (batch, particle_count, 4), device="cuda", generator=generator
    )
    hypotheses = torch.arange(particle_count, device="cuda").remainder(2)
    hypotheses = hypotheses[None].expand(batch, -1).clone()
    log_weights = torch.full(
        (batch, particle_count),
        -torch.log(torch.tensor(float(particle_count), device="cuda")),
        device="cuda",
    )
    for probe in range(probes):
        prediction = world.rollout(
            particles.reshape(-1, 4),
            hypotheses.reshape(-1),
            actions[:, probe, None].expand(-1, particle_count, -1).reshape(-1, 4),
            tasks.geometry[:, None].expand(-1, particle_count, -1).reshape(-1, 4),
        ).reshape(batch, particle_count, world.observation_frames, 4)
        residual = prediction - observations[:, probe, None]
        likelihood = -0.5 * residual.square().flatten(2).mean(dim=-1) / noise_std**2
        log_weights = torch.log_softmax(log_weights + likelihood, dim=-1)
        if probe + 1 < probes:
            weights = log_weights.softmax(dim=-1)
            ess = 1.0 / weights.square().sum(dim=-1)
            resample = ess < 0.35 * particle_count
            if resample.any():
                selected = torch.multinomial(
                    weights[resample],
                    particle_count,
                    replacement=True,
                    generator=generator,
                )
                selected_parameters = particles[resample].gather(
                    1, selected[..., None].expand(-1, -1, 4)
                )
                jitter = 0.008 * (high - low) * torch.randn(
                    selected_parameters.shape, device="cuda", generator=generator
                )
                particles[resample] = (selected_parameters + jitter).clamp(low, high)
                hypotheses[resample] = hypotheses[resample].gather(1, selected)
                log_weights[resample] = -torch.log(
                    torch.tensor(float(particle_count), device="cuda")
                )
    weights = log_weights.softmax(dim=-1)
    parameter_mean = (weights[..., None] * particles).sum(dim=1)
    sphere_probability = (
        weights * (hypotheses == 1).to(weights.dtype)
    ).sum(dim=-1)
    ess = 1.0 / weights.square().sum(dim=-1)
    return parameter_mean, sphere_probability, ess


@torch.inference_mode()
def evaluate_split(
    model,
    tasks: TaskBatch,
    selected_indices: torch.Tensor,
    *,
    particle_count: int,
    noise_std: float,
    chunk_size: int,
    seed: int,
) -> dict[str, object]:
    world = AmortizedContactWorld()
    action_bank = intervention_actions(device="cuda")
    actions = action_bank[selected_indices]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    observations = []
    for probe in range(actions.shape[1]):
        observations.append(
            world.observe(
                tasks, actions[:, probe], noise_std=noise_std, generator=generator
            )
        )
    observations_tensor = torch.stack(observations, dim=1)
    mask = torch.ones(actions.shape[:2], device="cuda", dtype=torch.bool)
    # Warm up kernels so per-task latency excludes one-time CUDA initialization.
    model(actions[:1], observations_tensor[:1], tasks.geometry[:1], mask[:1])
    torch.cuda.synchronize()
    model_started = time.perf_counter()
    posterior = model(actions, observations_tensor, tasks.geometry, mask)
    torch.cuda.synchronize()
    model_seconds = time.perf_counter() - model_started
    model_parameter = denormalize_parameters(posterior.parameter_mean)
    model_probability = posterior.hypothesis_probabilities[:, 1]

    particle_parameters = []
    particle_probabilities = []
    particle_ess = []
    torch.cuda.synchronize()
    particle_started = time.perf_counter()
    for start in range(0, len(tasks.parameters), chunk_size):
        stop = min(start + chunk_size, len(tasks.parameters))
        chunk_tasks = TaskBatch(
            tasks.parameters[start:stop],
            tasks.hypotheses[start:stop],
            tasks.geometry[start:stop],
        )
        estimate, probability, ess = particle_inference(
            chunk_tasks,
            actions[start:stop],
            observations_tensor[start:stop],
            particle_count=particle_count,
            noise_std=noise_std,
            generator=generator,
        )
        particle_parameters.append(estimate)
        particle_probabilities.append(probability)
        particle_ess.append(ess)
    torch.cuda.synchronize()
    particle_seconds = time.perf_counter() - particle_started
    particle_parameter = torch.cat(particle_parameters)
    particle_probability = torch.cat(particle_probabilities)
    particle_ess_tensor = torch.cat(particle_ess)
    target = normalize_parameters(tasks.parameters)
    low = torch.tensor(PRIOR_LOW, device="cuda")
    high = torch.tensor(PRIOR_HIGH, device="cuda")

    def summarize(
        parameter: torch.Tensor, sphere_probability: torch.Tensor, seconds: float
    ) -> dict[str, object]:
        normalized = (parameter - low) / (high - low)
        true_probability = torch.where(
            tasks.hypotheses == 1, sphere_probability, 1.0 - sphere_probability
        )
        error = (normalized - target).abs().mean(dim=-1)
        return {
            "normalized_parameter_mae": float(error.mean()),
            "structure_accuracy": float((true_probability > 0.5).float().mean()),
            "true_structure_probability": float(true_probability.mean()),
            "total_seconds": seconds,
            "milliseconds_per_task": seconds * 1000 / len(tasks.parameters),
            "per_task_error": error.cpu().tolist(),
        }

    return {
        "amortized": summarize(model_parameter, model_probability, model_seconds),
        "particle": summarize(
            particle_parameter, particle_probability, particle_seconds
        )
        | {"mean_effective_sample_size": float(particle_ess_tensor.mean())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=16_384)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.006)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Particle baseline requires CUDA")
    model = load_model(args.checkpoint, torch.device("cuda"))
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    results = {}
    for split_index, split in enumerate(("id", "ood_thin", "ood_round")):
        count = evaluation["config"]["tasks"]
        generator = torch.Generator(device="cuda").manual_seed(8701 + split_index)
        tasks = sample_tasks(count, device="cuda", generator=generator, split=split)
        indices = torch.tensor(
            evaluation["results"][split]["active"]["selected_action_indices"],
            device="cuda",
            dtype=torch.long,
        )
        print(f"evaluating {split}", flush=True)
        results[split] = evaluate_split(
            model,
            tasks,
            indices,
            particle_count=args.particles,
            noise_std=args.noise_std,
            chunk_size=args.chunk_size,
            seed=13101 + split_index,
        )
        print(json.dumps(results[split], indent=2), flush=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "evaluation": str(args.evaluation),
        "particles": args.particles,
        "results": results,
        "gpu": torch.cuda.get_device_name(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
