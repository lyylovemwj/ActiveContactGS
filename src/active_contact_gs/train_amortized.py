from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .amortized_model import AmortizedPhysicsPosterior, posterior_metrics
from .amortized_world import intervention_actions
from .video_model import GaussianVideoPosterior


def load_split(
    directory: Path, split: str, *, device: torch.device
) -> dict[str, torch.Tensor]:
    paths = sorted(directory.glob(f"{split}-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {directory}")
    rows = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    result = {
        key: torch.cat([row[key] for row in rows], dim=0).to(device)
        for key in rows[0]
    }
    print(f"loaded {split}: {len(result['parameters'])} tasks", flush=True)
    return result


def select_batch(
    data: dict[str, torch.Tensor],
    indices: torch.Tensor,
    lengths: torch.Tensor,
    action_bank: torch.Tensor,
    *,
    augmentation_noise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action_indices = data["action_indices"][indices].long()
    actions = action_bank[action_indices]
    observations = data["observations"][indices].float()
    if augmentation_noise:
        observations = observations + augmentation_noise * torch.randn_like(observations)
    probes = actions.shape[1]
    mask = torch.arange(probes, device=actions.device)[None] < lengths[:, None]
    return actions, observations, data["geometry"][indices], mask


@torch.inference_mode()
def validate(
    model: AmortizedPhysicsPosterior,
    data: dict[str, torch.Tensor],
    action_bank: torch.Tensor,
    *,
    max_tasks: int,
) -> dict[str, dict[str, object]]:
    model.eval()
    count = min(max_tasks, len(data["parameters"]))
    indices = torch.arange(count, device=action_bank.device)
    result: dict[str, dict[str, object]] = {}
    probes = data["action_indices"].shape[1]
    for length in (0, 1, 2, 3, probes):
        lengths = torch.full((count,), length, device=action_bank.device)
        actions, observations, geometry, mask = select_batch(
            data,
            indices,
            lengths,
            action_bank,
            augmentation_noise=0.0,
        )
        posterior = model(actions, observations, geometry, mask)
        metrics = posterior_metrics(
            posterior, data["parameters"][indices], data["hypotheses"][indices].long()
        )
        result[str(length)] = {
            key: value.detach().float().cpu().tolist() for key, value in metrics.items()
        }
    model.train()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument(
        "--geometry-mode",
        choices=("full", "isotropic", "no_shape"),
        default="full",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("pose", "gaussian_video"),
        default="pose",
    )
    parser.add_argument("--image-resolution", type=int, default=32)
    parser.add_argument("--video-frames", type=int, default=6)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=500)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    train = load_split(args.data, "train", device=device)
    validation = load_split(args.data, "id", device=device)
    action_bank = intervention_actions(device=device)
    probes = train["action_indices"].shape[1]
    frames = train["observations"].shape[-2]
    model_class = (
        GaussianVideoPosterior
        if args.observation_mode == "gaussian_video"
        else AmortizedPhysicsPosterior
    )
    model_options = dict(
        observation_frames=frames,
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        geometry_mode=args.geometry_mode,
    )
    if args.observation_mode == "gaussian_video":
        model_options.update(
            image_resolution=args.image_resolution,
            video_frames=args.video_frames,
        )
    model = model_class(**model_options).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    rolling: dict[str, float] = {}
    model.train()
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            len(train["parameters"]), (args.batch_size,), device=device
        )
        # Uniformly include the prior-only case and every interaction budget.
        lengths = torch.randint(probes + 1, (args.batch_size,), device=device)
        actions, observations, geometry, mask = select_batch(
            train,
            indices,
            lengths,
            action_bank,
            augmentation_noise=0.002,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            posterior = model(actions, observations, geometry, mask)
            loss, metrics = model.loss(
                posterior,
                train["parameters"][indices],
                train["hypotheses"][indices].long(),
            )
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for key, value in metrics.items():
            rolling[key] = rolling.get(key, 0.0) + float(value)
        if step % args.log_every == 0:
            elapsed = time.perf_counter() - started
            row: dict[str, object] = {
                "step": step,
                "seconds": elapsed,
                "steps_per_second": step / elapsed,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            row.update({key: value / args.log_every for key, value in rolling.items()})
            rolling.clear()
            print(json.dumps(row), flush=True)
            history.append(row)
        if step % args.validate_every == 0 or step == args.steps:
            validation_metrics = validate(
                model,
                validation,
                action_bank,
                max_tasks=1_024 if args.observation_mode == "gaussian_video" else 8_192,
            )
            event = {"step": step, "validation": validation_metrics}
            print(json.dumps(event), flush=True)
            history.append(event)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": {
                    "observation_frames": frames,
                    "width": args.width,
                    "heads": args.heads,
                    "layers": args.layers,
                    "probes": probes,
                    "geometry_mode": args.geometry_mode,
                    "observation_mode": args.observation_mode,
                    "image_resolution": args.image_resolution,
                    "video_frames": args.video_frames,
                },
                "validation": validation_metrics,
            }
            torch.save(checkpoint, args.output / "checkpoint-latest.pt")
            (args.output / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
    print(
        f"completed {args.steps} steps in {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
