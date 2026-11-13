from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .iclr_model import ICLRPhysicsPosterior, posterior_metrics
from .iclr_world import iclr_actions


def load_split(root: Path, split: str, device: torch.device) -> dict[str, torch.Tensor]:
    paths = sorted(root.glob(f"{split}-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards in {root}")
    rows = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    return {key: torch.cat([row[key] for row in rows]).to(device) for key in rows[0]}


def select_batch(data, indices, lengths, bank, augmentation_noise=0.0):
    actions = bank[data["action_indices"][indices].long()]
    observations = data["observations"][indices].float()
    if augmentation_noise:
        observations = observations + augmentation_noise * torch.randn_like(observations)
    mask = torch.arange(actions.shape[1], device=actions.device)[None] < lengths[:, None]
    return actions, observations, data["geometry"][indices], mask


@torch.inference_mode()
def validate(model, data, bank, max_tasks: int = 2048):
    model.eval()
    count = min(max_tasks, len(data["parameters"]))
    indices = torch.arange(count, device=bank.device)
    result = {}
    for length in (0, 1, 2, 4, data["action_indices"].shape[1]):
        lengths = torch.full((count,), length, device=bank.device)
        args = select_batch(data, indices, lengths, bank)
        metrics = posterior_metrics(model(*args), data["parameters"][:count], data["factors"][:count].long())
        result[str(length)] = {key: float(value.float().mean().cpu()) for key, value in metrics.items()}
    model.train()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=sorted(ICLRPhysicsPosterior.REPRESENTATIONS), default="independent")
    parser.add_argument("--steps", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=1000)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ICLR training requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    train = load_split(args.data, "train", device)
    validation = load_split(args.data, "id", device)
    bank = iclr_actions(device=device)
    model = ICLRPhysicsPosterior(
        observation_frames=train["observations"].shape[-2],
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        representation=args.representation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.05)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    rolling = {}
    started = time.perf_counter()
    probes = train["action_indices"].shape[1]
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train["parameters"]), (args.batch_size,), device=device)
        lengths = torch.randint(probes + 1, (args.batch_size,), device=device)
        inputs = select_batch(train, indices, lengths, bank, augmentation_noise=0.002)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            posterior = model(*inputs)
            loss, metrics = model.loss(posterior, train["parameters"][indices], train["factors"][indices].long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        for key, value in metrics.items():
            rolling[key] = rolling.get(key, 0.0) + float(value)
        if step % args.log_every == 0:
            row = {"step": step, "seconds": time.perf_counter() - started, **{key: value / args.log_every for key, value in rolling.items()}}
            print(json.dumps(row), flush=True)
            history.append(row)
            rolling.clear()
        if step % args.validate_every == 0 or step == args.steps:
            validation_metrics = validate(model, validation, bank)
            print(json.dumps({"step": step, "validation": validation_metrics}), flush=True)
            history.append({"step": step, "validation": validation_metrics})
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {"observation_frames": model.observation_frames, "width": args.width, "heads": args.heads, "layers": args.layers, "representation": args.representation},
                    "step": step,
                    "validation": validation_metrics,
                },
                args.output / "checkpoint-latest.pt",
            )
            (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
