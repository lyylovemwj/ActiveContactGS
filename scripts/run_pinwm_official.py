"""Reproducible headless launcher for the official PIN-WM Push-T pipeline.

The upstream entry point hard-codes GUI rendering and its experiment paths.  This
launcher changes only those runtime settings; the model, renderer, differentiable
simulator, loss, and optimizer remain the upstream implementations.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin-root", type=Path, required=True)
    parser.add_argument("--data-path", default="dataset/sim_push_t")
    parser.add_argument("--output-path", default="output/sim_push_t")
    parser.add_argument("--log-name", default="physical_parameters")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--iterations", type=int, default=125)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--optimization-intervals", type=int, default=8)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--metadata", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pin_root = args.pin_root.expanduser().resolve()
    if not (pin_root / "scripts" / "identify_physical_parameters.py").is_file():
        raise FileNotFoundError(f"PIN-WM checkout not found at {pin_root}")
    if args.frames % args.optimization_intervals:
        raise ValueError("frames must be divisible by optimization-intervals")

    os.chdir(pin_root)
    sys.path.insert(0, str(pin_root))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pin = importlib.import_module("scripts.identify_physical_parameters")
    pin.sys_args.update(
        seed=args.seed,
        output_path=args.output_path,
        log_dir_name=args.log_name,
    )
    pin.data_args.update(
        data_path=args.data_path,
        n_frames=args.frames,
        H=args.height,
        W=args.width,
    )
    pin.sim_args.update(
        train_iteration=args.iterations,
        save_iteration=args.save_interval,
        opt_interval_num=args.optimization_intervals,
    )

    if not args.vis:
        upstream_simulator = pin.Simulator

        def headless_simulator(dt, device, vis):
            del vis
            return upstream_simulator(dt, device=device, vis=False)

        pin.Simulator = headless_simulator

    if args.metadata:
        metadata = {
            "launcher": "scripts/run_pinwm_official.py",
            "pin_root": str(pin_root),
            "pin_commit": os.popen("git rev-parse HEAD").read().strip(),
            "seed": args.seed,
            "data_path": args.data_path,
            "output_path": args.output_path,
            "frames": args.frames,
            "height": args.height,
            "width": args.width,
            "iterations": args.iterations,
            "headless": not args.vis,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pin.identify_physical_parameters()


if __name__ == "__main__":
    main()
