"""Run a small end-to-end ActiveContactGS diagnostic on CUDA."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small CUDA identification experiment and summarize it."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/quickstart"))
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--particles", type=int, default=128)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "Quick Start requires a CUDA GPU. For a CPU-only installation check, "
            "run: python -c \"import torch, active_contact_gs\""
        )

    args.output.mkdir(parents=True, exist_ok=True)
    experiment = args.output / "experiment.json"
    analysis = args.output / "analysis.json"
    run(
        [
            sys.executable,
            "-m",
            "active_contact_gs.experiment",
            "--device",
            "cuda",
            "--probes",
            str(args.probes),
            "--seeds",
            str(args.seeds),
            "--particles",
            str(args.particles),
            "--output",
            str(experiment),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "active_contact_gs.analyze",
            str(experiment),
            "--output",
            str(analysis),
        ]
    )
    print(f"Quick Start complete. Results: {experiment}; summary: {analysis}")


if __name__ == "__main__":
    main()
