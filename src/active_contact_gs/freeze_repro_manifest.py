from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    tracked_patterns = (
        "src/**/*.py",
        "tests/**/*.py",
        "docs/*.md",
        "paper_assets/*.csv",
        "pyproject.toml",
        "README.md",
    )
    code_paths = sorted(
        {
            path.resolve()
            for pattern in tracked_patterns
            for path in root.glob(pattern)
            if path.is_file()
        }
    )
    package_names = (
        "torch",
        "numpy",
        "matplotlib",
        "trimesh",
        "scipy",
        "pytest",
    )
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu_query = command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    )
    git_commit = command("git", "-C", str(root), "rev-parse", "HEAD")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_query,
        "git_commit": git_commit,
        "packages": packages,
        "code": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in code_paths
        ],
        "results": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in args.result
            if path.is_file()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("created_utc", "gpu", "torch", "torch_cuda", "git_commit")}, indent=2))
    print(f"hashed {len(manifest['code'])} code/assets and {len(manifest['results'])} results")


if __name__ == "__main__":
    main()
