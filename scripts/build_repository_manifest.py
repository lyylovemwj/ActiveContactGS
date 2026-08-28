"""Create a deterministic SHA-256 manifest for the lightweight repository."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "REPOSITORY_MANIFEST.csv"
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "dist",
    "outputs",
    "third_party",
}
TEXT_SUFFIXES = {".cff", ".csv", ".html", ".js", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def canonical_bytes(path: Path) -> bytes:
    """Return stable bytes across Git checkouts with different line endings."""
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


def is_ignored(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return bool(IGNORED_PARTS.intersection(parts)) or any(
        part.endswith(".egg-info") for part in parts
    )


def main() -> None:
    paths = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != DESTINATION
        and not is_ignored(path)
        and path.suffix.lower() not in {".pyc", ".pyo", ".zip"}
    )
    with DESTINATION.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        for path in paths:
            content = canonical_bytes(path)
            writer.writerow(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    print(f"wrote {len(paths)} repository rows")


if __name__ == "__main__":
    main()
