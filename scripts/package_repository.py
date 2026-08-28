"""Create a clean source ZIP while excluding Git metadata and generated files."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / f"{ROOT.name}.zip",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.force:
        raise SystemExit(f"archive already exists: {destination}; pass --force to replace it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix.lower() not in {".pyc", ".pyo", ".zip"}
    )
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, (Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
    print(f"wrote {destination} with {len(files)} files")


if __name__ == "__main__":
    main()
