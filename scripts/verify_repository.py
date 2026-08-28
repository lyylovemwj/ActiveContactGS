"""Check that the public repository is lightweight and free of obvious secrets."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "README_CN.md",
    "LICENSE",
    "CITATION.cff",
    "CHANGELOG.md",
    "pyproject.toml",
    "environment.yml",
    "REPOSITORY_MANIFEST.csv",
    "docs/REPRODUCIBILITY.md",
    "src/active_contact_gs/__init__.py",
)
FORBIDDEN_DIRECTORIES = {"checkpoints", "figures", "outputs", "results", "third_party"}
FORBIDDEN_SUFFIXES = {".ckpt", ".gif", ".mp4", ".pdf", ".pt", ".pth", ".tar", ".tgz", ".zip"}
TEXT_SUFFIXES = {".cff", ".csv", ".html", ".js", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
MANIFEST_TEXT_SUFFIXES = TEXT_SUFFIXES | {".txt"}
MAX_FILE_BYTES = 5 * 1024 * 1024


def canonical_bytes(path: Path) -> bytes:
    """Return the same content on Windows and POSIX Git checkouts."""
    content = path.read_bytes()
    if path.suffix.lower() in MANIFEST_TEXT_SUFFIXES:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


def verify_manifest() -> list[str]:
    errors: list[str] = []
    manifest = ROOT / "REPOSITORY_MANIFEST.csv"
    if not manifest.is_file():
        return errors
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = ROOT / row["path"]
            if not path.is_file():
                errors.append(f"manifest file missing: {row['path']}")
                continue
            content = canonical_bytes(path)
            if len(content) != int(row["bytes"]):
                errors.append(f"manifest size mismatch: {row['path']}")
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                errors.append(f"manifest SHA-256 mismatch: {row['path']}")
    return errors


def main() -> None:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    errors.extend(verify_manifest())

    for directory in FORBIDDEN_DIRECTORIES:
        if (ROOT / directory).exists():
            errors.append(f"forbidden generated directory: {directory}")

    private_key = re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
    ssh_endpoint = re.compile(r"ssh\s+-p\s+\d+\s+\S+@", re.IGNORECASE)
    machine_path = re.compile(r"(?:[A-Za-z]:\\|/root/|/home/[^/< ]+/)")
    personal_email = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 5 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/archive: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.name == "verify_repository.py":
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if private_key.search(content):
            errors.append(f"private key marker: {relative}")
        if ssh_endpoint.search(content):
            errors.append(f"embedded SSH endpoint: {relative}")
        if machine_path.search(content):
            errors.append(f"machine-specific absolute path: {relative}")
        if personal_email.search(content):
            errors.append(f"public email address: {relative}")

    if errors:
        raise SystemExit("repository verification failed:\n- " + "\n- ".join(errors))
    print("repository verification passed")


if __name__ == "__main__":
    main()
