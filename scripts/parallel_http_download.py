#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def remote_size(url: str) -> int:
    request = Request(url, method="HEAD", headers={"User-Agent": "ContactSplat/0.1"})
    with urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def fetch_part(url: str, part: Path, start: int, end: int, retries: int) -> int:
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return expected
    temporary = part.with_suffix(part.suffix + ".tmp")
    for attempt in range(retries):
        try:
            request = Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "ContactSplat/0.1",
                },
            )
            with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                if response.status != 206:
                    raise RuntimeError(f"server ignored range request: HTTP {response.status}")
                while True:
                    block = response.read(1024 * 256)
                    if not block:
                        break
                    stream.write(block)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"short range {start}-{end}: {temporary.stat().st_size} != {expected}"
                )
            os.replace(temporary, part)
            return expected
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()
    size = remote_size(args.url)
    if args.output.exists() and args.output.stat().st_size == size:
        print(f"complete {args.output} {size} bytes")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_directory = args.output.with_name(args.output.name + ".parts")
    parts_directory.mkdir(parents=True, exist_ok=True)
    ranges = [
        (index, start, min(start + args.chunk_size, size) - 1)
        for index, start in enumerate(range(0, size, args.chunk_size))
    ]
    started = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_part,
                args.url,
                parts_directory / f"{index:06d}.part",
                start,
                end,
                args.retries,
            ): index
            for index, start, end in ranges
        }
        for future in as_completed(futures):
            completed += future.result()
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"progress {completed}/{size} bytes "
                f"({completed / size:.1%}) aggregate={completed / elapsed / 1024:.1f} KiB/s",
                flush=True,
            )

    temporary_output = args.output.with_suffix(args.output.suffix + ".assembling")
    digest = hashlib.sha256()
    with temporary_output.open("wb") as destination:
        for index, _, _ in ranges:
            with (parts_directory / f"{index:06d}.part").open("rb") as source:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    destination.write(block)
    if temporary_output.stat().st_size != size:
        raise RuntimeError("assembled file size mismatch")
    os.replace(temporary_output, args.output)
    print(f"complete {args.output} bytes={size} sha256={digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
