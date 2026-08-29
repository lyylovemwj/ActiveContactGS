# scripts — Helper Scripts

This directory contains the standalone helper scripts for **ActiveContactGS**, the research project behind the paper *Contact Geometry Is a Latent Variable: Active Bayesian Identification of Gaussian Physical Worlds*. These scripts handle data acquisition, quick-start diagnostics, experiment reproduction, benchmarking, packaging, and repository hygiene — they are not part of the `active_contact_gs` Python package itself.

## Quick Start

```bash
# Minimal end-to-end diagnostic on a CUDA GPU
python scripts/quickstart.py --output outputs/quickstart --probes 2 --seeds 2

# Fast smoke test (shell)
bash scripts/smoke_test.sh outputs/smoke
```

## Script Overview

| Script | Purpose |
| --- | --- |
| `quickstart.py` | Run a small end-to-end ActiveContactGS identification experiment on CUDA and summarize the results. |
| `smoke_test.sh` | Run a tiny experiment + analysis pipeline to verify the installation works end to end. |
| `reproduce_main.sh` | Reproduce the main paper results: generate the amortized dataset, train the amortized model with 3 seeds, evaluate Active/Random/Fixed strategies, and aggregate the probe-budget analysis. |
| `download_ycb_subset.sh` | Download a subset of YCB objects (Google 16K meshes) via parallel chunked HTTP and extract them. |
| `parallel_http_download.py` | Multi-worker, resumable HTTP downloader using HTTP `Range` requests, used by the YCB download script. |
| `run_ycb_contact_benchmarks.sh` | Run contact-geometry benchmarks on pairs of YCB objects as anisotropic Gaussian ellipsoids. |
| `make_demo_videos.py` | Render qualitative trajectory GIFs comparing Active vs Random vs Fixed probing strategies. |
| `run_pinwm_official.py` | Headless launcher for the official PIN-WM Push-T pipeline (only runtime settings are changed; the upstream model/simulator/loss are untouched). |
| `build_repository_manifest.py` | Create a deterministic SHA-256 manifest (`REPOSITORY_MANIFEST.csv`) of the lightweight repository. |
| `package_repository.py` | Build a clean source ZIP, excluding Git metadata and generated artifacts. |
| `verify_repository.py` | Verify the public repository is lightweight, complete, and free of obvious secrets. |

## Typical Workflows

**Reproducing the main results**

```bash
bash scripts/reproduce_main.sh
```

This drives the full pipeline: dataset generation (`generate_amortized_dataset`), training with 3 seeds (`train_amortized`), evaluation under the active/random/fixed strategies (`evaluate_amortized`), and final analysis (`analyze_probe_budget`).

**Preparing YCB contact benchmarks**

```bash
bash scripts/download_ycb_subset.sh data/external
bash scripts/run_ycb_contact_benchmarks.sh data/processed/ycb16k/gaussians outputs/ycb-contact
```

**Packaging and verification (before a release)**

```bash
python scripts/build_repository_manifest.py   # refresh REPOSITORY_MANIFEST.csv
python scripts/verify_repository.py           # check the repo is lean and secret-free
python scripts/package_repository.py          # create the source ZIP
```

## Notes

- Most Python scripts assume you can import the package (e.g., via `pip install -e .` or by running from the repo root with `src/` on `PYTHONPATH`).
- Shell scripts are written for Bash and use `set -euo pipefail`.
- Scripts intentionally write generated artifacts (datasets, checkpoints, outputs) outside the repository (`data/`, `outputs/`) so the repo stays lightweight.
