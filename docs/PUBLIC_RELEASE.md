# Maintainer release guide

## Lightweight repository

This repository should remain lightweight. Keep generated data, checkpoints,
logs, and full result JSON files out of ordinary Git history.

The bounded exception is `paper_assets/iclr_review/`: it contains the compact
PNG/PDF figures, LaTeX table, and CSV sources needed to audit the ICLR reviewer
experiments. Every file must remain below the repository size limit and be
listed in both its local manifest and `REPOSITORY_MANIFEST.csv`.

## Archival artifacts

If model weights and formal results are later published, create a versioned archival artifact with:

- SHA-256 checksums;
- training seed and exact configuration per checkpoint;
- per-task structured results rather than only aggregate tables;
- environment and hardware manifests;
- external asset provenance and licenses;
- a permanent DOI or release URL.

Link that artifact from the README without copying it into this repository.

## Before tagging

Run the GPU test suite, validate all CLI commands used in the paper, scan the Git history for secrets and large binaries, freeze the citation metadata, and complete `RELEASE_CHECKLIST.md`.

Create a clean source archive without local `.git` metadata or generated outputs:

```bash
python scripts/build_repository_manifest.py
python scripts/verify_repository.py
python scripts/package_repository.py --force
```
