# Public release checklist

- [ ] Replace placeholder authors in `CITATION.cff` with the final paper author list.
- [ ] Add the paper/preprint URL and repository URL.
- [ ] Add private maintainer contacts to `SECURITY.md` and `CODE_OF_CONDUCT.md`.
- [ ] Confirm that all code contributors approve the MIT License.
- [ ] Re-check external dataset and upstream-project licenses.
- [ ] Run all tests on a CUDA GPU.
- [ ] Run `python scripts/verify_repository.py`.
- [ ] Rebuild `REPOSITORY_MANIFEST.csv` and package with `python scripts/package_repository.py --force`.
- [ ] Confirm that the Git history contains no checkpoints, datasets, results, credentials, or private reviewer material.
- [ ] Tag the frozen version, for example `v0.1.0`.
