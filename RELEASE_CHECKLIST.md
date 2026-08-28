# Public release checklist

- [ ] Confirm that the `CITATION.cff` author list matches the final paper author list.
- [ ] Add the paper/preprint URL and preferred paper citation once public.
- [x] Record the repository URL in package and citation metadata.
- [x] Route private vulnerability reports through GitHub Security Advisories.
- [ ] Confirm that all code contributors approve the MIT License.
- [ ] Re-check external dataset and upstream-project licenses.
- [ ] Run all tests on a CUDA GPU.
- [ ] Run `python scripts/verify_repository.py`.
- [ ] Rebuild `REPOSITORY_MANIFEST.csv` and package with `python scripts/package_repository.py --force`.
- [ ] Confirm that the Git history contains no checkpoints, datasets, results, credentials, or private reviewer material.
- [ ] Tag the frozen version, for example `v0.1.0`.
