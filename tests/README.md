# tests — Test Suite

Pytest-based tests for the **ActiveContactGS** package. The suite covers the core physics simulation, contact geometry, belief inference, amortized models, experiments, and analysis code.

## Running the tests

```bash
# All tests (CPU-only tests will be skipped if no CUDA is available)
pytest tests/

# A single file
pytest tests/test_ellipsoid.py

# Full GPU suite (requires a CUDA-capable PyTorch)
pytest tests/ -m gpu
```

## Layout

| File | What it covers |
| --- | --- |
| `conftest.py` | Pytest policy: marks CUDA-dependent files as `gpu` and skips them on CPU-only runners. |
| `test_cpu_smoke.py` | Small analytic checks that run on standard CPU runners (public API + contact math). |
| `test_physics.py` | Planar rigid-body simulator (friction, restitution, contacts). |
| `test_ellipsoid.py` | Anisotropic Gaussian ellipsoid contact and time-of-impact. |
| `test_object_contact.py` | Gaussian object contact routines. |
| `test_belief.py` | Particle-based belief states. |
| `test_video_model.py` | Gaussian video posterior model. |
| `test_amortized.py` | Amortized inference / action-selection models. |
| `test_evaluate_amortized.py` | Active / Random / Fixed evaluation. |
| `test_experiment.py` | End-to-end experiment harness. |
| `test_particle_baseline.py` | Particle-filter baselines. |
| `test_analyze.py` | Analysis helpers. |
| `test_identifiability.py` | Identifiability checks. |
| `test_hypothesis.py` | Hypothesis-testing utilities. |
| `test_prepare_ycb_gaussians.py` | YCB → Gaussian ellipsoid preprocessing. |

## Notes

- Most files require a CUDA GPU and are auto-skipped on CPU-only CI runners.
- Install the package first (`pip install -e .`) so `active_contact_gs` is importable.
