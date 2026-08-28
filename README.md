# ActiveContactGS

[![CI](https://github.com/lyylovemwj/ActiveContactGS/actions/workflows/ci.yml/badge.svg)](https://github.com/lyylovemwj/ActiveContactGS/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Research code for **Contact Geometry Is a Latent Variable: Active Bayesian Identification of Gaussian Physical Worlds**.

<p align="center">
  <img src="assets/images/fig1_method_overview.png" alt="Figure 1: ActiveContactGS method overview" width="100%">
</p>
<p align="center"><em>Figure 1. Geometry and interaction history form a hybrid posterior used for counterfactual information-gain action selection.</em></p>

ActiveContactGS studies active rigid-body system identification from object-centric anisotropic Gaussian geometry. The codebase provides:

- differentiable planar rigid-body simulation with friction, restitution, off-centre impulses, and wall contact;
- contact computation on native anisotropic Gaussian ellipsoids;
- a hybrid posterior over latent contact structure and continuous physical parameters;
- amortized information-gain action selection from safe interaction candidates;
- pose-observation and Gaussian-video posterior models;
- ID/OOD evaluation, geometry ablations, identifiability analysis, YCB contact preparation, and PIN-WM reproduction helpers.

This lightweight GitHub package contains code, reproduction instructions, and five figures extracted losslessly from the manuscript. It does not include checkpoints, generated datasets, raw experimental results, or logs.

## Quick Start

After installing the package on a CUDA machine, run:

```bash
python scripts/quickstart.py
```

This executes a small Active/Random/Fixed identification diagnostic and writes `outputs/quickstart/experiment.json` plus `analysis.json`. It checks the end-to-end path but does not reproduce the paper-scale experiment. See the [step-by-step Quick Start](docs/QUICKSTART.md).

## Installation

Python 3.10+ and PyTorch 2.1+ are required. Full training and evaluation require a CUDA GPU.

```bash
git clone https://github.com/lyylovemwj/ActiveContactGS.git
cd ActiveContactGS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,assets,dev]"
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

For the frozen CUDA environment used in the experiments, see [environment.yml](environment.yml).

## Installation check

```bash
python -c "import torch, active_contact_gs; print(torch.__version__)"
python -m pytest --collect-only -q
```

Experiment entry points are intentionally GPU-only. Use `scripts/quickstart.py` for the cross-platform CUDA diagnostic or `scripts/smoke_test.sh` from Bash.

## Results at a glance

<p align="center">
  <img src="assets/images/fig3_sample_efficiency.png" alt="Figure 3: Sample efficiency and contact-normal fidelity" width="100%">
</p>
<p align="center"><em>Figure 3. Active probing reduces parameter error across ID and OOD splits; native anisotropic contact preserves contact normals far more accurately than sphere proxies.</em></p>

### Geometry-aware interaction

<table>
  <tr>
    <td width="38%"><img src="assets/images/fig5_anisotropic_contact.png" alt="Figure 5: Native anisotropic contact on Venus"></td>
    <td width="62%"><img src="assets/images/fig7_cross_object.png" alt="Figure 7: Cross-object active probing"></td>
  </tr>
  <tr>
    <td align="center">Figure 5. Native contact versus sphere approximation</td>
    <td align="center">Figure 7. Cup, bunny, and Venus active-probing sequences</td>
  </tr>
</table>

<details>
<summary><strong>Figure 4: geometry and action-selection ablation</strong></summary>
<p align="center">
  <img src="assets/images/fig4_geometry_ablation.png" alt="Figure 4: Geometry and action-selection ablation" width="90%">
</p>
</details>

These are manuscript figures, not third-party illustrations. Their raw result files are intentionally kept outside the lightweight Git repository; the commands required to reproduce them are documented below. Figure provenance is recorded in [`assets/images/README.md`](assets/images/README.md).

## Full CUDA pipeline

```bash
# Generate deterministic procedural shards.
python -m active_contact_gs.generate_amortized_dataset \
  --output data/amortized-v1 --train-tasks 131072 \
  --validation-tasks 16384 --test-tasks 16384 --probes 6

# Train one model.
python -m active_contact_gs.train_amortized \
  --data data/amortized-v1 --output outputs/main-seed7301 \
  --steps 16000 --batch-size 2048 --width 256 --layers 6 --heads 8 \
  --seed 7301 --geometry-mode full --observation-mode pose

# Evaluate Active, Random, and Fixed policies.
python -m active_contact_gs.evaluate_amortized \
  --checkpoint outputs/main-seed7301/checkpoint-latest.pt \
  --output outputs/main-seed7301/eval.json \
  --tasks 128 --probes 6 --strategies active random fixed
```

The complete three-seed, ablation, video, YCB, and PIN-WM protocols are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). CLI options are the source of truth and can be inspected with `python -m <module> --help`.

## Tests

The formal test suite exercises CUDA kernels and therefore requires a CUDA-capable runner:

```bash
python -m pytest -q
```

Standard GitHub-hosted CPU runners execute two analytic smoke tests and skip 26 explicitly marked CUDA tests. A CUDA runner executes all 28 tests. Run the complete GPU suite before tagging a release.

## Repository structure

```text
src/active_contact_gs/   Core method, training, evaluation, and analysis code
tests/                   CUDA-focused test suite
scripts/                 Reproduction and external-asset helpers
docs/                    Data, protocol, and reproducibility documentation
assets/sources/          External-source provenance metadata
assets/images/           Five losslessly extracted manuscript figures
.github/                 CI and contribution templates
```

## External data and projects

The procedural amortized dataset is generated by this repository. YCB files and PIN-WM are not redistributed; download them from their official sources and comply with their licenses. See [docs/DATA.md](docs/DATA.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you use this code, please cite it via [CITATION.cff](CITATION.cff) and cite the accompanying paper once it is published.

## License

Original ActiveContactGS code is released under the [MIT License](LICENSE). External code and datasets remain under their respective licenses.
