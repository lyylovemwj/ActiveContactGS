# Quick Start

The Quick Start is a small end-to-end diagnostic: it samples rigid-body systems, compares Active, Random, and Fixed probes, writes structured JSON, and produces a statistical summary. It verifies the installation and control flow; it is not the full paper experiment.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,assets,dev]"
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## 2. Check CUDA

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

All experiment entry points are GPU-only. If CUDA is unavailable, installation and test collection can still be checked with:

```bash
python -c "import torch, active_contact_gs; print(torch.__version__)"
python -m pytest --collect-only -q
```

## 3. Run the diagnostic

```bash
python scripts/quickstart.py
```

The command writes:

```text
outputs/quickstart/
  experiment.json   per-strategy trajectories and errors
  analysis.json     aggregate curves and paired comparisons
```

To make the diagnostic slightly larger:

```bash
python scripts/quickstart.py --probes 3 --seeds 4 --particles 512
```

Do not use Quick Start numbers as paper results. For the frozen multi-seed protocol, dataset generation, amortized training, ID/OOD evaluation, geometry ablations, and external baselines, follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
