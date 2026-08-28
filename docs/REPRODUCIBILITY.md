# Reproducibility guide

This repository contains no generated data, checkpoints, or result snapshots. Every experiment writes to ignored `data/` and `outputs/` directories.

## 1. Environment

The frozen experiment environment used Python 3.12, PyTorch 2.8 with CUDA 12.8, NumPy 1.26.4, Matplotlib 3.7.5, and Trimesh 4.7.1. Install with:

```bash
conda env create -f environment.yml
conda activate active-contact-gs
```

If CUDA 12.8 packages are unavailable for your platform, install the closest PyTorch/CUDA combination supported by your driver and record the deviation.

## 2. Generate the procedural dataset

```bash
python -m active_contact_gs.generate_amortized_dataset \
  --output data/amortized-v1 \
  --train-tasks 131072 --validation-tasks 16384 --test-tasks 16384 \
  --shard-size 8192 --probes 6 --noise-std 0.006
```

The generator freezes separate seeds for train, held-out ID, OOD-thin, and OOD-near-round splits in source code.

## 3. Train three independent main models

```bash
for seed in 7301 7302 7303; do
  python -m active_contact_gs.train_amortized \
    --data data/amortized-v1 --output outputs/main-seed${seed} \
    --steps 16000 --batch-size 2048 \
    --width 256 --layers 6 --heads 8 \
    --seed ${seed} --geometry-mode full --observation-mode pose
done
```

Training is CUDA-only. Keep the complete `history.json`, final checkpoint, environment information, and stdout for each seed outside Git or in a separate archival release.

## 4. Evaluate Active, Random, and Fixed policies

```bash
for seed in 7301 7302 7303; do
  python -m active_contact_gs.evaluate_amortized \
    --checkpoint outputs/main-seed${seed}/checkpoint-latest.pt \
    --output outputs/probe-budget-seed${seed}.json \
    --tasks 128 --probes 6 --strategies active random fixed
done

python -m active_contact_gs.analyze_probe_budget \
  --run outputs/probe-budget-seed7301.json \
  --run outputs/probe-budget-seed7302.json \
  --run outputs/probe-budget-seed7303.json \
  --output outputs/probe-budget-analysis
```

Use paired task-level resampling nested within independently trained model seeds. Do not compute formal intervals from only three seed means.

## 5. Geometry ablations

Train `isotropic` and `no_shape` with the same dataset, architecture, optimization budget, and seed as the full model:

```bash
for mode in isotropic no_shape; do
  python -m active_contact_gs.train_amortized \
    --data data/amortized-v1 --output outputs/ablation-${mode} \
    --steps 16000 --batch-size 2048 \
    --width 256 --layers 6 --heads 8 --seed 7301 \
    --geometry-mode ${mode} --observation-mode pose
done
```

Evaluate with the same task count, split seeds, policy set, posterior sample count, and probe budget as the full model. Use `active_contact_gs.analyze_geometry_active_interaction` for the interaction analysis; inspect its exact arguments with `--help`.

## 6. Gaussian-video model

```bash
python -m active_contact_gs.train_amortized \
  --data data/amortized-v1 --output outputs/video-v2 \
  --steps 16000 --batch-size 1024 \
  --width 256 --layers 6 --heads 8 --seed 7301 \
  --geometry-mode full --observation-mode gaussian_video \
  --image-resolution 32 --video-frames 6
```

Evaluate the video checkpoint with `active_contact_gs.evaluate_amortized`. The observation is rendered Gaussian-video interaction history plus the known Gaussian geometry token; it is not end-to-end scene reconstruction from an unconstrained camera stream.

## 7. YCB geometry/contact protocol

```bash
bash scripts/download_ycb_subset.sh data/external
python -m active_contact_gs.prepare_ycb_gaussians \
  --input data/external/ycb16k/objects \
  --output data/processed/ycb16k/gaussians
bash scripts/run_ycb_contact_benchmarks.sh \
  data/processed/ycb16k/gaussians outputs/ycb-contact
```

YCB provides real object geometry, not real-robot dynamics. Record downloaded-file checksums and the terms served by the official host.

## 8. PIN-WM native-protocol reproduction

```bash
git clone https://github.com/XuAdventurer/PIN-WM third_party/PIN-WM
git -C third_party/PIN-WM checkout 99d3fde5d233aeffabfa287f94831cf7c7afee64

python scripts/run_pinwm_official.py \
  --pin-root third_party/PIN-WM --seed 0 \
  --frames 32 --height 800 --width 800 --iterations 125 \
  --metadata outputs/pinwm-seed0-metadata.json
```

Repeat with seeds 1 and 2. This launcher changes runtime arguments and disables GUI rendering; it leaves the upstream method, renderer, differentiable simulator, loss, and optimizer intact. Target dynamics should be read from PyBullet `getDynamicsInfo`. Native PIN-WM results are not automatically a common-task head-to-head comparison.

## 9. Verification and provenance

```bash
python -m pytest -q
python -m active_contact_gs.freeze_repro_manifest \
  --root . --output outputs/repro-manifest.json
python scripts/verify_repository.py
```

Archive the exact source revision, environment, command lines, generated-data manifest, checkpoints, per-task outputs, and logs separately from the lightweight Git repository.
