#!/usr/bin/env bash
set -euo pipefail

dataset_root="${ACGS_DATA_ROOT:-data/amortized-v1}"
run_root="${ACGS_OUTPUT_ROOT:-outputs}"

python -m active_contact_gs.generate_amortized_dataset \
  --output "${dataset_root}" \
  --train-tasks 131072 --validation-tasks 16384 --test-tasks 16384 \
  --shard-size 8192 --probes 6 --noise-std 0.006

for seed in 7301 7302 7303; do
  python -m active_contact_gs.train_amortized \
    --data "${dataset_root}" --output "${run_root}/main-seed${seed}" \
    --steps 16000 --batch-size 2048 \
    --width 256 --layers 6 --heads 8 \
    --seed "${seed}" --geometry-mode full --observation-mode pose

  python -m active_contact_gs.evaluate_amortized \
    --checkpoint "${run_root}/main-seed${seed}/checkpoint-latest.pt" \
    --output "${run_root}/probe-budget-seed${seed}.json" \
    --tasks 128 --probes 6 --strategies active random fixed
done

python -m active_contact_gs.analyze_probe_budget \
  --run "${run_root}/probe-budget-seed7301.json" \
  --run "${run_root}/probe-budget-seed7302.json" \
  --run "${run_root}/probe-budget-seed7303.json" \
  --output "${run_root}/probe-budget-analysis"
