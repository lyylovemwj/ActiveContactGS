#!/usr/bin/env bash
set -euo pipefail

data_root="${ACGS_ICLR_DATA_ROOT:-data/iclr-v1}"
output_root="${ACGS_ICLR_OUTPUT_ROOT:-outputs/iclr-review}"
train_tasks="${ACGS_ICLR_TRAIN_TASKS:-262144}"
validation_tasks="${ACGS_ICLR_VALIDATION_TASKS:-16384}"
test_tasks="${ACGS_ICLR_TEST_TASKS:-16384}"
steps="${ACGS_ICLR_STEPS:-16000}"
batch_size="${ACGS_ICLR_BATCH_SIZE:-1024}"
eval_tasks="${ACGS_ICLR_EVAL_TASKS:-128}"

python -m active_contact_gs.generate_iclr_dataset \
  --output "${data_root}" --train-tasks "${train_tasks}" \
  --validation-tasks "${validation_tasks}" --test-tasks "${test_tasks}"

for representation in flat independent autoregressive structured; do
  for seed in 7301 7302 7303; do
    run="${output_root}/${representation}-seed${seed}"
    python -m active_contact_gs.train_iclr \
      --data "${data_root}" --output "${run}" \
      --representation "${representation}" --seed "${seed}" \
      --steps "${steps}" --batch-size "${batch_size}"
    python -m active_contact_gs.evaluate_iclr \
      --checkpoint "${run}/checkpoint-latest.pt" \
      --output "${run}/evaluation.json" --tasks "${eval_tasks}" \
      --strategies active eig_matched balanced_fixed random fixed
  done
done

checkpoint="${output_root}/independent-seed7301/checkpoint-latest.pt"
for samples in 2 4 8 16 32 64; do
  python -m active_contact_gs.evaluate_iclr \
    --checkpoint "${checkpoint}" --output "${output_root}/samples-${samples}.json" \
    --tasks "${eval_tasks}" --posterior-samples "${samples}" \
    --strategies active --splits id
done

python -m active_contact_gs.analyze_iclr_sampling \
  --checkpoint "${checkpoint}" \
  --output "${output_root}/sampling-diagnostics.json"

for mismatch in 0.0 0.05 0.10 0.20 0.30; do
  label="${mismatch/./p}"
  python -m active_contact_gs.evaluate_iclr \
    --checkpoint "${checkpoint}" --output "${output_root}/mismatch-${label}.json" \
    --tasks "${eval_tasks}" --mismatch "${mismatch}" \
    --strategies active balanced_fixed random --splits id
done

python -m active_contact_gs.analyze_iclr_identifiability \
  --checkpoint "${checkpoint}" --output "${output_root}/identifiability.json" \
  --tasks "${eval_tasks}"

python -m active_contact_gs.plot_iclr_review_assets \
  --results "${output_root}" --output paper_assets/iclr_review
