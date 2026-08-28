#!/usr/bin/env bash
set -euo pipefail

asset_root="${1:-data/processed/ycb16k/gaussians}"
output_root="${2:-outputs/ycb-contact}"
mkdir -p "${output_root}"

python -m active_contact_gs.benchmark_object_contact \
  --object-a "${asset_root}/011_banana.pt" \
  --object-b "${asset_root}/024_bowl.pt" \
  --output "${output_root}/banana_bowl.json"

python -m active_contact_gs.benchmark_object_contact \
  --object-a "${asset_root}/003_cracker_box.pt" \
  --object-b "${asset_root}/006_mustard_bottle.pt" \
  --output "${output_root}/box_bottle.json"

python -m active_contact_gs.benchmark_object_contact \
  --object-a "${asset_root}/048_hammer.pt" \
  --object-b "${asset_root}/077_rubiks_cube.pt" \
  --output "${output_root}/hammer_cube.json"

python -m active_contact_gs.benchmark_object_contact \
  --object-a "${asset_root}/025_mug.pt" \
  --object-b "${asset_root}/035_power_drill.pt" \
  --output "${output_root}/mug_drill.json"
