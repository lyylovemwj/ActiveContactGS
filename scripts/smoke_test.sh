#!/usr/bin/env bash
set -euo pipefail

smoke_root="${1:-outputs/smoke}"
mkdir -p "${smoke_root}"

python -m active_contact_gs.experiment \
  --device cuda --probes 2 --seeds 2 --particles 128 \
  --output "${smoke_root}/experiment.json"

python -m active_contact_gs.analyze \
  "${smoke_root}/experiment.json" \
  --output "${smoke_root}/analysis.json"

printf 'Smoke test completed: %s\n' "${smoke_root}"
