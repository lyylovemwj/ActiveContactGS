#!/usr/bin/env bash
set -euo pipefail

data_root="${1:-data/external}"
archive_dir="${data_root}/ycb16k/archives"
extract_dir="${data_root}/ycb16k/objects"
mkdir -p "${archive_dir}" "${extract_dir}"

objects=(
  003_cracker_box
  006_mustard_bottle
  011_banana
  024_bowl
  025_mug
  035_power_drill
  048_hammer
  077_rubiks_cube
)

for object in "${objects[@]}"; do
  archive="${archive_dir}/${object}_google_16k.tgz"
  url="https://ycb-benchmarks.s3.amazonaws.com/data/google/${object}_google_16k.tgz"
  python3 ./scripts/parallel_http_download.py \
    "${url}" "${archive}" --workers 16 --chunk-size 1048576
done

sha256sum "${archive_dir}"/*.tgz > "${data_root}/ycb16k/SHA256SUMS"

for archive in "${archive_dir}"/*.tgz; do
  tar -xzf "${archive}" -C "${extract_dir}"
done

curl --fail --location --retry 5 \
  --output "${data_root}/ycb16k/LICENSE_SOURCE.html" \
  "https://ycb-benchmarks.s3.amazonaws.com/index.html"

find "${extract_dir}" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${data_root}/ycb16k/EXTRACTED_SHA256SUMS"

printf 'YCB subset downloaded to %s\n' "${data_root}/ycb16k"
