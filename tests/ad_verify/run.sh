#!/usr/bin/env bash
# This file is a part of TiledArray.
# Copyright (C) 2026  Virginia Tech
#
# run.sh
# One pass/fail gate for the AD dual-verification harness (tests/ad_verify/).
# It builds the C++ producer, runs it to write golden.json, runs the selected
# oracles, and exits nonzero if a verifier fails.
#
# Usage:   tests/ad_verify/run.sh [jax|torch|both]
#          (default: both)
#
# Config:  TA_BUILD_DIR   build directory holding the ad_produce target
#                         (default: build/ under the repo root)
#          GOLDEN         path for the golden file (default: a temp file)
#          BACKEND        fallback backend selector if no positional arg given

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../.." && pwd)"

build_dir="${TA_BUILD_DIR:-${repo_root}/build}"
golden="${GOLDEN:-$(mktemp -t golden.XXXXXX.json)}"
backend="${1:-${BACKEND:-both}}"

case "${backend}" in
  jax|torch|both) ;;
  *)
    echo "error: backend must be jax, torch, or both (got '${backend}')" >&2
    exit 2
    ;;
esac

if [[ ! -d "${build_dir}" ]]; then
  echo "error: build dir '${build_dir}' not found; set TA_BUILD_DIR" >&2
  exit 2
fi

echo ">> building ad_produce in ${build_dir}"
cmake --build "${build_dir}" --target ad_produce

producer="${build_dir}/tests/ad_produce"
echo ">> producing ${golden}"
# At np=1 one process owns every tile, so the dump is complete and needs no
# cross-rank gather. MAD_NUM_THREADS keeps the standard test configuration.
MAD_NUM_THREADS="${MAD_NUM_THREADS:-2}" "${producer}" "${golden}"

overall=0

if [[ "${backend}" == "jax" || "${backend}" == "both" ]]; then
  echo ">> verifying with JAX (verify_jax.py)"
  python3 "${here}/verify_jax.py" "${golden}" || overall=1
fi

if [[ "${backend}" == "torch" || "${backend}" == "both" ]]; then
  echo ">> verifying with PyTorch (verify_torch.py)"
  python3 "${here}/verify_torch.py" "${golden}" || overall=1
fi

exit "${overall}"
