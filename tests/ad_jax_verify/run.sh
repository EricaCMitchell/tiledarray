#!/usr/bin/env bash
# This file is a part of TiledArray.
# Copyright (C) 2026  Virginia Tech
#
# run.sh
# Single pass/fail gate for the AD dual-verification harness
# (ad_jax_dual_verification_plan.md): build the C++ producer, run it to emit
# golden.json, run the JAX verifier, and propagate the verifier's exit code.
#
# Usage:   tests/ad_jax_verify/run.sh
# Config:  TA_BUILD_DIR   build directory holding the ad_jax_produce target
#                         (default: build/ under the repo root)
#          GOLDEN         path for the golden file (default: a temp file)

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../.." && pwd)"

build_dir="${TA_BUILD_DIR:-${repo_root}/build}"
golden="${GOLDEN:-$(mktemp -t golden.XXXXXX.json)}"

if [[ ! -d "${build_dir}" ]]; then
  echo "error: build dir '${build_dir}' not found; set TA_BUILD_DIR" >&2
  exit 2
fi

echo ">> building ad_jax_produce in ${build_dir}"
cmake --build "${build_dir}" --target ad_jax_produce

producer="${build_dir}/tests/ad_jax_produce"
echo ">> producing ${golden}"
# np=1 (plan section 7): one process owns every tile, so the dump is complete
# and canonical without any cross-rank gather. MAD_NUM_THREADS keeps the runtime
# in its standard test configuration.
MAD_NUM_THREADS="${MAD_NUM_THREADS:-2}" "${producer}" "${golden}"

echo ">> verifying with JAX"
python3 "${here}/verify.py" "${golden}"
