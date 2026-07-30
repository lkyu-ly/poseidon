#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# 单一开关：train.py 据此在 import paddle 之前设 CINN FLAGS 并对 model 做 to_static。
export POSEIDON_USE_CINN=1

export WANDB_MODE=disabled
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

python "${SCRIPT_DIR}/scOT/train.py" \
  --config "${SCRIPT_DIR}/configs/run_small.yaml" \
  --wandb_run_name "se-af-scratch-small" \
  --wandb_project_name "scOT" \
  --checkpoint_path "${REPO_ROOT}/tmp_checkpoints" \
  --data_path "${SCRIPT_DIR}/../dataset"
