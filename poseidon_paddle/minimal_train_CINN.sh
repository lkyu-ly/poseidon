#!/usr/bin/env bash
set -euo pipefail

# 打开组合算子
export FLAGS_prim_enable_dynamic=true && export FLAGS_prim_all=true

# 打开 CINN 编译器
export FLAGS_use_cinn=true

# 是否打印 Program IR 信息 (用于调试)
export FLAGS_print_ir=false

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export WANDB_MODE=disabled
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

python "${SCRIPT_DIR}/scOT/train.py" \
  --config "${SCRIPT_DIR}/configs/run_small.yaml" \
  --wandb_run_name "se-af-scratch-small" \
  --wandb_project_name "scOT" \
  --checkpoint_path "${REPO_ROOT}/tmp_checkpoints" \
  --data_path "${SCRIPT_DIR}/../dataset"
