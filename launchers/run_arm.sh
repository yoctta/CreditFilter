#!/bin/bash
# Submits one training arm of the paper to a Ray cluster.
#
#   export RAY_HEAD=<head_ip:8265> GEN_PATH=/path/to/Qwen3-8B-Base
#   bash launchers/run_arm.sh <arm>            # arms: see ARM_LIST in launchers/arms.sh
#
# Optional: ACTOR_NNODES (nodes of 8 GPUs, default 2), DATA_SEED (default 201), HW_PROFILE (h100 | b200),
#           SMOKE=1 (3 steps on a small batch), DRY_RUN=1 (write the job to the run directory, submit nothing).
set -e
set -o pipefail
ARM_NAME=${1:?arm required (see ARM_LIST in launchers/arms.sh)}
PKG_ROOT=${PKG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
PKG_ROOT=$(cd "$PKG_ROOT" && pwd -P)
source "$PKG_ROOT/launchers/arms.sh"
export RAY_HEAD=${RAY_HEAD:?RAY_HEAD required (head_ip:8265)}

# the paper's runs use the official Qwen/Qwen3-8B-Base with the ChatML template of its tokenizer_config.json
export GEN_PATH=${GEN_PATH:-$PKG_ROOT/models/Qwen3-8B-Base}
[ -f "$GEN_PATH/tokenizer_config.json" ] || { echo "[run_arm] no model at GEN_PATH=$GEN_PATH" >&2; exit 4; }
_tj=$(md5sum "$GEN_PATH/tokenizer.json" | cut -c1-32)
_tc=$(md5sum "$GEN_PATH/tokenizer_config.json" | cut -c1-32)
if [ "$_tj" != 3f99a313c3754208a7f3acfccadc53cd ] || [ "$_tc" != 5a655ca7878bcd7dd398020de8f7df9b ]; then
  echo "[run_arm] WARNING: tokenizer.json / tokenizer_config.json differ from those of Qwen/Qwen3-8B-Base used in the paper" >&2
fi
if [ -e "$GEN_PATH/chat_template.jinja" ]; then
  echo "[run_arm] WARNING: $GEN_PATH/chat_template.jinja takes precedence over the template of tokenizer_config.json" >&2
fi

export CODE_ROOT="$PKG_ROOT"
export TRAIN_FILE="$PKG_ROOT/data/dapo17k_boxed.parquet"
export VAL_FILES="['$PKG_ROOT/data/eval/MATH500/test.parquet','$PKG_ROOT/data/eval/AIME2024/test.parquet','$PKG_ROOT/data/eval/AIME2025/test.parquet']"
export ACTOR_NNODES=${ACTOR_NNODES:-2}
export HW_PROFILE=${HW_PROFILE:-h100}

resolve_arm "$ARM_NAME"
export STAGE="$ARM_NAME"
if [ "${SMOKE:-0}" = "1" ]; then
  export RUN_TAG=${RUN_TAG:-${ARM_NAME}_smoke_$(date +%Y%m%d_%H%M%S)}
else
  export RUN_TAG=${RUN_TAG:-${ARM_NAME}_$(date +%Y%m%d_%H%M%S)}
fi
bash "$PKG_ROOT/creditfilter/scripts/launch_recipe.sh"
echo "[run_arm] run directory: $PKG_ROOT/runs/$RUN_TAG"
