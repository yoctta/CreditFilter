#!/bin/bash
# The training recipe shared by all arms, and the settings of each route (ARM); calls launch_train.sh.
#
# Recipe: Qwen3-8B-Base on DAPO-17k (boxed-answer prompts), 128 prompts x 16 rollouts per step, mini-batch 32,
# learning rate 1e-6, no KL loss, token-mean loss, response cap 20480 with the DAPO soft overlong penalty (buffer 4096),
# DAPO dynamic sampling, 400 steps, checkpoint every 25, in-loop validation every 20, data seed 201.
#
# ARM: dapo | grpo | entropy_mask | creditfilter | stare | scgrpo.  SMOKE=1: 16 prompts x 8 rollouts, 3 steps.
set -e
ARM=${ARM:?ARM=dapo|grpo|entropy_mask|creditfilter|stare|scgrpo required}
export RAY_HEAD=${RAY_HEAD:?RAY_HEAD required (head_ip:8265)}
RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CODE_ROOT=${CODE_ROOT:-$(cd "${RECIPE_DIR}/../.." && pwd)}
export GEN_PATH=${GEN_PATH:-${CODE_ROOT}/models/Qwen3-8B-Base}
export TRAIN_FILE=${TRAIN_FILE:-${CODE_ROOT}/data/dapo17k_boxed.parquet}
export VAL_FILES=${VAL_FILES:-"['${CODE_ROOT}/data/eval/MATH500/test.parquet','${CODE_ROOT}/data/eval/AIME2024/test.parquet','${CODE_ROOT}/data/eval/AIME2025/test.parquet']"}
export STAGE=${STAGE:-${ARM}}
export ACTOR_NNODES=${ACTOR_NNODES:-1}
export N_GPUS_PER_NODE=8
export ROLLOUT_N=${ROLLOUT_N:-16}
export TRAIN_BATCH=${TRAIN_BATCH:-128}
export PPO_MINI_BATCH=${PPO_MINI_BATCH:-32}
export MAX_RESP_LEN=${MAX_RESP_LEN:-20480}
export MAX_PROMPT_LEN=1024
export TOTAL_STEPS=${TOTAL_STEPS:-400}
export TEST_FREQ=20
export SAVE_FREQ=${SAVE_FREQ:-25}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}
export DATA_SEED=${DATA_SEED:-201}
export KL_LOSS_COEF=0
export KL_CTRL_COEF=0
export USE_KL_LOSS=False
export FSDP_OFFLOAD=${FSDP_OFFLOAD:-True}

# Hardware profile: throughput settings only (CPU offload, tensor parallelism of the rollout engine, its memory share,
# micro-batch sizes). The update does not depend on them.
#   h100: 8 x 80 GB    b200: 8 x 180 GB
export HW_PROFILE=${HW_PROFILE:-h100}
case "$HW_PROFILE" in
  h100) export TP_SIZE=${TP_SIZE:-2} PPO_MICRO_BSZ=${PPO_MICRO_BSZ:-1} LOGP_MICRO_BSZ=${LOGP_MICRO_BSZ:-4} ;;
  b200) export FSDP_OFFLOAD=False GPU_MEM_UTIL=0.85 TP_SIZE=${TP_SIZE:-1} PPO_MICRO_BSZ=${PPO_MICRO_BSZ:-2} LOGP_MICRO_BSZ=${LOGP_MICRO_BSZ:-8} ;;
  *) echo "[recipe] unknown HW_PROFILE='$HW_PROFILE' (h100|b200)" >&2; exit 2 ;;
esac

export CLIP_RATIO_LOW=0.2
export CLIP_RATIO_HIGH=0.28
export CLIP_RATIO_C=3.0
export LOSS_AGG_MODE=token-mean
export CF_OVERLONG_BUFFER_ENABLE=1
export CF_OVERLONG_BUFFER_LEN=${CF_OVERLONG_BUFFER_LEN:-4096}
export CF_OVERLONG_PENALTY_FACTOR=1.0
# DAPO dynamic sampling: only prompt groups with mixed outcomes are trained on; at most 8 generation chunks per step
export EXTRA_HYDRA="++algorithm.filter_groups.enable=true ++algorithm.filter_groups.metric=acc ++algorithm.filter_groups.max_num_gen_batches=8 ++data.gen_batch_size=${TRAIN_BATCH} ${EXTRA_OVERRIDES:-}"

if [ "${SMOKE:-0}" = "1" ]; then
  export TRAIN_BATCH=16 PPO_MINI_BATCH=8 ROLLOUT_N=8 TOTAL_STEPS=3 TEST_FREQ=1000 SAVE_FREQ=1000
  export EXTRA_HYDRA="++algorithm.filter_groups.enable=true ++algorithm.filter_groups.metric=acc ++algorithm.filter_groups.max_num_gen_batches=8 ++data.gen_batch_size=16 ${EXTRA_OVERRIDES:-}"
  export STAGE=smoke_${ARM}
fi

# the clip range is 0.2 / 0.28 (DAPO) unless the arm changes it
case "$ARM" in
  dapo)
    export CF_ROUTE=off ;;
  grpo)                 # DAPO without clip-higher
    export CF_ROUTE=off CLIP_RATIO_HIGH=0.2 ;;
  entropy_mask)         # 80/20 rule: the top 20% entropy tokens keep their gradient
    export CF_ROUTE=entropy_mask CF_MASK_TOP=0.2 CF_MASK_SCOPE=${CF_MASK_SCOPE:-response} ;;
  creditfilter)         # the window itself is set by launchers/arms.sh
    export CF_ROUTE=creditfilter CLIP_RATIO_HIGH=0.27
    export CF_FALLBACK_H_LO=${CF_FALLBACK_H_LO:-1.5} CF_FALLBACK_H_HI=${CF_FALLBACK_H_HI:-3.0}
    export CF_DAMPING_POS=${CF_DAMPING_POS:-1.0} CF_DAMPING_NEG=${CF_DAMPING_NEG:-0.0} ;;
  stare)                # STARE (arXiv 2606.19236) on the DAPO recipe, with its batch-level entropy gate and fixed weights
    export CF_ROUTE=stare
    export CF_STARE_W=${CF_STARE_W:-1.1} CF_STARE_M=${CF_STARE_M:-1.0} CF_STARE_P=${CF_STARE_P:-0.10}
    export CF_STARE_H_TARGET=${CF_STARE_H_TARGET:-0.3} ;;
  scgrpo)               # SC-GRPO (arXiv 2606.18810) on the GRPO recipe
    export CF_ROUTE=scgrpo CLIP_RATIO_HIGH=0.2
    export CF_SCGRPO_PCT=${CF_SCGRPO_PCT:-75} CF_SCGRPO_CMIN=${CF_SCGRPO_CMIN:-1e-4} CF_SCGRPO_NCMIN=${CF_SCGRPO_NCMIN:-2}
    export CF_SCGRPO_DEMO_MAX_TOK=${CF_SCGRPO_DEMO_MAX_TOK:-8192} CF_SCGRPO_KL_MAX_RESP=${CF_SCGRPO_KL_MAX_RESP:-12288} ;;
  *) echo "[recipe] unknown ARM=$ARM (dapo|grpo|entropy_mask|creditfilter|stare|scgrpo)" >&2; exit 1 ;;
esac

export RUN_TAG=${RUN_TAG:-${STAGE}_$(date +%Y%m%d_%H%M%S)}
export OUT_DIR=${OUT_DIR:-${CODE_ROOT}/runs/${RUN_TAG}}
export OUT_DIR_LOCAL=${OUT_DIR_LOCAL:-${OUT_DIR}}
echo "[recipe] arm=$ARM run=$RUN_TAG model=$GEN_PATH batch=${TRAIN_BATCH}x${ROLLOUT_N} steps=$TOTAL_STEPS hw=$HW_PROFILE nodes=$ACTOR_NNODES"
bash "${RECIPE_DIR}/launch_train.sh"
