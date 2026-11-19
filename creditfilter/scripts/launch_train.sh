#!/bin/bash
# Submits one verl GRPO training job to a Ray cluster. Called by launch_recipe.sh, which sets the recipe.
# The CF_* knobs below are the complete set the job receives; they are written to <run dir>/runtime_env.json.
# DRY_RUN=1 writes runtime_env.json and launch_payload.json and submits nothing.
set -e

RAY_HEAD=${RAY_HEAD:?RAY_HEAD required (Ray dashboard address, host:port)}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=${CODE_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}      # this tree as seen by the job
GEN_PATH=${GEN_PATH:?GEN_PATH required (policy model directory)}
TRAIN_FILE=${TRAIN_FILE:-${CODE_ROOT}/data/dapo17k_boxed.parquet}
VAL_FILES=${VAL_FILES:-"['${CODE_ROOT}/data/eval/MATH500/test.parquet','${CODE_ROOT}/data/eval/AIME2024/test.parquet','${CODE_ROOT}/data/eval/AIME2025/test.parquet']"}
ROLLOUT_N=${ROLLOUT_N:-16}
TRAIN_BATCH=${TRAIN_BATCH:-128}
PPO_MINI_BATCH=${PPO_MINI_BATCH:-32}
MAX_RESP_LEN=${MAX_RESP_LEN:-20480}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
TP_SIZE=${TP_SIZE:-2}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-kl}
ACTOR_NNODES=${ACTOR_NNODES:-1}
TEST_FREQ=${TEST_FREQ:-20}
SAVE_FREQ=${SAVE_FREQ:-25}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}
TOTAL_STEPS=${TOTAL_STEPS:-400}
RUN_TAG=${RUN_TAG:-${STAGE:-train}_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${CODE_ROOT}/runs/${RUN_TAG}}      # run directory as seen by the job
OUT_DIR_LOCAL=${OUT_DIR_LOCAL:-${OUT_DIR}}            # the same directory on the submitting host
JOB_HOME=${JOB_HOME:-/root}                           # HOME of the job
JOB_CACHE_ROOT=${JOB_CACHE_ROOT:-/tmp/creditfilter_cache}   # compiler and engine caches: node-local, writable, executable

mkdir -p "$OUT_DIR_LOCAL"

CMD="set -x; mkdir -p ${OUT_DIR} ${OUT_DIR}/metrics ${OUT_DIR}/tb ~/.cache/flashinfer ~/.triton; \
cd ${CODE_ROOT}; \
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=${CODE_ROOT}; \
export VERL_FILE_LOGGER_PATH=${OUT_DIR}/metrics/metrics.jsonl \
VERL_FILE_LOGGER_ROOT=${OUT_DIR}/metrics \
TENSORBOARD_DIR=${OUT_DIR}/tb; \
python3 -m verl.trainer.main_ppo \
hydra.run.dir=${OUT_DIR}/hydra \
algorithm.adv_estimator=grpo \
data.train_files=\"['${TRAIN_FILE}']\" \
data.val_files=\"${VAL_FILES}\" \
data.train_batch_size=${TRAIN_BATCH} \
data.max_prompt_length=${MAX_PROMPT_LEN} \
data.max_response_length=${MAX_RESP_LEN} \
data.filter_overlong_prompts=True \
data.truncation='error' \
data.return_raw_chat=True \
data.seed=${DATA_SEED:-null} \
actor_rollout_ref.model.path=${GEN_PATH} \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH} \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BSZ:-1} \
actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS:-False} \
actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_OFFLOAD:-False} \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OFFLOAD:-False} \
actor_rollout_ref.ref.fsdp_config.param_offload=${FSDP_OFFLOAD:-False} \
actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF:-0} \
actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2} \
actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28} \
actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C:-3.0} \
actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE:-token-mean} \
actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
algorithm.use_kl_in_reward=False \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGP_MICRO_BSZ:-4} \
actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
actor_rollout_ref.rollout.n=${ROLLOUT_N} \
++actor_rollout_ref.rollout.calculate_log_probs=false \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGP_MICRO_BSZ:-4} \
++actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config={pass_config:{fuse_allreduce_rms:false}} \
++actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true \
++reward.custom_reward_function.path=${CODE_ROOT}/creditfilter/custom_score.py \
++reward.custom_reward_function.name=compute_score \
algorithm.kl_ctrl.kl_coef=${KL_CTRL_COEF:-0} \
reward.reward_manager.source=importlib \
+reward.reward_manager.module.path=${CODE_ROOT}/creditfilter/outcome_reward.py \
reward.reward_manager.name=OutcomeRewardManager \
trainer.critic_warmup=0 \
trainer.logger=[console,file,tensorboard] \
trainer.project_name=CreditFilter \
trainer.experiment_name=${RUN_TAG} \
trainer.default_local_dir=${OUT_DIR} \
trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-8} \
trainer.nnodes=${ACTOR_NNODES} \
trainer.save_freq=${SAVE_FREQ} \
trainer.test_freq=${TEST_FREQ} \
trainer.total_training_steps=${TOTAL_STEPS} ${EXTRA_HYDRA:-}"

# RESUME_CKPT=<run dir>/global_step_N continues from that checkpoint up to TOTAL_STEPS, writing into this run's OUT_DIR
if [ -n "${RESUME_CKPT:-}" ]; then
  case "$RESUME_CKPT" in *global_step_*) ;; *) echo "[launch] RESUME_CKPT must point to a global_step_N directory"; exit 1;; esac
  ls "$RESUME_CKPT"/actor/optim_world_size_*_rank_0.pt >/dev/null 2>&1 || {
    echo "[launch] $RESUME_CKPT has no optimizer state (optim_world_size_*_rank_0.pt)"; exit 1; }
  CMD="$CMD \\
trainer.resume_mode=resume_path \\
trainer.resume_from_path=${RESUME_CKPT}"
  echo "[launch] resuming from ${RESUME_CKPT} -> ${TOTAL_STEPS} steps"
fi

# NCCL_SOCKET_IFNAME (the interface NCCL uses between nodes) is passed on only when it is set in the launching shell
PAYLOAD=$(NCCL_IF="${NCCL_SOCKET_IFNAME:-}" python3 -c "
import json, os, sys
env_vars = {
    'PYTHONPATH': '${CODE_ROOT}',
    'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONNOUSERSITE': '1',
    'PYTHONUNBUFFERED': '1',
    'HOME': '${JOB_HOME}',
    'FLASHINFER_WORKSPACE_BASE': '${JOB_HOME}',
    'TORCHINDUCTOR_CACHE_DIR': '${JOB_CACHE_ROOT}/inductor',
    'VLLM_CACHE_ROOT': '${JOB_CACHE_ROOT}/vllm',
    'TRITON_CACHE_DIR': '${JOB_CACHE_ROOT}/triton',
    'HF_HOME': '${JOB_CACHE_ROOT}/hf',
    'XDG_CACHE_HOME': '${JOB_CACHE_ROOT}/xdg',
    'XDG_CONFIG_HOME': '${JOB_CACHE_ROOT}/xdgc',
    'VLLM_ATTENTION_BACKEND': 'XFORMERS',
    'VLLM_NO_USAGE_STATS': '1',
    'PYTORCH_ALLOC_CONF': 'expandable_segments:True',
    'OTEL_SDK_DISABLED': 'true',
    'RAY_DISABLE_USAGE_STATS': '1',
    'VERL_FILE_LOGGER_PATH': '$OUT_DIR/metrics/metrics.jsonl',
    'VERL_FILE_LOGGER_ROOT': '$OUT_DIR/metrics',
    'TENSORBOARD_DIR': '$OUT_DIR/tb',
    # outcome reward: DAPO soft overlong penalty
    'CF_OVERLONG_BUFFER_ENABLE': '${CF_OVERLONG_BUFFER_ENABLE:-1}',
    'CF_OVERLONG_BUFFER_LEN': '${CF_OVERLONG_BUFFER_LEN:-4096}',
    'CF_OVERLONG_PENALTY_FACTOR': '${CF_OVERLONG_PENALTY_FACTOR:-1.0}',
    # token weighting: off | entropy_mask | creditfilter | stare | scgrpo
    'CF_ROUTE': '${CF_ROUTE:-off}',
    # 80/20 rule: fraction of tokens kept by entropy; cut within each response or over the batch
    'CF_MASK_TOP': '${CF_MASK_TOP:-0.2}',
    'CF_MASK_SCOPE': '${CF_MASK_SCOPE:-response}',
    # CreditFilter window: widths for correct / wrong responses, floors by the sign of the advantage
    'CF_WIDTH_POS': '${CF_WIDTH_POS:-0.10}',
    'CF_WIDTH_NEG': '${CF_WIDTH_NEG:-0.14}',
    'CF_FLOOR_POS': '${CF_FLOOR_POS:-0.1}',
    'CF_FLOOR_NEG': '${CF_FLOOR_NEG:-0.0}',
    'CF_KERNEL': '${CF_KERNEL:-exp}',                 # exp | hard (weight 1 on the top CF_HARD_TOP of the rank, 0 elsewhere)
    'CF_HARD_TOP': '${CF_HARD_TOP:-0.2}',
    'CF_RANK_KEY': '${CF_RANK_KEY:-entropy}',         # entropy | random | reversed
    'CF_RANK_SCOPE': '${CF_RANK_SCOPE:-response}',    # response | batch
    # top-rank gain: the exp kernel is multiplied by CF_TOP_GAIN on the ranks q <= CF_TOP_GAIN_Q (0 = off)
    'CF_TOP_GAIN_Q': '${CF_TOP_GAIN_Q:-0}',
    'CF_TOP_GAIN': '${CF_TOP_GAIN:-1.0}',
    # class masses: equalise the two classes, or rescale them to given masses (0 = off); tolerance of the mass check
    'CF_MASS_EQUALIZE': '${CF_MASS_EQUALIZE:-0}',
    'CF_MASS_TARGET_POS': '${CF_MASS_TARGET_POS:-0}',
    'CF_MASS_TARGET_NEG': '${CF_MASS_TARGET_NEG:-0}',
    'CF_MASS_GUARD_TOL': '${CF_MASS_GUARD_TOL:-0.03}',
    'CF_MASS_GUARD_MINLEN': '${CF_MASS_GUARD_MINLEN:-0}',
    # difficulty damping exponents
    'CF_DAMPING_POS': '${CF_DAMPING_POS:-1.0}',
    'CF_DAMPING_NEG': '${CF_DAMPING_NEG:-0.0}',
    # fallback band in nats of batch-mean token entropy (required by CF_ROUTE=creditfilter)
    'CF_FALLBACK_H_LO': '${CF_FALLBACK_H_LO:-}',
    'CF_FALLBACK_H_HI': '${CF_FALLBACK_H_HI:-}',
    # tracked widths (CreditFilter-D)
    'CF_WIDTH_MODE': '${CF_WIDTH_MODE:-fixed}',       # fixed | tracked
    'CF_TRACK_C_POS': '${CF_TRACK_C_POS:-1.0}',
    'CF_TRACK_C_NEG': '${CF_TRACK_C_NEG:-1.0}',
    'CF_TRACK_ALPHA': '${CF_TRACK_ALPHA:-0.1}',
    'CF_TRACK_MIN_POS': '${CF_TRACK_MIN_POS:-0.05}',
    'CF_TRACK_MAX_POS': '${CF_TRACK_MAX_POS:-0.30}',
    'CF_TRACK_MIN_NEG': '${CF_TRACK_MIN_NEG:-0.07}',
    'CF_TRACK_MAX_NEG': '${CF_TRACK_MAX_NEG:-0.40}',
    'CF_TRACK_MIN_ROWS': '${CF_TRACK_MIN_ROWS:-64}',
    'CF_TRACK_STATE_PATH': '${CF_TRACK_STATE_PATH:-${OUT_DIR}/track_state.json}',
    # STARE (CF_STARE_H_TARGET is required by CF_ROUTE=stare)
    'CF_STARE_W': '${CF_STARE_W:-1.1}',
    'CF_STARE_M': '${CF_STARE_M:-1.0}',
    'CF_STARE_P': '${CF_STARE_P:-0.10}',
    'CF_STARE_H_TARGET': '${CF_STARE_H_TARGET:-}',
    # SC-GRPO
    'CF_SCGRPO_PCT': '${CF_SCGRPO_PCT:-75}',
    'CF_SCGRPO_CMIN': '${CF_SCGRPO_CMIN:-1e-4}',
    'CF_SCGRPO_NCMIN': '${CF_SCGRPO_NCMIN:-2}',
    'CF_SCGRPO_DEMO_MAX_TOK': '${CF_SCGRPO_DEMO_MAX_TOK:-8192}',
    'CF_SCGRPO_KL_MAX_RESP': '${CF_SCGRPO_KL_MAX_RESP:-12288}',
}
if os.environ.get('NCCL_IF'):
    env_vars['NCCL_SOCKET_IFNAME'] = os.environ['NCCL_IF']
print(json.dumps({'entrypoint': 'bash -c \"' + sys.argv[1].replace('\"', '\\\\\"') + '\"', 'runtime_env': {'env_vars': env_vars}}))" "$CMD")
RUNTIME_ENV_JSON=$(echo "$PAYLOAD" | python3 -c "import json, sys; print(json.dumps(json.loads(sys.stdin.read())['runtime_env']))")

# every exported CF_* knob must reach the job with its value
for _v in $(env | grep -oE '^CF_[A-Z_0-9]+=' | sed 's/=$//'); do
  _val=$(printenv "$_v")
  echo "$RUNTIME_ENV_JSON" | grep -qF "\"$_v\": \"$_val\"" || {
    echo "[launch] $_v=$_val is exported but is not a knob of the job environment" >&2; exit 1; }
done

echo "$RUNTIME_ENV_JSON" | python3 -m json.tool > "$OUT_DIR_LOCAL/runtime_env.json"
echo "[launch] run directory $OUT_DIR, route ${CF_ROUTE:-off}, data seed ${DATA_SEED:-none}"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "$PAYLOAD" > "$OUT_DIR_LOCAL/launch_payload.json"
  echo "[launch] DRY_RUN=1: nothing submitted; see $OUT_DIR_LOCAL/runtime_env.json and launch_payload.json"
  exit 0
fi
SID="${RUN_TAG}_$(date +%H%M%S)"
python3 -m ray.scripts.scripts job submit --address http://${RAY_HEAD} --no-wait --submission-id "$SID" --runtime-env-json "$RUNTIME_ENV_JSON" -- bash -c "$CMD" >&2 || {
  echo "[launch] ray job submit failed" >&2; exit 1; }
echo "[launch] job id $SID"
echo "$SID" > "$OUT_DIR_LOCAL/submission_id.txt"
