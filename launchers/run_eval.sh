#!/bin/bash
# Evaluates the checkpoints of a plan with the protocol of the paper's tables, as a Ray job (RAY_HEAD set) or on this node.
#
#   python3 eval/make_plan.py --out plan.json --base /path/to/Qwen3-8B-Base --runs-root runs --steps 350 375 400
#   [RAY_HEAD=<head_ip:8265>] bash launchers/run_eval.sh plan.json [<results name>]
#   python3 eval/grade_mv_sidecar.py 'results/<name>/*/samples.jsonl'
#
# Optional: NGPU (default all visible), K (32), SEED (0), EVAL_EXTRA (extra flags of eval32.py), ENTRY_GPUS (GPUs reserved
# for the Ray job). Output: results/<name>/{runcard.json, <tag>/samples.jsonl, <tag>/summary.json, driver.log}.
set -u
PKG_ROOT=${PKG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}; PKG_ROOT=$(cd "$PKG_ROOT" && pwd -P); L=$PKG_ROOT/launchers
PLAN=${1:?plan.json required}; PLAN=$(cd "$(dirname "$PLAN")" && pwd -P)/$(basename "$PLAN")
NAME=${2:-eval_$(date +%Y%m%d_%H%M%S)}; OUT=$PKG_ROOT/results/$NAME
python3 -c "import json,sys; p=json.load(open('$PLAN')); assert isinstance(p,list) and p and all('tag' in e and 'hf' in e for e in p), 'plan must be a non-empty list of {tag,hf[,actor]}'; print('[run_eval] plan:', ', '.join(e['tag'] for e in p))" || exit 5
mkdir -p "$OUT"
ARGS="--plan $PLAN --out $OUT --k ${K:-32} --seed ${SEED:-0} --ngpu ${NGPU:-0} ${EVAL_EXTRA:-}"
ENTRY="cd $PKG_ROOT && python3 eval/eval32.py $ARGS"
echo "[run_eval] $NAME: $ENTRY"
ENV_JSON='{"PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false","VLLM_WORKER_MULTIPROC_METHOD":"spawn","TRITON_CACHE_DIR":"/tmp/triton_'$USER'","TORCHINDUCTOR_CACHE_DIR":"/tmp/inductor_'$USER'","VLLM_CACHE_ROOT":"/tmp/vllm_'$USER'","HF_HOME":"/tmp/hf_'$USER'"}'
if [ -n "${RAY_HEAD:-}" ]; then
  SID="eval_${NAME}_$(date +%H%M%S)"
  python3 "$L/rayjob.py" submit "$RAY_HEAD" "$SID" "$ENTRY" "$ENV_JSON" "${ENTRY_GPUS:-0}" || { echo "[run_eval] submit failed (is the Ray head reachable?)"; exit 6; }
  echo "$SID" > "$OUT/submission_id.txt"; echo "[run_eval] submitted $SID to http://$RAY_HEAD; waiting (hours for a full plan) ..."
  ST=$(python3 "$L/rayjob.py" wait "$RAY_HEAD" "$SID" ${WAIT_S:-172800})
  python3 "$L/rayjob.py" logs "$RAY_HEAD" "$SID" "$OUT/driver.log" 2>/dev/null
  echo "[run_eval] job $SID: $ST"
else
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
  bash -c "$ENTRY" 2>&1 | tee "$OUT/driver.log"; ST=$([ ${PIPESTATUS[0]} -eq 0 ] && echo SUCCEEDED || echo FAILED)
fi
grep -E '^\[VERDICT|^\[template\]|^\[eval32\] runcard' "$OUT/driver.log" | cut -c1-300
echo "[run_eval] status=$ST results=$OUT"; [ "$ST" = SUCCEEDED ]
