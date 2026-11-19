#!/bin/bash
# The full-support instrument on one model, on a Ray cluster of GPU nodes, inside the training environment:
#   rollouts -> per-token profile -> selection of responses and positions -> branch law -> continuations -> records -> tables
# A stage is skipped when its output exists, and the continuations resume per position.
#
#   MODEL=/path/to/Qwen3-8B-Base TAG=base bash measurement/full_support/run.sh
#
# The defaults are the settings of the paper's run.  SMOKE=1: 24 prompts x 8 rollouts, 6 responses per class, 4 continuations.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=${TAG:?TAG required (name of the run)}; MODEL=${MODEL:?MODEL required (model directory)}
OUT=${OUT:-$HERE/runs/$TAG}; mkdir -p "$OUT"
NQ=${NQ:-512}; N=${N:-16}; MAXNEW=${MAXNEW:-5800}; GEN_MAXLEN=${GEN_MAXLEN:-8192}; GSEED=${GSEED:-13}
NPER=${NPER:-400}; MAXPERQ=${MAXPERQ:-2}; NQS=${NQS:-1}; MINLEN=${MINLEN:-200}; SEED=${SEED:-13}
COVER=${COVER:-0.9}; KMIN=${KMIN:-2}; KMAX=${KMAX:-6}; RMIN=${RMIN:-0.002}
M=${M:-48}; MAXLEN=${MAXLEN:-6144}; GPU_UTIL=${GPU_UTIL:-0.9}; MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}; BATCH=${BATCH:-8}; NGRADE=${NGRADE:-48}
if [ "${SMOKE:-0}" = 1 ]; then NQ=24; N=8; NPER=6; M=4; BATCH=3; fi
CORPUS=$OUT/corpus.jsonl; LIGHT=$OUT/light.jsonl; TASKS=$OUT/tasks.jsonl; DESIGN=$OUT/design.json; LAW=$OUT/law.jsonl; CONT=$OUT/continuations.jsonl
ts(){ date '+%F %T'; }
echo "[$(ts)] settings TAG=$TAG MODEL=$MODEL NQ=$NQ N=$N MAXNEW=$MAXNEW GEN_MAXLEN=$GEN_MAXLEN GSEED=$GSEED NPER=$NPER MAXPERQ=$MAXPERQ NQS=$NQS MINLEN=$MINLEN SEED=$SEED COVER=$COVER KMIN=$KMIN KMAX=$KMAX RMIN=$RMIN M=$M MAXLEN=$MAXLEN GPU_UTIL=$GPU_UTIL MAX_NUM_SEQS=$MAX_NUM_SEQS BATCH=$BATCH NGRADE=$NGRADE SMOKE=${SMOKE:-0}"
if [ ! -f "$CORPUS" ] || [ "$(wc -l < "$CORPUS")" -lt $((NQ * N - 8)) ]; then
  MODEL=$MODEL OUT=$CORPUS NQ=$NQ N=$N MAXNEW=$MAXNEW GEN_MAXLEN=$GEN_MAXLEN SEED=$GSEED NW=${NW:-16} python3 "$HERE/rollouts.py" || { echo "[$(ts)] rollouts FAILED"; exit 2; }
fi
echo "[$(ts)] corpus rows $(wc -l < "$CORPUS")"
if [ ! -f "$LIGHT" ] || [ "$(wc -l < "$LIGHT")" -lt "$(wc -l < "$CORPUS")" ]; then
  python3 "$HERE/token_profile.py" light "$CORPUS" "$MODEL" "$LIGHT" || { echo "[$(ts)] profile FAILED"; exit 3; }
fi
echo "[$(ts)] profile rows $(wc -l < "$LIGHT")"
[ -f "$TASKS" ] || python3 "$HERE/select_positions.py" "$LIGHT" "$TASKS" "$DESIGN" "$NPER" "$MAXPERQ" "$NQS" "$MINLEN" "$SEED" || { echo "[$(ts)] selection FAILED"; exit 4; }
echo "[$(ts)] selected responses $(wc -l < "$TASKS")"
if [ ! -f "$LAW" ] || [ "$(wc -l < "$LAW")" -lt "$(wc -l < "$TASKS")" ]; then
  python3 "$HERE/token_profile.py" law "$TASKS" "$MODEL" "$LAW" "$COVER" "$KMIN" "$KMAX" "$RMIN" || { echo "[$(ts)] branch law FAILED"; exit 5; }
fi
echo "[$(ts)] branch law rows $(wc -l < "$LAW")"
python3 "$HERE/continuations.py" "$LAW" "$TASKS" "$CONT" "$MODEL" "$M" "$MAXNEW" "$MAXLEN" "$GPU_UTIL" "$MAX_NUM_SEQS" "$BATCH" "$NGRADE" || { echo "[$(ts)] continuations FAILED"; exit 6; }
echo "[$(ts)] position records $(wc -l < "$CONT")"
python3 "$HERE/export_records.py" "$CONT" "$TASKS" "$LIGHT" "$DESIGN" "$OUT/records" || { echo "[$(ts)] export FAILED"; exit 7; }
python3 "$HERE/analyze.py" "$OUT/records/positions.jsonl.gz" "$OUT/records/responses.jsonl.gz" "$OUT/records/design.json" --json "$OUT/tables.json" | tee "$OUT/tables.txt" || { echo "[$(ts)] analysis FAILED"; exit 8; }
echo "[$(ts)] done TAG=$TAG"
