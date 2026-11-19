#!/bin/bash
# The four-branch instrument on one model, on a Ray cluster of GPU nodes, inside the training environment:
#   rollouts -> per-token entropy and grade -> selection of responses and positions -> branch continuations -> tables
# A stage is skipped when its output exists, and the measurement resumes per response.
#
#   MODEL=/path/to/Qwen3-8B-Base TAG=base bash measurement/four_branch/run.sh
#
# The defaults are the settings of the paper's measurement of the base model; its model table uses NRESP=120.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=${TAG:?TAG required (name of the run)}; MODEL=${MODEL:?MODEL required (model directory)}
OUT=${OUT:-$HERE/runs/$TAG}; mkdir -p "$OUT"
NQ=${NQ:-288}; N=${N:-8}; NRESP=${NRESP:-640}; J=${J:-8}
CORPUS=$OUT/corpus.jsonl; STATS=$OUT/entropy.jsonl; SEL=$OUT/selection.jsonl; REC=$OUT/records.jsonl
ts(){ date '+%F %T'; }
echo "[$(ts)] settings TAG=$TAG MODEL=$MODEL NQ=$NQ N=$N NRESP=$NRESP J=$J"
if [ ! -f "$CORPUS" ] || [ "$(wc -l < "$CORPUS")" -lt $((NQ * N - 200)) ]; then
  MODEL=$MODEL OUT=$CORPUS NQ=$NQ N=$N python3 "$HERE/rollouts.py" || { echo "[$(ts)] rollouts FAILED"; exit 2; }
fi
echo "[$(ts)] corpus rows $(wc -l < "$CORPUS")"
if [ ! -f "$STATS" ] || [ "$(wc -l < "$STATS")" -lt 100 ]; then
  python3 "$HERE/entropy.py" "$CORPUS" "$MODEL" "$STATS" || { echo "[$(ts)] entropy FAILED"; exit 3; }
fi
echo "[$(ts)] entropy rows $(wc -l < "$STATS")"
[ -f "$SEL" ] || python3 "$HERE/measure.py" select "$STATS" "$CORPUS" "$SEL" "$NRESP" || { echo "[$(ts)] selection FAILED"; exit 4; }
python3 "$HERE/measure.py" run "$SEL" "$REC" "$MODEL" "$J" || { echo "[$(ts)] measurement FAILED"; exit 5; }
echo "[$(ts)] position records $(wc -l < "$REC")"
python3 "$HERE/analyze.py" base "$REC" "$REC" --json "$OUT/tables.json" | tee "$OUT/tables.txt" || { echo "[$(ts)] analysis FAILED"; exit 6; }
echo "[$(ts)] done TAG=$TAG"
