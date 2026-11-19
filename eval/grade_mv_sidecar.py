#!/usr/bin/env python3
"""The grading of the paper's tables: last \\boxed{} -> DAPO normalisation (the `pred` field that eval32.py stores) -> Math-Verify
equivalence with the reference answer.

  python3 eval/grade_mv_sidecar.py [--force] [--bench-root data/eval] <samples.jsonl or glob> ...

For every samples.jsonl it writes a sidecar samples.mv.jsonl next to it with one record per sample in the same order:
{"set", "qidx", "s", "acc_mv"}. Unboxed and empty predictions score 0. The original files are not modified. CPU only; needs
`pip install math-verify`."""
import glob
import json
import os
import sys

import pandas as pd
from math_verify import parse, verify

SETS = ("MATH500", "AIME2024", "AIME2025")
argv = sys.argv[1:]
force = "--force" in argv
bench_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval")
if "--bench-root" in argv:
    bench_root = argv[argv.index("--bench-root") + 1]
    del argv[argv.index("--bench-root"):argv.index("--bench-root") + 2]
pats = [a for a in argv if a != "--force"]
if not pats:
    sys.exit(__doc__)

gold = {}
for s in SETS:
    df = pd.read_parquet(f"{bench_root}/{s}/test.parquet")
    for i, r in df.iterrows():
        idx = r["extra_info"]["index"] if isinstance(r["extra_info"], dict) and "index" in r["extra_info"] else i
        gold[(s, int(idx))] = parse("$" + r["reward_model"]["ground_truth"] + "$")
cache = {}


def mv(s, q, pred):
    key = (s, q, pred)
    if key in cache:
        return cache[key]
    try:
        p = parse("$" + pred + "$") if pred else []
        v = int(bool(verify(gold[(s, q)], p, timeout_seconds=3))) if (gold[(s, q)] and p) else 0
    except Exception:
        v = 0
    cache[key] = v
    return v


done = 0
for f in sorted(set(x for p in pats for x in glob.glob(p))):
    out = os.path.join(os.path.dirname(os.path.abspath(f)), "samples.mv.jsonl")
    if os.path.exists(out) and not force and os.path.getmtime(out) > os.path.getmtime(f):
        continue
    rows = [json.loads(line) for line in open(f)]
    if not rows or not any(r.get("pred") for r in rows):
        print("skip (no pred):", f)
        continue
    with open(out, "w") as g:
        for r in rows:
            g.write(json.dumps({"set": r["set"], "qidx": r["qidx"], "s": r["s"], "acc_mv": mv(r["set"], r["qidx"], r.get("pred", ""))}) + "\n")
    m = [json.loads(line) for line in open(out)]
    acc = {s: sum(x["acc_mv"] for x in m if x["set"] == s) / max(1, sum(1 for x in m if x["set"] == s)) for s in SETS}
    print(f"{'/'.join(f.split('/')[-3:-1]):50s} mv {acc['MATH500']:.3f} / {acc['AIME2024']:.3f} / {acc['AIME2025']:.3f}", flush=True)
    done += 1
print("sidecars written:", done)
