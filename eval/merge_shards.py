#!/usr/bin/env python3
"""Assembles an evaluation that was split over several nodes with eval32.py --shard I/N.

  python3 eval/merge_shards.py --out results/<name>/<tag> --shards results/<name>_shard0/<tag> results/<name>_shard1/<tag> ...

Checks that the shards are disjoint and cover every problem with k samples, then writes samples.jsonl, summary.json and
runcard.json as a single-node run does."""
import argparse, json, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import eval32

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--shards", nargs="+", required=True)
ap.add_argument("--bench-root", default=os.path.join(eval32.PKG_ROOT, "data", "eval"))
a = ap.parse_args()
metas = [json.load(open(os.path.join(d, "shard.json"))) for d in a.shards]
n = {m["shard"].split("/")[1] for m in metas}; assert len(n) == 1, f"shards from different splits: {n}"
N = int(n.pop()); idx = sorted(int(m["shard"].split("/")[0]) for m in metas); assert idx == list(range(N)), f"shards present {idx}, expected 0..{N-1}"
for key in ("tag", "hf_dir", "k", "temp", "top_p", "max_new", "seed", "sets"):
    vals = {json.dumps(m[key]) for m in metas}; assert len(vals) == 1, f"shards disagree on {key}: {vals}"
m0 = metas[0]; args = types.SimpleNamespace(k=m0["k"], temp=m0["temp"], top_p=m0["top_p"], max_new=m0["max_new"], seed=m0["seed"], sets=m0["sets"])
tasks = eval32.load_tasks(a.bench_root, args.sets, int(m0.get("limit", 0))); want = {(t["set"], t["qidx"]) for t in tasks}
rows = []
for d in a.shards:
    for line in open(os.path.join(d, "samples.jsonl")):
        rows.append(json.loads(line))
seen = {}
for r in rows:
    seen.setdefault((r["set"], r["qidx"]), []).append(r["s"])
missing = want - set(seen); extra = set(seen) - want
assert not missing and not extra, f"coverage: missing {sorted(missing)[:5]}... ({len(missing)}), extra {sorted(extra)[:5]}... ({len(extra)})"
bad = {q: len(v) for q, v in seen.items() if len(v) != args.k or sorted(v) != list(range(args.k))}
assert not bad, f"sample counts != k for {len(bad)} problems, e.g. {list(bad.items())[:5]}"
order = {s: i for i, s in enumerate(args.sets)}; rows.sort(key=lambda r: (order[r["set"]], r["qidx"], r["s"]))
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "samples.jsonl"), "w") as fo:
    for r in rows: fo.write(json.dumps(r) + "\n")
summ = eval32.summarize(m0["tag"], m0["hf_dir"], rows, args); summ["shards"] = [m["shard"] for m in metas]
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
rc = {"merged_from": a.shards, "shard_meta": metas}
for d in a.shards:
    p = os.path.join(os.path.dirname(d), "runcard.json")
    if os.path.exists(p): rc.setdefault("runcards", []).append(json.load(open(p)))
json.dump(rc, open(os.path.join(a.out, "runcard.json"), "w"), indent=1)
line = " | ".join(f"{s} {summ[s]['acc']:.4f} [{summ[s]['ci95'][0]:.3f},{summ[s]['ci95'][1]:.3f}] n={summ[s]['n_q']}" for s in args.sets)
print(f"[eval32] merged {len(rows)} samples from {N} shards into {a.out}", flush=True)
print(f"[VERDICT] {m0['tag']}: mean@{args.k}  {line}", flush=True)
