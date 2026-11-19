"""Compact records of a run, the files that analyze.py reads.

  python3 export_records.py <continuations.jsonl> <tasks.jsonl> <light_profile.jsonl> <design.json> <out dir>

positions.jsonl.gz   one line per measured position: the record of continuations.py, with the lengths of the continuations of a
                     branch summed (ntok), plus x, the token that the response sampled at the position
responses.jsonl.gz   one line per selected response: entropy and fork probability of every token, the indices of the format
                     tokens, and how many rollouts of its prompt and class are eligible (n_eligible) and selected (n_selected)
design.json          the selection design"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimator import FORMAT_TOKENS  # noqa: E402
from select_positions import eligible_positions  # noqa: E402

DESIGN_KEYS = ("n_rows", "n_questions", "n_mixed_questions", "n_eligible", "picked", "picked_correct", "questions_used", "g_thresholds", "g_quant",
               "q_edges", "nqs", "maxperq", "minlen", "seed", "totals")


def gz_write(path, lines):
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as f:
            for line in lines:
                f.write((line + "\n").encode())


def export(continuations, tasks, light, design, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(design))
    T = {}
    for l in open(tasks):
        t = json.loads(l)
        T[t["uid"]] = t
    # a selected response of a prompt and class stands for all eligible rollouts of that prompt and class
    byq = {}
    for l in open(light):
        r = json.loads(l)
        byq.setdefault(r["qid"], []).append((r["acc"], r.get("finish") == "stop" and len(eligible_positions(r)) >= meta["minlen"]))
    nsel = {}
    for t in T.values():
        nsel[(t["qid"], t["acc"])] = nsel.get((t["qid"], t["acc"]), 0) + 1
    responses = []
    for u, t in T.items():
        nel = sum(1 for acc, ok in byq[t["qid"]] if ok and acc == t["acc"])
        fmt = [i for i, tok in enumerate(t["resp_ids"]) if tok in FORMAT_TOKENS]
        responses.append(json.dumps(dict(uid=u, qid=t["qid"], acc=t["acc"], prate=t["prate"], n_eligible=nel, n_selected=nsel[(t["qid"], t["acc"])],
                                         fmt=fmt, ent=t["ent"], C=t["C"]), separators=(",", ":")))
    positions = []
    for l in open(continuations):
        r = json.loads(l)
        d = {k: v for k, v in r.items() if k != "br"}
        d["x"] = T[r["uid"]]["resp_ids"][r["i"]]
        d["br"] = [dict(t=b["t"], p=b["p"], k=b["k"], S=b["S"], m=b["m"], nlen=b["nlen"], nbox=b["nbox"], ntok=sum(b["len"])) for b in r["br"]]
        positions.append(json.dumps(d, separators=(",", ":")))
    gz_write(os.path.join(out_dir, "positions.jsonl.gz"), positions)
    gz_write(os.path.join(out_dir, "responses.jsonl.gz"), responses)
    with open(os.path.join(out_dir, "design.json"), "w") as f:
        json.dump({k: v for k, v in meta.items() if k in DESIGN_KEYS}, f, indent=1)
        f.write("\n")
    print(f"[export] {len(positions)} positions of {len(responses)} responses -> {out_dir}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    export(*sys.argv[1:6])
