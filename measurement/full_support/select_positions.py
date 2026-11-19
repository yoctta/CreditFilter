"""Selection of responses and positions with known inclusion probabilities.

  python3 select_positions.py <light_profile.jsonl> <tasks_out.jsonl> <design_out.json> NPER MAXPERQ NQS MINLEN SEED

Responses: finished, at least MINLEN ranked tokens, from prompts with mixed outcomes (some but not all rollouts correct; a rollout
cut at the cap counts as wrong, as in training); NPER per class, at most MAXPERQ per class and prompt, prompts taken in turn in a
seeded shuffle.
Positions of a response: NQS uniform draws from every non-empty rank stratum, and one uniform draw from every non-empty stratum
of the token entropy over the whole corpus (the top ENTROPY_QUANTILES).  A position drawn by both designs is measured once; its
inclusion probability is pi = 1 - (1 - nq / N_s)(1 - [entropy stratum sampled] / N_g), and every analysis weights by 1 / pi."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimator import ENTROPY_QUANTILES, FORMAT_TOKENS, RANK_EDGES, TAIL_EXCLUDED, entropy_stratum, rank_stratum, seed_of  # noqa: E402


def eligible_positions(r):
    n = len(r["resp_ids"])
    return [i for i, t in enumerate(r["resp_ids"]) if t not in FORMAT_TOKENS and i + TAIL_EXCLUDED < n]


def positions_for(r, thr, nqs, seed, edges=RANK_EDGES):
    """Stratified draw for one response; returns (positions, number of ranked tokens).  Every eligible position has pi > 0."""
    nf = eligible_positions(r)
    order = sorted(nf, key=lambda i: (-r["ent"][i], i))
    L = len(order)
    qof = {i: (rk + 0.5) / L for rk, i in enumerate(order)}
    sof = {i: rank_stratum(qof[i], edges) for i in nf}
    gof = {i: entropy_stratum(r["ent"][i], thr) for i in nf}
    members_s, members_g = {}, {}
    for i in nf:
        members_s.setdefault(sof[i], []).append(i)
        if gof[i] < len(thr):
            members_g.setdefault(gof[i], []).append(i)
    rg = random.Random(seed_of("sel", seed, r["uid"]))
    chosen = set()
    for s in sorted(members_s):
        chosen.update(rg.sample(members_s[s], min(nqs, len(members_s[s]))))
    for g in sorted(members_g):
        chosen.update(rg.sample(members_g[g], 1))
    pos = []
    for i in sorted(chosen):
        Ns = len(members_s[sof[i]])
        pq = min(nqs, Ns) / Ns
        Ng = len(members_g[gof[i]]) if gof[i] < len(thr) else 0
        pg = (1.0 / Ng) if Ng else 0.0
        pi = 1.0 - (1.0 - pq) * (1.0 - pg)
        pos.append(dict(i=i, q=round(qof[i], 6), H=r["ent"][i], C=r["C"][i], s=sof[i], g=gof[i], Ns=Ns, Ng=Ng, pi=round(pi, 6)))
    return pos, L


def global_thresholds(elig):
    allH = sorted((r["ent"][i] for r in elig for i in eligible_positions(r)), reverse=True)
    return [allH[min(len(allH) - 1, int(q * len(allH)))] for q in ENTROPY_QUANTILES]


def pick_responses(rows, nper, maxperq, minlen, seed):
    byq = {}
    for r in rows:
        byq.setdefault(r["qid"], []).append(r)
    prate = {q: sum(x["acc"] for x in v) / len(v) for q, v in byq.items()}
    elig = [r for r in rows if r.get("finish") == "stop" and len(eligible_positions(r)) >= minlen and 0 < prate[r["qid"]] < 1]
    rng = random.Random(seed)
    picked, used_q = [], {}
    for cls in (1, 0):
        cand = {}
        for r in elig:
            if r["acc"] == cls:
                cand.setdefault(r["qid"], []).append(r)
        qids = sorted(cand)
        rng.shuffle(qids)
        for v in cand.values():
            rng.shuffle(v)
        sel = []
        for rnd in range(maxperq):
            for q in qids:
                if len(sel) >= nper:
                    break
                if len(cand[q]) > rnd:
                    sel.append(cand[q][rnd])
        picked += sel
        used_q[cls] = len({r["qid"] for r in sel})
    return picked, elig, prate, byq, used_q


def main():
    src, out, meta_out = sys.argv[1:4]
    nper, maxperq, nqs, minlen, seed = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
    rows = [json.loads(l) for l in open(src)]
    picked, elig, prate, byq, used_q = pick_responses(rows, nper, maxperq, minlen, seed)
    thr = global_thresholds(elig)
    tot = {"q": [0] * (len(RANK_EDGES) - 1), "g": [0] * (len(ENTROPY_QUANTILES) + 1), "pos": 0, "resp": len(picked)}
    with open(out, "w") as f:
        for r in picked:
            pos, L = positions_for(r, thr, nqs, seed)
            for p in pos:
                tot["q"][p["s"]] += 1
                tot["g"][p["g"]] += 1
            tot["pos"] += len(pos)
            f.write(json.dumps(dict(uid=r["uid"], qid=r["qid"], acc=r["acc"], gt=r["gt"], prate=round(prate[r["qid"]], 4), L=L,
                                    prompt_ids=r["prompt_ids"], resp_ids=r["resp_ids"], ent=r["ent"], C=r["C"], positions=pos)) + "\n")
    meta = dict(n_rows=len(rows), n_questions=len(byq), n_mixed_questions=sum(1 for q in byq if 0 < prate[q] < 1),
                n_eligible=len(elig), picked=len(picked), picked_correct=sum(1 for r in picked if r["acc"] == 1),
                questions_used={"correct": used_q.get(1, 0), "wrong": used_q.get(0, 0)}, g_thresholds=[round(t, 4) for t in thr], g_quant=ENTROPY_QUANTILES,
                q_edges=RANK_EDGES, nqs=nqs, maxperq=maxperq, minlen=minlen, seed=seed, totals=tot)
    json.dump(meta, open(meta_out, "w"), indent=1)
    print(f"[select] rows {len(rows)} prompts {len(byq)} mixed {meta['n_mixed_questions']} eligible {len(elig)} -> picked {len(picked)} "
          f"(correct {meta['picked_correct']}, prompts {used_q}) positions {tot['pos']} per rank stratum {tot['q']} per entropy stratum {tot['g']} thresholds {meta['g_thresholds']}", flush=True)


if __name__ == "__main__":
    main()
