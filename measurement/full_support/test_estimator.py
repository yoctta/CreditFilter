"""CPU checks of the full-support instrument:  python3 test_estimator.py   (or pytest)

  1. nu_hat is unbiased for Var_pi(Q) under the full law, its head-restricted value for the head-normalised law, and without
     tail draws its bias is at most the tail mass
  2. the stratified draw of positions: Horvitz-Thompson totals are unbiased for every rank stratum and every sampled entropy
     stratum, and every non-empty rank stratum is sampled
  3. end to end: synthetic laws with a known consequence per stratum -> position records -> export -> analysis recovers it"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import analyse, read  # noqa: E402
from estimator import ENTROPY_QUANTILES, RANK_EDGES, entropy_stratum, nu_hat, rank_stratum  # noqa: E402
from export_records import export  # noqa: E402
from select_positions import eligible_positions, global_thresholds, positions_for  # noqa: E402

rng = np.random.default_rng(1)


def make_law(H, V=8, cover=0.9, kmin=2, kmax=4, rmin=0.002):
    u = rng.dirichlet(np.ones(V))
    tau = float(np.clip(H / 2.0, 0.05, 2.0))
    pi = u ** (1.0 / tau)
    pi /= pi.sum()
    pi = pi[np.argsort(-pi)]
    cum = np.cumsum(pi)
    K = max(kmin, min(kmax, int((cum < cover).sum()) + 1))
    r = max(0.0, 1.0 - float(cum[K - 1]))
    return pi, K, (r if r >= rmin else 0.0)


def draw_tail(pi, K, r):
    pt = pi.copy()
    pt[:K] = 0
    pt /= pt.sum()
    return [int(y) for y in rng.choice(len(pi), 2, replace=True, p=pt)]


def synth_response(uid, L, cls):
    ent = np.clip(rng.exponential(0.8, L), 0.0, 8.0)
    C = 1.0 - np.exp(-ent * rng.uniform(0.6, 1.2, L))
    ids = [1000 + int(x) for x in rng.integers(0, 5000, L)]
    return dict(uid=uid, qid=uid.split("_op")[0], acc=cls, finish="stop", gt="1", prompt_ids=[1] * 20, resp_ids=ids,
                ent=[round(float(x), 4) for x in ent], C=[round(float(x), 4) for x in C])


def test_estimator(R=40000, m=8):
    pi, K, r = make_law(H=2.5)
    Q = rng.uniform(0.1, 0.9, len(pi))
    true = float((pi * Q * Q).sum() - (pi * Q).sum() ** 2)
    w = pi[:K] / pi[:K].sum()
    true_head = float((w * Q[:K] * Q[:K]).sum() - (w * Q[:K]).sum() ** 2)
    est, est_head, est_no_tail = [], [], []
    for _ in range(R):
        head = [(float(pi[k]), int(rng.binomial(m, Q[k])), m) for k in range(K)]
        tail = [(y, int(rng.binomial(m, Q[y])), m) for y in draw_tail(pi, K, r)]
        nu, nuh, _ = nu_hat(head, tail, r)
        est.append(nu)
        est_head.append(nuh)
        est_no_tail.append(nu_hat(head, [], r)[0])
    est, est_head, est_no_tail = map(np.array, (est, est_head, est_no_tail))
    se, se_head = est.std() / np.sqrt(R), est_head.std() / np.sqrt(R)
    print(f"[1] K={K} r={r:.3f}: nu {true:.5f}, estimate {est.mean():.5f} +- {se:.5f}; head law {true_head:.5f}, estimate {est_head.mean():.5f} +- {se_head:.5f}; "
          f"bias without tail draws {est_no_tail.mean() - true:+.5f}")
    assert abs(est.mean() - true) < 4 * se, "the full-law estimator is biased"
    assert abs(est_head.mean() - true_head) < 4 * se_head, "the head-restricted estimator is biased"
    assert abs(est_no_tail.mean() - true) <= r + 4 * est_no_tail.std() / np.sqrt(R), "the bias without tail draws exceeds the tail mass"


def test_selection(nresp=60, nseeds=200):
    rows = [synth_response(f"q{k}_op0", int(rng.integers(300, 1500)), 1) for k in range(nresp)]
    thr = global_thresholds(rows)
    NS, NG = len(RANK_EDGES) - 1, len(ENTROPY_QUANTILES) + 1
    worst = 0.0
    for r in rows[:12]:
        nf = eligible_positions(r)
        order = sorted(nf, key=lambda i: (-r["ent"][i], i))
        L = len(order)
        q = {i: (rk + 0.5) / L for rk, i in enumerate(order)}
        true_s, true_g = np.zeros(NS), np.zeros(NG)
        for i in nf:
            true_s[rank_stratum(q[i])] += r["ent"][i]
            true_g[entropy_stratum(r["ent"][i], thr)] += r["ent"][i]
        est_s, est_g = np.zeros(NS), np.zeros(NG)
        for sd in range(nseeds):
            pos, L2 = positions_for(r, thr, 1, sd)
            assert L2 == L
            for p in pos:
                assert 0 < p["pi"] <= 1.0
                est_s[p["s"]] += p["H"] / p["pi"] / nseeds
                est_g[p["g"]] += p["H"] / p["pi"] / nseeds
        rel_s = np.abs(est_s - true_s)[true_s > 0] / true_s[true_s > 0]
        sampled = true_g[:len(thr)] > 0                 # only the top entropy strata are sampled
        rel_g = np.abs(est_g - true_g)[:len(thr)][sampled] / true_g[:len(thr)][sampled]
        worst = max(worst, rel_s.max(), rel_g.max() if len(rel_g) else 0)
    print(f"[2] Horvitz-Thompson totals over {nseeds} draws: worst relative error {worst:.3f}")
    assert worst < 0.12, "Horvitz-Thompson totals are off"
    r = rows[0]
    pos, L = positions_for(r, thr, 1, 0)
    assert len({p["s"] for p in pos}) == len({rank_stratum((rk + 0.5) / L) for rk in range(L)}), "a non-empty rank stratum was not sampled"


def test_end_to_end(nper=60, m=32):
    NS = len(RANK_EDGES) - 1
    kappa_true = np.array([0.02 if RANK_EDGES[s + 1] <= 0.0101 else 0.008 if RANK_EDGES[s + 1] <= 0.0501 else 0.003 for s in range(NS)])
    rows = [synth_response(f"q{k // 2}_op{k % 2}", int(rng.integers(300, 1400)), 1 if k < nper else 0) for k in range(nper * 2)]
    thr = global_thresholds(rows)
    tmp = tempfile.mkdtemp()
    tasks_p, design_p, cont_p, light_p = [os.path.join(tmp, f) for f in ("tasks.jsonl", "design.json", "continuations.jsonl", "light.jsonl")]
    truth = {1: {}, 0: {}}
    with open(tasks_p, "w") as ft, open(cont_p, "w") as fc:
        for r in rows:
            nf = eligible_positions(r)
            order = sorted(nf, key=lambda i: (-r["ent"][i], i))
            L = len(order)
            qof = {i: (rk + 0.5) / L for rk, i in enumerate(order)}
            laws = {}
            for i in nf:
                s = rank_stratum(qof[i])
                pi, K, rr = make_law(r["ent"][i])
                Cf = float(1.0 - (pi * pi).sum())
                z = rng.choice([-1.0, 1.0], len(pi))
                vz = float((pi * z * z).sum() - (pi * z).sum() ** 2)
                Q = np.clip(0.5 + np.sqrt(kappa_true[s] * Cf / max(vz, 1e-9)) * z, 0.0, 1.0)
                nu_t = float((pi * Q * Q).sum() - (pi * Q).sum() ** 2)
                laws[i] = (pi, K, rr, Q)
                r["C"][i] = round(Cf, 5)
                acc = truth[r["acc"]].setdefault(s, [0.0, 0.0])
                acc[0] += nu_t
                acc[1] += Cf
            pos, _ = positions_for(r, thr, 1, 7)
            ft.write(json.dumps(dict(uid=r["uid"], qid=r["qid"], acc=r["acc"], gt="1", prate=0.5, L=L, prompt_ids=r["prompt_ids"], resp_ids=r["resp_ids"],
                                     ent=r["ent"], C=r["C"], positions=pos)) + "\n")
            for p in pos:
                pi, K, rr, Q = laws[p["i"]]
                br = [dict(t=k, p=float(pi[k]), k="h", S=int(rng.binomial(m, Q[k])), m=m, len=[10] * m, nlen=0, nbox=m) for k in range(K)]
                if rr > 0:
                    for y in draw_tail(pi, K, rr):
                        br.append(dict(t=y, p=float(pi[y] / rr), k="t", S=int(rng.binomial(m, Q[y])), m=m, len=[10] * m, nlen=0, nbox=m))
                fc.write(json.dumps(dict(uid=r["uid"], qid=r["qid"], acc=r["acc"], prate=0.5, L=L, i=p["i"], q=p["q"], H=p["H"], C=r["C"][p["i"]],
                                         s=p["s"], g=p["g"], Ns=p["Ns"], Ng=p["Ng"], pi=p["pi"], r=rr, cover=1 - rr, br=br)) + "\n")
    with open(light_p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(dict(g_thresholds=thr, q_edges=RANK_EDGES, minlen=200), open(design_p, "w"))
    export(cont_p, tasks_p, light_p, design_p, os.path.join(tmp, "records"))
    positions = read(os.path.join(tmp, "records", "positions.jsonl.gz"))
    responses = {r["uid"]: r for r in read(os.path.join(tmp, "records", "responses.jsonl.gz"))}
    ok = tot = 0
    for cls, name in ((1, "correct"), (0, "wrong")):
        res = analyse(name, cls, positions, responses, RANK_EDGES, B=300)
        for s in range(NS):
            row = res["strata"][s]
            if row["n"] < 10:
                continue
            kt = truth[cls][s][0] / truth[cls][s][1]
            lo, hi = row["kappa_ci"]
            tot += 1
            ok += (lo - 0.0005 <= kt <= hi + 0.0005)
            if s in (0, 2, 5, 8, 12):
                print(f"[3] {name} stratum {row['bin']:>14} n={row['n']:>3} consequence {row['kappa'] * 1e3:.2f} [{lo * 1e3:.2f}, {hi * 1e3:.2f}] true {kt * 1e3:.2f}")
    print(f"[3] the interval holds the true consequence in {ok} of {tot} strata")
    assert ok >= 0.85 * tot, "the intervals miss the true consequence too often"


if __name__ == "__main__":
    test_estimator()
    test_selection()
    test_end_to_end()
    print("ALL CPU TESTS PASSED")
