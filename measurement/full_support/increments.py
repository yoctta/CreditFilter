#!/usr/bin/env python3
"""Two analyses of the full-support records at the level of single positions.

  python3 increments.py <positions.jsonl.gz> <responses.jsonl.gz> <design.json> [--json out.json]

assumption    The test of the look-ahead assumption.  At a position whose sampled token is one of the head branches, the realized
              increment is Q(sampled token) - V, with V the success probability at the position.  Under the assumption its mean,
              given the prefix and the outcome, is nu / V in correct and -nu / (1 - V) in wrong responses.  Within each response
              class the realized increment is regressed on indicators of rank bands, with the token entropy as a control; least
              squares is linear in the dependent variable, so each coefficient is the sum of the coefficient of the implied
              increment and of a remainder.  Also: the share of the rank profile in the top 2.5%, implied and realized.
value factor  The forking spectrum over the rank strata against the value-weighted one, in which the forking value of a position
              is divided by V (correct) or 1 - V (wrong): their squared cosine, strata weighted by their width.

Denominators below 0.05 are set to 0.05.  Intervals are 95% percentile intervals of a bootstrap over prompts (2,000 replicates)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import read  # noqa: E402
from estimator import record_nu  # noqa: E402

DEN_MIN = 0.05
TOP = 0.025
RANK_BANDS = [("top 0.25%", 0.0, 0.0025), ("0.25-2.5%", 0.0025, 0.025), ("2.5-10%", 0.025, 0.1), ("10-30%", 0.1, 0.3)]   # against the remaining 70%


def head_law(r):
    """Probabilities, successes and continuations of the head branches, their tokens, the tail mass and the two tail draws; without
    tail draws the head-normalised law."""
    head = [b for b in r["br"] if b["k"] == "h"]
    tail = [b for b in r["br"] if b["k"] == "t"]
    p = np.array([b["p"] for b in head], float)
    S = np.array([b["S"] for b in head], float)
    m = np.array([b["m"] for b in head], float)
    tok = [b["t"] for b in head]
    if len(tail) == 2 and r["r"] > 0:
        return p, S, m, tok, float(r["r"]), [(tail[0]["S"], tail[0]["m"]), (tail[1]["S"], tail[1]["m"])]
    return p / p.sum(), S, m, tok, 0.0, None


def increment(p, S, m, rmass, tails, j):
    """Realized increment of head branch j, and numerator and denominator of the implied increment for a correct (1) and a wrong
    (0) response.  Squares use S (S - 1) / (m (m - 1)), so the numerators are unbiased."""
    Q = S / m
    U = S * (S - 1) / (m * (m - 1))
    qt = 0.0 if tails is None else 0.5 * (tails[0][0] / tails[0][1] + tails[1][0] / tails[1][1])
    PH = float(np.dot(p, Q))
    V = PH + rmass * qt
    cover = float(p.sum())
    PH2 = float(np.dot(p * p, U) + PH * PH - np.dot(p * p, Q * Q))
    num1 = float(np.dot(p, U)) - (PH2 + rmass * qt * PH)
    return Q[j] - V, num1, PH, (PH - V * cover) - num1, cover - PH


def band_of(q):
    for k, (_, lo, hi) in enumerate(RANK_BANDS):
        if lo < q <= hi or (k == 0 and q <= hi):
            return k
    return len(RANK_BANDS)


def fit(X, y, ci, NC, W, ncoef):
    """Least-squares coefficients 1 .. ncoef of y on X, and the same under the prompt counts of every bootstrap replicate."""
    k = X.shape[1]
    XtX = np.zeros((NC, k, k))
    for a in range(k):
        for b in range(k):
            XtX[:, a, b] = np.bincount(ci, weights=X[:, a] * X[:, b], minlength=NC)
    Xty = np.column_stack([np.bincount(ci, weights=X[:, a] * y, minlength=NC) for a in range(k)])
    beta = np.linalg.solve(XtX.sum(0), Xty.sum(0))[1:1 + ncoef]
    boot = np.linalg.solve(np.einsum("bc,ckl->bkl", W, XtX), (W @ Xty)[..., None])[..., 0][:, 1:1 + ncoef]
    return beta, boot


def interval(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def assumption_test(positions, edges, B=2000, seed=8):
    rows = []
    for r in positions:
        p, S, m, tok, rmass, tails = head_law(r)
        if r["x"] not in tok:
            continue
        delta, n1, d1, n0, d0 = increment(p, S, m, rmass, tails, tok.index(r["x"]))
        num, den = (n1, d1) if r["acc"] == 1 else (n0, d0)
        rows.append(dict(qid=r["qid"], cls=r["acc"], q=float(r["q"]), s=int(r["s"]), H=r["H"], delta=delta, implied=num / max(den, DEN_MIN)))
    prompts = sorted({w["qid"] for w in rows})
    index = {q: k for k, q in enumerate(prompts)}
    NC = len(prompts)
    W = np.random.default_rng(seed).multinomial(NC, np.full(NC, 1.0 / NC), size=B).astype(float)
    width = np.diff(np.minimum(np.asarray(edges, float), 1.0))
    out = dict(positions=len(positions), usable=len(rows), prompts=NC)
    for cls, name in ((1, "correct"), (0, "wrong")):
        rc = [w for w in rows if w["cls"] == cls]
        q, H, s = np.array([w["q"] for w in rc]), np.array([w["H"] for w in rc]), np.array([w["s"] for w in rc])
        d, m = np.array([w["delta"] for w in rc]), np.array([w["implied"] for w in rc])
        ci = np.array([index[w["qid"]] for w in rc])
        one = np.ones_like(q)
        band = np.array([band_of(x) for x in q])
        designs = [(np.column_stack([one, (q <= TOP).astype(float), H]), ["top 2.5%"]),
                   (np.column_stack([one] + [(band == k).astype(float) for k in range(len(RANK_BANDS))] + [H]), [n for n, _, _ in RANK_BANDS])]
        res = dict(n=len(rc), bands={})
        for X, names in designs:
            parts = {key: fit(X, y, ci, NC, W, len(names)) for key, y in (("coefficient", d), ("implied", m), ("remainder", d - m))}
            for k, n in enumerate(names):
                res["bands"][n] = dict(n=int(X[:, 1 + k].sum()), **{key: [float(beta[k])] + interval(boot[:, k]) for key, (beta, boot) in parts.items()})
        # share of the rank profile in the top 2.5%: stratum means weighted by the stratum widths
        NS = len(width)
        cnt = np.zeros((NC, NS))
        np.add.at(cnt, (ci, s), 1.0)
        for key, y in (("implied", m), ("realized", d)):
            sm = np.zeros((NC, NS))
            np.add.at(sm, (ci, s), y)

            def share(mean):
                return (mean[..., :5] * width[:5]).sum(-1) / (mean * width).sum(-1)

            res[f"top_share_{key}"] = [float(share(sm.sum(0) / cnt.sum(0)))] + interval(share((W @ sm) / np.maximum(W @ cnt, 1e-9)))
        out[name] = res
    return out


def success_probability(r):
    """V at the position: the head branches exactly and the tail mass through the mean of its two draws; without tail draws the
    head-normalised mean."""
    head = [(b["p"], b["S"] / b["m"]) for b in r["br"] if b["k"] == "h"]
    tail = [b["S"] / b["m"] for b in r["br"] if b["k"] == "t"]
    vh = sum(p * q for p, q in head)
    if len(tail) == 2 and r["r"] > 0:
        return vh + r["r"] * 0.5 * (tail[0] + tail[1])
    return vh / sum(p for p, _ in head)


def value_factor(positions, responses, edges, B=2000, seed=11):
    prompts = sorted({r["qid"] for r in positions})
    index = {q: k for k, q in enumerate(prompts)}
    cls = np.array([r["acc"] for r in positions])
    s = np.array([r["s"] for r in positions])
    clu = np.array([index[r["qid"]] for r in positions])
    weight = np.array([responses[r["uid"]]["n_eligible"] / responses[r["uid"]]["n_selected"] / r["pi"] for r in positions])
    nu = np.array([record_nu(r)[0] for r in positions])
    V = np.array([success_probability(r) for r in positions])
    val = nu / np.maximum(np.where(cls == 1, V, 1.0 - V), DEN_MIN)
    width = np.diff(np.minimum(np.asarray(edges, float), 1.0))
    NS = len(width)

    def cosines(f):
        out = {}
        for c in (1, 0):
            mk = cls == c
            ww = weight[mk] * f[mk]
            tot = np.bincount(s[mk], weights=ww, minlength=NS)
            a, b = (np.clip(np.bincount(s[mk], weights=ww * v[mk], minlength=NS) / np.where(tot > 0, tot, np.nan), 0, None) for v in (nu, val))
            den = (width * a * a).sum() * (width * b * b).sum()
            out[c] = float((width * a * b).sum() ** 2 / den) if den > 0 else float("nan")
        return out

    point = cosines(np.ones_like(weight))
    rng = np.random.default_rng(seed)
    boots = [cosines(np.bincount(rng.integers(0, len(prompts), len(prompts)), minlength=len(prompts)).astype(float)[clu]) for _ in range(B)]
    res = {}
    for c, name in ((1, "correct"), (0, "wrong")):
        v = np.array([b[c] for b in boots])
        res[name] = [point[c]] + interval(v[np.isfinite(v)])
    return res


def main():
    args = sys.argv[1:]
    out_json = None
    if "--json" in args:
        out_json = args[args.index("--json") + 1]
        del args[args.index("--json"):args.index("--json") + 2]
    if len(args) != 3:
        sys.exit(__doc__)
    positions, responses, design = read(args[0]), {r["uid"]: r for r in read(args[1])}, json.load(open(args[2]))
    res = dict(assumption=assumption_test(positions, design["q_edges"]), value_factor=value_factor(positions, responses, design["q_edges"]))
    a = res["assumption"]
    print(f"== look-ahead assumption: {a['usable']} of {a['positions']} positions have their sampled token among the head branches; {a['prompts']} prompts")

    def f(t):
        return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"

    for name in ("correct", "wrong"):
        print(f"  {name} responses ({a[name]['n']} positions)        coefficient                   implied part                  remainder")
        for band, r in a[name]["bands"].items():
            print(f"    {band:10s} n={r['n']:5d}   {f(r['coefficient'])}   {f(r['implied'])}   {f(r['remainder'])}")
        i, d = a[name]["top_share_implied"], a[name]["top_share_realized"]
        print(f"    share of the rank profile in the top 2.5%: implied {100 * i[0]:.0f}% [{100 * i[1]:.0f}, {100 * i[2]:.0f}], realized {100 * d[0]:.0f}% [{100 * d[1]:.0f}, {100 * d[2]:.0f}]")
    print("== value factor: squared cosine of the forking spectrum and the value-weighted one")
    for name, v in res["value_factor"].items():
        print(f"  {name}: {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
