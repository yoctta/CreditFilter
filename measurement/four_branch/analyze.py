#!/usr/bin/env python3
"""Tables of the four-branch instrument, computed from its per-position records.

  python3 analyze.py base   <records of correct responses> <records of wrong responses> [--json out.json]
  python3 analyze.py models <model.jsonl.gz> ... [--replicates 600] [--json out.json]

base    per response class of the base model: the forking spectrum (widths of the fork-probability envelope and of the forking
        value, the pooled fork consequence, its relative slope, the top-band gain), the ten-band table, cover and truncation, and
        the continuation-horizon check
models  per model, classes pooled: the forking-value width, the envelope width and the coarse-group means of the forking value

A record is one measured position: uid (response), acc (1 correct, 0 wrong), i (token index), q (entropy rank of the position within
its response), H (entropy in nats), cover (probability mass of the four branches), w (branch weights, renormalised), g (0/1 grade of
every continuation of every branch), fin (length-truncated continuations per branch), box (continuations with a boxed answer per
branch).  Intervals are 95% percentile intervals of a bootstrap over responses."""
import gzip
import json
import math
import sys

import numpy as np

BANDS = [(0, .025), (.025, .05), (.05, .075), (.075, .1), (.1, .15), (.15, .2), (.2, .3), (.3, .4), (.4, .6), (.6, .8)]
CENTERS = np.array([(a + b) / 2 for a, b in BANDS])
NB = len(BANDS)
GROUPS = {"G1": [0, 1, 2, 3], "G2": [4, 5, 6, 7], "G3": [8, 9]}     # q <= 0.1, 0.1 < q <= 0.4, q > 0.4


def forking_value(w, G):
    """Unbiased estimate of the variance of the branch success probability under the weights w.  G: grades [branches, continuations]."""
    m = G.shape[1]
    S = G.sum(1)
    Q = S / m
    U = S * (S - 1) / (m * (m - 1))
    v = float((w * (1 - w) * U).sum() - (np.outer(w, w) * np.outer(Q, Q))[~np.eye(len(w), dtype=bool)].sum())
    return 0.0 if abs(v) < 1e-12 else v


def read(path, acc=None):
    """Records of a file as (uid, q, H, cover, w, G, fin, box) tuples; acc keeps one response class."""
    rows = []
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        for line in f:
            r = json.loads(line)
            if acc is not None and r["acc"] != acc:
                continue
            w = np.array(r["w"], float)
            rows.append((r["uid"], float(r["q"]), float(r["H"]), float(r["cover"]), w / w.sum(), np.array(r["g"], float),
                         np.array(r["fin"], int), np.array(r["box"], int)))
    return rows


def arrays(rows, truncated="wrong"):
    """Per-position arrays.  truncated: 'wrong' grades a length-truncated continuation 0 (as measured), 'drop' keeps only the
    positions without a truncated continuation, 'right' grades every truncated continuation 1."""
    uid, q, H, nu, C, cover, ntr, nbox = [], [], [], [], [], [], [], []
    for u, qq, hh, cv, w, G, fin, box in rows:
        if truncated == "drop" and fin.sum() > 0:
            continue
        if truncated == "right":
            G = G.copy()
            for k in range(len(w)):
                G[k, np.where(G[k] == 0)[0][:fin[k]]] = 1
        uid.append(u)
        q.append(qq)
        H.append(hh)
        nu.append(forking_value(w, G))
        C.append(float(1 - (w ** 2).sum()))
        cover.append(cv)
        ntr.append(int(fin.sum()))
        nbox.append(int(box.sum()))
    return dict(uid=np.array(uid), q=np.array(q), H=np.array(H), nu=np.array(nu), C=np.array(C), cover=np.array(cover),
                ntr=np.array(ntr), nbox=np.array(nbox))


def band_index(q):
    idx = np.full(q.shape, -1, int)
    for b, (lo, hi) in enumerate(BANDS):
        idx[(q > lo) & (q <= hi)] = b
    return idx


def per_response(uid, q, val):
    """S[r, b] and N[r, b]: sums and counts of val per response and band (responses sorted by uid)."""
    uids = sorted(set(uid.tolist()))
    ui = {u: k for k, u in enumerate(uids)}
    S = np.zeros((len(uids), NB))
    N = np.zeros_like(S)
    bi = band_index(q)
    ok = bi >= 0
    rows = np.array([ui[u] for u in uid[ok]])
    np.add.at(S, (rows, bi[ok]), np.asarray(val, float)[ok])
    np.add.at(N, (rows, bi[ok]), 1.0)
    return S, N


def band_means(S, N, B=400, seed=7):
    """Band means (every response has the same design in every band) with their bootstrap interval."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = S.sum(0) / N.sum(0)
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    V = np.empty((B, S.shape[1]))
    for b in range(B):
        idx = rng.integers(0, n, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            V[b] = S[idx].sum(0) / N[idx].sum(0)
    return pt, np.nanpercentile(V, 2.5, axis=0), np.nanpercentile(V, 97.5, axis=0)


def width(S, N):
    """ln 2 / (-slope) of the least-squares fit of log(band mean) on the band centre over the bands with a positive mean;
    nan with fewer than five such bands or a fit that does not decay."""
    with np.errstate(divide="ignore", invalid="ignore"):
        m = S / N
    ok = (N > 0) & (m > 0)
    if ok.sum() < 5:
        return np.nan
    x, y = CENTERS[ok], np.log(m[ok])
    x0, y0 = x.mean(), y.mean()
    b = ((x - x0) * (y - y0)).sum() / max(((x - x0) ** 2).sum(), 1e-12)
    return math.log(2) / -b if b < 0 else np.nan


def width_interval(S, N, B=300, seed=7):
    """(width, lower, upper); the interval is over the replicates whose fit decays."""
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    pt = width(S.sum(0), N.sum(0))
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = width(S[idx].sum(0), N[idx].sum(0))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return pt, None, None
    vals = np.sort(vals)
    return pt, float(vals[int(.025 * len(vals))]), float(vals[min(len(vals) - 1, int(.975 * len(vals)))])


def log_fit(mean):
    """Log-linear least squares over the bands with a positive mean: (width, R2 in the log domain); the width is inf when the fit
    does not decay."""
    m = np.asarray(mean, float)
    ok = m > 0
    if ok.sum() < 3:
        return np.nan, np.nan
    b, a = np.polyfit(CENTERS[ok], np.log(m[ok]), 1)
    y = np.log(m[ok])
    yh = a + b * CENTERS[ok]
    return (math.log(2) / -b if b < 0 else float("inf")), float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def percentile_interval(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


def consequence(S, SC, N, rng, B=600):
    """Fork consequence kappa = E[nu | q] / E[C | q] per band, pooled over bands 2-10, the top-band gain kappa_1 / pooled and the
    relative slope (count-weighted least-squares slope of kappa over the centres of bands 2-10, divided by the pooled value)."""
    n = S.shape[0]

    def stats(idx):
        kap = (S[idx].sum(0) / N[idx].sum(0)) / (SC[idx].sum(0) / N[idx].sum(0))
        pooled = S[idx][:, 1:].sum() / SC[idx][:, 1:].sum()
        wts = N[idx][:, 1:].sum(0)
        x, y = CENTERS[1:], kap[1:]
        xm, ym = np.average(x, weights=wts), np.average(y, weights=wts)
        slope = (wts * (x - xm) * (y - ym)).sum() / (wts * (x - xm) ** 2).sum()
        return kap, pooled, (kap[0] / pooled if pooled > 0 else np.nan), (slope / pooled if pooled > 0 else np.nan)

    pt = stats(np.arange(n))
    boots = [stats(rng.integers(0, n, n)) for _ in range(B)]
    band_ci = [percentile_interval([b[0][i] for b in boots]) for i in range(NB)]
    return dict(band=pt[0].tolist(), band_lo=[c[0] for c in band_ci], band_hi=[c[1] for c in band_ci],
                pooled=float(pt[1]), pooled_ci=percentile_interval([b[1] for b in boots]),
                gain=float(pt[2]), gain_ci=percentile_interval([b[2] for b in boots]),
                slope=float(pt[3]), slope_ci=percentile_interval([b[3] for b in boots]))


def spectrum(rows, rng):
    a = arrays(rows)
    S, N = per_response(a["uid"], a["q"], a["nu"])
    SC, _ = per_response(a["uid"], a["q"], a["C"])
    SH, _ = per_response(a["uid"], a["q"], a["H"])
    out = dict(responses=int(S.shape[0]), positions=int(N.sum()), n=N.sum(0).astype(int).tolist())
    for key, SS in (("nu", S), ("C", SC), ("H", SH)):
        pt, lo, hi = band_means(SS, N)
        out[key] = dict(mean=pt.tolist(), lo=lo.tolist(), hi=hi.tolist())
    for key, SS in (("nu", S), ("C", SC)):
        out[key]["width"] = list(width_interval(SS, N))
        out[key]["R2"] = log_fit(out[key]["mean"])[1]
    out["kappa"] = consequence(S, SC, N, rng)
    bi = band_index(a["q"])
    K, m = rows[0][5].shape
    out["cover"] = dict(mean=float(a["cover"].mean()), band=[float(a["cover"][bi == b].mean()) for b in range(NB)])
    out["continuations"] = dict(branches=int(K), per_branch=int(m), truncated=float(a["ntr"].sum() / (K * m * len(a["ntr"]))),
                                positions_with_truncation=float((a["ntr"] > 0).mean()), boxed=float(a["nbox"].sum() / (K * m * len(a["nbox"]))))
    return out


def horizon(rows, B=600, seed=11):
    """The spectrum statistics under three treatments of the length-truncated continuations."""
    res = {}
    for name in ("wrong", "drop", "right"):
        a = arrays(rows, truncated=name)
        S, N = per_response(a["uid"], a["q"], a["nu"])
        SC, _ = per_response(a["uid"], a["q"], a["C"])
        n = S.shape[0]

        def one(idx):
            Ns = N[idx].sum(0)
            kap = (S[idx].sum(0) / Ns) / (SC[idx].sum(0) / Ns)
            pooled = S[idx][:, 1:].sum() / SC[idx][:, 1:].sum()
            head = S[idx][:, 1:6].sum() / SC[idx][:, 1:6].sum()         # q <= 0.2 within bands 2-10
            tail = S[idx][:, 6:].sum() / SC[idx][:, 6:].sum()           # q > 0.2
            return dict(width_C=width(SC[idx].sum(0), Ns), width_nu=width(S[idx].sum(0), Ns), pooled=pooled,
                        gain=kap[0] / pooled if pooled > 0 else np.nan, head_to_tail=head / tail if tail > 0 else np.nan)

        rng = np.random.default_rng(seed)
        pt = one(np.arange(n))
        boots = [one(rng.integers(0, n, n)) for _ in range(B)]
        res[name] = dict(responses=int(n), positions=int(N.sum()),
                         **{k: [float(pt[k])] + percentile_interval([b[k] for b in boots]) for k in pt})
    return res


def model_row(rows, B=600, seed=11):
    """One model, classes pooled: widths from the log-linear fit over the positive band means (the interval of the forking-value
    width is over the replicates whose fit decays) and the mean forking value of the coarse groups."""
    a = arrays(rows)
    S, N = per_response(a["uid"], a["q"], a["nu"])
    SC, _ = per_response(a["uid"], a["q"], a["C"])
    n = S.shape[0]

    def stats(idx):
        Nn = N[idx].sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            nu, C = S[idx].sum(0) / Nn, SC[idx].sum(0) / Nn
        return log_fit(nu)[0], log_fit(C)[0], {g: S[idx][:, b].sum() / N[idx][:, b].sum() for g, b in GROUPS.items()}

    pt = stats(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [stats(rng.integers(0, n, n)) for _ in range(B)]
    return dict(responses=int(n), positions=int(N.sum()), width_nu=pt[0], width_nu_ci=percentile_interval([b[0] for b in boots]),
                width_C=pt[1], group_nu={g: float(v) for g, v in pt[2].items()})


def fmt(v, d=2, scale=1.0):
    return "--" if v is None or not np.isfinite(v) else f"{scale * v:.{d}f}"


def interval(v, lo, hi, d=2, scale=1.0):
    return f"{fmt(v, d, scale)} [{fmt(lo, d, scale)}, {fmt(hi, d, scale)}]"


def main():
    args = sys.argv[1:]
    opts = {"--json": None, "--replicates": "600"}
    for name in opts:
        if name in args:
            opts[name] = args[args.index(name) + 1]
            del args[args.index(name):args.index(name) + 2]
    out_json = opts["--json"]
    mode, files = args[0], args[1:]
    res = {}
    if mode == "base":
        classes = {"correct": read(files[0], acc=1), "wrong": read(files[1], acc=0)}
        classes = {c: rows for c, rows in classes.items() if rows}
        rng = np.random.default_rng(11)          # one stream for the consequence statistics of both classes, correct first
        res["spectrum"] = {c: spectrum(rows, rng) for c, rows in classes.items()}
        res["horizon"] = {c: horizon(rows) for c, rows in classes.items()}
        for c, s in res["spectrum"].items():
            k = s["kappa"]
            print(f"== {c}: {s['responses']} responses, {s['positions']} positions")
            print(f"  envelope width        {interval(*s['C']['width'], 3)}   R2 {s['C']['R2']:.2f}")
            print(f"  forking-value width   {interval(*s['nu']['width'], 3)}   R2 {s['nu']['R2']:.2f}")
            print(f"  top-band forking value (1e-3)   {interval(s['nu']['mean'][0], s['nu']['lo'][0], s['nu']['hi'][0], 2, 1e3)}")
            print(f"  consequence, bands 2-10 (1e-3)  {interval(k['pooled'], *k['pooled_ci'], 2, 1e3)}")
            print(f"  relative slope of the consequence   {interval(k['slope'], *k['slope_ci'], 1)}")
            print(f"  top-band gain   {interval(k['gain'], *k['gain_ci'], 1)}")
            print(f"  cover: mean {s['cover']['mean']:.3f}, by band " + " ".join(f"{v:.2f}" for v in s["cover"]["band"]))
            t = s["continuations"]
            print(f"  continuations {t['branches']} x {t['per_branch']}: truncated {100 * t['truncated']:.1f}%, positions with a truncated one "
                  f"{100 * t['positions_with_truncation']:.0f}%, boxed {100 * t['boxed']:.1f}%")
            print("  band          n   forking value (1e-3)   fork probability       entropy             consequence (1e-3)")
            for b, (lo, hi) in enumerate(BANDS):
                print(f"  {lo:g}-{hi:g}".ljust(13) + f"{s['n'][b]:5d}   {interval(s['nu']['mean'][b], s['nu']['lo'][b], s['nu']['hi'][b], 2, 1e3):22s} "
                      f"{interval(s['C']['mean'][b], s['C']['lo'][b], s['C']['hi'][b], 3):22s} {interval(s['H']['mean'][b], s['H']['lo'][b], s['H']['hi'][b], 2):19s} "
                      f"{interval(k['band'][b], k['band_lo'][b], k['band_hi'][b], 2, 1e3)}")
        print("== continuation horizon: truncated continuations graded wrong (as measured) / positions with one dropped / graded right")
        for c, h in res["horizon"].items():
            for name, s in h.items():
                print(f"  {c:7s} {name:5s} positions {s['positions']:5d}  envelope width {interval(*s['width_C'], 3)}  forking-value width {interval(*s['width_nu'], 3)}  "
                      f"consequence (1e-3) {interval(*s['pooled'], 2, 1e3)}  gain {interval(*s['gain'], 1)}  head-to-tail {interval(*s['head_to_tail'], 2)}")
    elif mode == "models":
        print("model".ljust(28) + "responses  forking-value width        envelope width  forking value (1e-3), q <= 0.1 / 0.1-0.4 / > 0.4")
        for path in files:
            name = path.rsplit("/", 1)[-1].replace(".jsonl.gz", "").replace(".jsonl", "")
            r = model_row(read(path), B=int(opts["--replicates"]))
            res[name] = r
            print(f"{name:28s}{r['responses']:9d}  {interval(r['width_nu'], *r['width_nu_ci'], 3):25s}  {r['width_C']:.3f}           "
                  + " / ".join(fmt(r["group_nu"][g], 2, 1e3) for g in GROUPS))
    else:
        sys.exit(__doc__)
    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
