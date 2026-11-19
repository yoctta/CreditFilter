#!/usr/bin/env python3
"""Tables of the full-support instrument, computed from its records.

  python3 analyze.py <positions.jsonl.gz> <responses.jsonl.gz> <design.json> [--json out.json]

Per response class: the 16 rank strata (sampled positions, cover of the head branches, fork probability, forking value, fork
consequence under the full law and under the head-restricted law), the consequence pooled over three rank regions with their
ratios, the ten bands of the four-branch instrument, and the matched-filter gain that weight profiles retain against the
forking-value profile.

Every mean over sampled positions is a Horvitz-Thompson mean with the weight (response weight) / (inclusion probability); the
fork probability is averaged over every ranked token of the selected responses.  A selected response of a prompt and class
stands for all eligible rollouts of that prompt and class (n_eligible / n_selected), the population a training batch sees.
Intervals are 95% percentile intervals of a bootstrap over prompts (2,000 replicates)."""
import gzip
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimator import TAIL_EXCLUDED, rank_stratum, record_nu  # noqa: E402

REGIONS = [("0.5-2.5%", .005, .025), ("2.5-20%", .025, .2), ("20-80%", .2, .8)]
BANDS = [(0, .025), (.025, .05), (.05, .075), (.075, .1), (.1, .15), (.15, .2), (.2, .3), (.3, .4), (.4, .6), (.6, .8)]
WINDOW = {"correct": (0.10, 0.1), "wrong": (0.14, 0.0)}      # width and floor of the CreditFilter window


def read(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        return [json.loads(line) for line in f]


def bins_of(stratum, q, n_strata):
    """Columns of a position: its rank stratum, then the rank regions and the ten bands that contain q."""
    out = [stratum]
    out += [n_strata + k for k, (_, lo, hi) in enumerate(REGIONS) if lo <= q < hi]
    out += [n_strata + len(REGIONS) + k for k, (lo, hi) in enumerate(BANDS) if lo <= q < hi]
    return out


def class_matrices(cls, positions, responses, edges):
    """Per prompt and column: weighted sums of the forking value (full and head-restricted law), of the head fork probability and
    of the weights over the sampled positions; of the fork probability and of the weights over every ranked token."""
    NS = len(edges) - 1
    ncol = NS + len(REGIONS) + len(BANDS)
    R = [r for r in positions if r["acc"] == cls]
    uids = sorted({r["uid"] for r in R})
    qids = sorted({r["qid"] for r in R})
    qi = {q: k for k, q in enumerate(qids)}
    weight = {u: responses[u]["n_eligible"] / responses[u]["n_selected"] for u in uids}
    M = {k: np.zeros((len(qids), ncol)) for k in ("nu", "w", "nu_head", "C_head", "C", "n")}
    for u in uids:
        t = responses[u]
        n = len(t["ent"])
        fmt = set(t["fmt"])
        ranked = [i for i in range(n) if i not in fmt and i + TAIL_EXCLUDED < n]
        order = sorted(ranked, key=lambda i: (-t["ent"][i], i))
        for rk, i in enumerate(order):
            q = (rk + 0.5) / len(order)
            for b in bins_of(rank_stratum(q, edges), q, NS):
                M["C"][qi[t["qid"]], b] += weight[u] * t["C"][i]
                M["n"][qi[t["qid"]], b] += weight[u]
    count = np.zeros(ncol)
    cover = np.zeros(ncol)
    for r in R:
        nu, nu_head, C_head = record_nu(r)
        w = weight[r["uid"]] / r["pi"]
        for b in bins_of(r["s"], r["q"], NS):
            M["nu"][qi[r["qid"]], b] += w * nu
            M["w"][qi[r["qid"]], b] += w
            M["nu_head"][qi[r["qid"]], b] += w * nu_head
            M["C_head"][qi[r["qid"]], b] += w * C_head
            count[b] += 1
            cover[b] += w * r["cover"]
    return M, count, cover / np.maximum(M["w"].sum(0), 1e-300), dict(responses=len(uids), prompts=len(qids), positions=len(R))


def estimates(M, cnt):
    """Columns under prompt counts cnt: forking value, consequence, head-restricted consequence, fork probability."""
    nu = (cnt @ M["nu"]) / np.maximum(cnt @ M["w"], 1e-300)
    C = (cnt @ M["C"]) / np.maximum(cnt @ M["n"], 1e-300)
    nu_head = (cnt @ M["nu_head"]) / np.maximum(cnt @ M["w"], 1e-300)
    C_head = (cnt @ M["C_head"]) / np.maximum(cnt @ M["w"], 1e-300)
    return np.stack([nu, nu / np.maximum(C, 1e-12), nu_head / np.maximum(C_head, 1e-12), C])


def retained(weights, target, widths):
    """Share of the matched-filter gain of the target profile that a weight profile keeps (strata weighted by their width)."""
    return float((widths * weights * target).sum() ** 2 / ((widths * weights * weights).sum() * (widths * target * target).sum()))


def width_of(means, centers):
    """ln 2 / (-slope) of the log-linear fit over the positive means; nan with fewer than three of them, inf without decay."""
    ok = means > 0
    if ok.sum() < 3:
        return float("nan")
    b = np.polyfit(centers[ok], np.log(means[ok]), 1)[0]
    return math.log(2) / -b if b < 0 else float("inf")


def analyse(cls_name, cls, positions, responses, edges, B=2000):
    NS = len(edges) - 1
    M, count, cover, sizes = class_matrices(cls, positions, responses, edges)
    nq = M["nu"].shape[0]
    point = estimates(M, np.ones(nq))
    rng = np.random.default_rng(0)
    draws = [rng.multinomial(nq, np.ones(nq) / nq) for _ in range(B)]
    boots = np.stack([estimates(M, c) for c in draws])
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)

    def row(b, name):
        return dict(bin=name, n=int(count[b]), cover=float(cover[b]), C=float(point[3, b]), nu=float(point[0, b]), nu_ci=[float(lo[0, b]), float(hi[0, b])],
                    kappa=float(point[1, b]), kappa_ci=[float(lo[1, b]), float(hi[1, b])], kappa_head=float(point[2, b]))

    out = dict(sizes)
    out["strata"] = [row(s, f"{edges[s]:g}-{min(edges[s + 1], 1):g}") for s in range(NS)]
    out["regions"] = [row(NS + k, name) for k, (name, _, _) in enumerate(REGIONS)]
    out["bands"] = [row(NS + len(REGIONS) + k, f"{a:g}-{b:g}") for k, (a, b) in enumerate(BANDS)]
    up, mid, tail = NS, NS + 1, NS + 2

    def ratio(est, a, b):
        return est[1, a] / est[1, b] if est[1, b] > 0 else np.nan

    out["ratios"] = {name: dict(value=float(ratio(point, a, b)), ci=[float(x) for x in np.nanpercentile([ratio(e, a, b) for e in boots], [2.5, 97.5])])
                     for name, a, b in (("0.5-2.5% / 2.5-20%", up, mid), ("2.5-20% / 20-80%", mid, tail))}
    band_cols = slice(NS + len(REGIONS), NS + len(REGIONS) + len(BANDS))
    centers = np.array([(a + b) / 2 for a, b in BANDS])
    out["band_widths"] = dict(C=width_of(point[3, band_cols], centers), nu=width_of(point[0, band_cols], centers))

    # matched-filter gain retained by weight profiles against the forking-value profile of the strata
    e = np.minimum(np.array(edges, float), 1.0)
    widths, qc = np.diff(e), (e[:-1] + e[1:]) / 2
    width, floor = WINDOW[cls_name]
    window = np.maximum(2.0 ** (-qc / width), floor)
    kappa_by_region = np.where(qc < .0025, point[1, 0], np.where(qc < .025, point[1, up], np.where(qc < .2, point[1, mid], point[1, tail])))
    profiles = [("uniform weights", np.ones_like(qc)),
                ("top 20% window", (qc <= 0.2).astype(float)),
                ("top 0.25% of the rank only", (qc <= 0.0025).astype(float)),
                ("CreditFilter window", window),
                ("wide window (double widths)", np.maximum(2.0 ** (-qc / (2 * width)), floor)),
                ("reversed rank", np.maximum(2.0 ** (-(1 - qc) / width), floor)),
                ("exponential fitted to the forking value", 2.0 ** (-qc / round(out["band_widths"]["nu"], 3))),
                ("CreditFilter window with a threefold gain on the top 0.25%", window * np.where(qc <= 0.0025, 3.0, 1.0)),
                ("piecewise fork probability x consequence", point[3, :NS] * kappa_by_region)]
    target = np.clip(point[0, :NS], 0, None)
    reps = np.array([[retained(v, np.clip((c @ M["nu"][:, :NS]) / np.maximum(c @ M["w"][:, :NS], 1e-300), 0, None), widths) for _, v in profiles] for c in draws])
    names = [n for n, _ in profiles]
    out["retention"] = {n: dict(value=retained(v, target, widths), ci=[float(x) for x in np.percentile(reps[:, k], [2.5, 97.5])]) for k, (n, v) in enumerate(profiles)}
    out["retention_differences"] = {}
    for a, b in (("CreditFilter window", "top 20% window"), ("CreditFilter window", "uniform weights"), ("CreditFilter window", "wide window (double widths)")):
        d = reps[:, names.index(a)] - reps[:, names.index(b)]
        out["retention_differences"][f"{a} - {b}"] = dict(value=out["retention"][a]["value"] - out["retention"][b]["value"],
                                                         ci=[float(x) for x in np.percentile(d, [2.5, 97.5])])
    return out


def main():
    args = sys.argv[1:]
    out_json = None
    if "--json" in args:
        out_json = args[args.index("--json") + 1]
        del args[args.index("--json"):args.index("--json") + 2]
    if len(args) != 3:
        sys.exit(__doc__)
    positions, responses, design = read(args[0]), {r["uid"]: r for r in read(args[1])}, json.load(open(args[2]))
    res = {name: analyse(name, cls, positions, responses, design["q_edges"]) for name, cls in (("correct", 1), ("wrong", 0))}

    def f3(x):
        return f"{1e3 * x:.2f}"

    for name, r in res.items():
        print(f"== {name}: {r['responses']} responses of {r['prompts']} prompts, {r['positions']} positions")
        print("  rank             n  cover      C   forking value (1e-3)     consequence (1e-3)      head-restricted")
        for row in r["strata"] + r["regions"] + r["bands"]:
            print(f"  {row['bin']:14s}{row['n']:5d}  {row['cover']:.2f}  {row['C']:.3f}   {f3(row['nu']):>5} [{f3(row['nu_ci'][0])}, {f3(row['nu_ci'][1])}]".ljust(62)
                  + f"{f3(row['kappa']):>6} [{f3(row['kappa_ci'][0])}, {f3(row['kappa_ci'][1])}]".ljust(26) + f"{f3(row['kappa_head']):>6}")
        for k, v in r["ratios"].items():
            print(f"  consequence ratio {k}: {v['value']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]")
        print(f"  widths over the ten bands: fork probability {r['band_widths']['C']:.3f}, forking value {r['band_widths']['nu']:.3f}")
        for k, v in r["retention"].items():
            print(f"  retained gain  {k:60s} {v['value']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]")
        for k, v in r["retention_differences"].items():
            print(f"  difference     {k:60s} {v['value']:.2f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
