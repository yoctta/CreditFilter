"""CreditFilter: the per-token advantage coefficient.

Within each response the tokens are ranked by entropy (rank 1 = highest) and mapped to q = rank / n_valid in (0, 1]
(CF_RANK_SCOPE=batch ranks over all valid tokens of the batch instead).  The advantage of token t of response i becomes

    A_it <- g_i * [ m * w_it + (1 - m) ] * A_it

    w_it = max(kernel(q_it), floor)    kernel exp: 2^(-q / width), hard: 1[q <= CF_HARD_TOP];
                                       width = CF_WIDTH_POS for a correct response, CF_WIDTH_NEG for a wrong one;
                                       floor = CF_FLOOR_POS / CF_FLOOR_NEG by the sign of the response's advantage
    m    = clip((H_hi - Hbar) / (H_hi - H_lo), 0, 1)    fallback: Hbar is the batch-mean token entropy, the band is
                                       CF_FALLBACK_H_LO / CF_FALLBACK_H_HI
    g_i  = (1 - p_g)^gamma             difficulty damping: p_g is the accuracy of the response's prompt group,
                                       gamma = CF_DAMPING_POS / CF_DAMPING_NEG by the sign of the advantage

Options: CF_TOP_GAIN_Q / CF_TOP_GAIN multiply the exp kernel by a gain on the ranks q <= Q; CF_MASS_EQUALIZE=1 rescales
both classes to the larger of the two class masses E_q[max(kernel(q), floor)]; CF_MASS_TARGET_POS/NEG rescale to given
masses; CF_WIDTH_MODE=tracked (CreditFilter-D) replaces the fixed widths by a running average of the width of the
fork-probability profile 1 - sum_v pi_v^2 over the entropy rank, estimated per class on every batch.
"""
import json
import math
import os

import torch


def _envf(name, default):
    v = os.environ.get(name, "")
    return float(v) if v.strip() else default


def _envs(name, default):
    return os.environ.get(name, default).strip().lower()


# tracked widths (CreditFilter-D): per-class width of the fork-probability profile, running average kept across batches
_TRACK = {"loaded": False, "path": None, "step": 0, "width_pos": None, "width_neg": None}
_TRACK_BANDS = [(0.0, .025), (.025, .05), (.05, .075), (.075, .1), (.1, .15), (.15, .2), (.2, .3), (.3, .4), (.4, .6), (.6, .8)]


def reset_tracker():
    """Forget the in-process state; the state file, if any, is read again on the next call."""
    _TRACK.update(loaded=False, path=None, step=0, width_pos=None, width_neg=None)


def _track_half_life_by_class(ent, val, mask, corr, min_rows):
    """Width of the profile of `val` (a per-token quantity) over the within-row entropy rank: ten rank bands up to 0.8,
    band means averaged over rows, log-linear least-squares fit.  Returns {class: (width, r2, n_rows)}; the width is None
    when the class has fewer than min_rows rows, a band is empty or non-positive, or the fitted slope is not negative."""
    B, T = ent.shape
    H = torch.where(mask, ent, torch.full_like(ent, float("-inf")))
    order = torch.argsort(H, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, T + 1, device=ent.device).unsqueeze(0).expand(B, -1).contiguous())
    n_valid = mask.sum(dim=1).clamp(min=1)
    q = (ranks.float() - 0.5) / n_valid.float().unsqueeze(1)
    edges = torch.tensor([b for _, b in _TRACK_BANDS], device=ent.device)
    band = torch.bucketize(q, edges)                       # 0..9 inside the bands, 10 = q > 0.8
    ok = mask & (band < 10)
    idx = (torch.arange(B, device=ent.device).unsqueeze(1) * 10 + band).clamp(max=B * 10 - 1)
    sums = torch.zeros(B * 10, device=ent.device).index_add_(0, idx[ok], val[ok].float())
    cnts = torch.zeros(B * 10, device=ent.device).index_add_(0, idx[ok], torch.ones_like(ent[ok]))
    sums = sums.view(B, 10)
    cnts = cnts.view(B, 10)
    centers = torch.tensor([(a + b) / 2 for a, b in _TRACK_BANDS], device=ent.device)
    out = {}
    for name, sel in (("pos", corr), ("neg", ~corr)):
        n = int(sel.sum())
        if n < min_rows:
            out[name] = (None, None, n)
            continue
        m_rows = sums[sel] / cnts[sel].clamp(min=1)
        has = cnts[sel] > 0
        m = (m_rows * has.float()).sum(dim=0) / has.float().sum(dim=0).clamp(min=1)   # mean over the rows that have the band
        if bool((has.sum(dim=0) == 0).any()) or bool((m <= 0).any()):
            out[name] = (None, None, n)
            continue
        x, y = centers, torch.log(m)
        xm, ym = x.mean(), y.mean()
        slope = float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())
        yhat = ym + slope * (x - xm)
        r2 = float(1.0 - ((y - yhat) ** 2).sum() / ((y - ym) ** 2).sum().clamp(min=1e-12))
        out[name] = ((math.log(2.0) / -slope) if slope < 0 else None, r2, n)
    return out


def _track_state_load(path, init_pos, init_neg):
    if _TRACK["loaded"]:
        return
    _TRACK.update(loaded=True, path=path, step=0, width_pos=init_pos, width_neg=init_neg)
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                d = json.load(f)
            _TRACK.update(step=int(d.get("step", 0)), width_pos=float(d["width_pos"]), width_neg=float(d["width_neg"]))
            print(f"[creditfilter] tracked widths: resumed state from {path}: step={_TRACK['step']} "
                  f"width_pos={_TRACK['width_pos']:.4f} width_neg={_TRACK['width_neg']:.4f}", flush=True)
        except Exception as e:
            print(f"[creditfilter] tracked widths: could not read {path} ({e}); starting from "
                  f"{init_pos:.4f}/{init_neg:.4f}", flush=True)


def _track_state_save():
    path = _TRACK.get("path")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"step": _TRACK["step"], "width_pos": _TRACK["width_pos"], "width_neg": _TRACK["width_neg"]}, f)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[creditfilter] tracked widths: could not save state to {path} ({e})", flush=True)


def apply_credit_filter(data, records, adv, mask, ent) -> dict:
    """Multiply adv ([B,T], the advantages of data.batch) in place by the CreditFilter coefficient.
    records: one object per row with `prompt_uid` and `answer_correct`; mask: bool [B,T]; ent: token entropies [B,T].
    Returns the telemetry dict that the trainer prints as the `[route]` line."""
    width_pos = _envf("CF_WIDTH_POS", 0.10)
    width_neg = _envf("CF_WIDTH_NEG", 0.14)
    floor_pos = _envf("CF_FLOOR_POS", 0.1)
    floor_neg = _envf("CF_FLOOR_NEG", 0.0)
    damp_pos = _envf("CF_DAMPING_POS", 1.0)
    damp_neg = _envf("CF_DAMPING_NEG", 0.0)
    h_lo = _envf("CF_FALLBACK_H_LO", -1.0)
    h_hi = _envf("CF_FALLBACK_H_HI", -1.0)
    kernel = _envs("CF_KERNEL", "exp")
    hard_top = _envf("CF_HARD_TOP", 0.2)
    rank_scope = _envs("CF_RANK_SCOPE", "response")
    rank_key = _envs("CF_RANK_KEY", "entropy")
    width_mode = _envs("CF_WIDTH_MODE", "fixed")
    top_gain_q = _envf("CF_TOP_GAIN_Q", 0.0)
    top_gain = _envf("CF_TOP_GAIN", 1.0)
    if rank_scope not in ("response", "batch"):
        raise RuntimeError(f"[creditfilter] CF_RANK_SCOPE={rank_scope!r} not in (response|batch)")
    if rank_key not in ("entropy", "random", "reversed"):
        raise RuntimeError(f"[creditfilter] CF_RANK_KEY={rank_key!r} not in (entropy|random|reversed)")
    if width_mode not in ("fixed", "tracked"):
        raise RuntimeError(f"[creditfilter] CF_WIDTH_MODE={width_mode!r} not in (fixed|tracked)")
    if rank_key != "entropy" and width_mode != "fixed":
        raise RuntimeError("[creditfilter] CF_RANK_KEY=random|reversed needs CF_WIDTH_MODE=fixed")
    width_init_pos, width_init_neg = width_pos, width_neg
    track = None
    if width_mode == "tracked":
        track_c_pos = _envf("CF_TRACK_C_POS", 1.0)
        track_c_neg = _envf("CF_TRACK_C_NEG", 1.0)
        track_alpha = _envf("CF_TRACK_ALPHA", 0.1)
        track_min_pos = _envf("CF_TRACK_MIN_POS", 0.05)
        track_max_pos = _envf("CF_TRACK_MAX_POS", 0.30)
        track_min_neg = _envf("CF_TRACK_MIN_NEG", 0.07)
        track_max_neg = _envf("CF_TRACK_MAX_NEG", 0.40)
        track_min_rows = int(_envf("CF_TRACK_MIN_ROWS", 64))
        track_path = os.environ.get("CF_TRACK_STATE_PATH", "").strip() or None
        if not (track_c_pos > 0 and track_c_neg > 0 and 0.0 < track_alpha <= 1.0 and 0 < track_min_pos < track_max_pos and 0 < track_min_neg < track_max_neg):
            raise RuntimeError("[creditfilter] tracked widths need CF_TRACK_C_POS/NEG > 0, 0 < CF_TRACK_ALPHA <= 1 and 0 < MIN < MAX per class")
        if rank_scope != "response":
            raise RuntimeError("[creditfilter] CF_WIDTH_MODE=tracked needs CF_RANK_SCOPE=response")
        _track_state_load(track_path, width_init_pos, width_init_neg)
        track = dict(c_pos=track_c_pos, c_neg=track_c_neg, alpha=track_alpha, lo=(track_min_pos, track_min_neg),
                     hi=(track_max_pos, track_max_neg), min_rows=track_min_rows)
    if kernel not in ("exp", "hard"):
        raise RuntimeError(f"[creditfilter] CF_KERNEL={kernel!r} not in (exp|hard)")
    if kernel == "hard" and not (0.0 < hard_top < 1.0):
        raise RuntimeError(f"[creditfilter] CF_HARD_TOP={hard_top} must be in (0,1)")
    if not (0.0 <= top_gain_q < 1.0) or top_gain <= 0.0:
        raise RuntimeError(f"[creditfilter] CF_TOP_GAIN_Q={top_gain_q} must be in [0,1) and CF_TOP_GAIN={top_gain} > 0")
    if top_gain_q > 0.0 and kernel != "exp":
        raise RuntimeError("[creditfilter] CF_TOP_GAIN_Q/CF_TOP_GAIN only apply to CF_KERNEL=exp")
    if not (h_lo > 0.0 and h_hi > h_lo):
        raise RuntimeError("[creditfilter] set CF_FALLBACK_H_LO/H_HI (0 < H_LO < H_HI, in nats of batch-mean token entropy)")
    if not (damp_pos >= 0.0 and damp_neg >= 0.0):
        raise RuntimeError(f"[creditfilter] CF_DAMPING_POS/NEG must be >= 0 (got {damp_pos}, {damp_neg})")
    B, T = adv.shape
    if len(records) != B:
        raise RuntimeError(f"[creditfilter] records {len(records)} != B {B}")

    # rank coordinate
    corr = torch.tensor([1 if getattr(r, "answer_correct", False) else 0 for r in records],
                        device=adv.device, dtype=torch.bool)
    if rank_key == "random":
        key_ent = torch.rand_like(ent)          # a fresh random order of each row's positions every step
    elif rank_key == "reversed":
        key_ent = -ent                          # lowest-entropy tokens ranked first
    else:
        key_ent = ent
    H = torch.where(mask, key_ent, torch.full_like(ent, float("-inf")))
    n_valid = mask.sum(dim=1).clamp(min=1)
    if rank_scope == "batch":
        # rank over all valid tokens of the batch; padded positions get q = 1 and are masked below
        flat_h = key_ent[mask]
        n_b = int(flat_h.numel())
        if n_b < 1:
            raise RuntimeError("[creditfilter] CF_RANK_SCOPE=batch: no valid token in the batch")
        order_f = torch.argsort(flat_h, descending=True, stable=True)
        ranks_f = torch.empty_like(order_f)
        ranks_f[order_f] = torch.arange(1, n_b + 1, device=adv.device)
        q = torch.ones_like(ent)
        q[mask] = ranks_f.float() / float(n_b)
    else:
        order = torch.argsort(H, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ar = torch.arange(1, T + 1, device=adv.device).unsqueeze(0).expand(B, -1).contiguous()
        ranks.scatter_(1, order, ar)
        q = ranks.float() / n_valid.float().unsqueeze(1)

    # tracked widths: estimate on this batch, running average, clamp
    track_tel = {}
    if track is not None:
        if "sum_pi2" not in data.batch.keys():
            raise RuntimeError("[creditfilter] CF_WIDTH_MODE=tracked needs the per-token sum_pi2 of the actor in the batch")
        sum_pi2 = data.batch["sum_pi2"].to(adv.device).float()
        fork_prob = torch.where(mask, (1.0 - sum_pi2).clamp(min=0.0, max=1.0), torch.zeros_like(sum_pi2))
        est = _track_half_life_by_class(ent, fork_prob, mask, corr, track["min_rows"])
        for name, c_k, lo_k, hi_k in (("pos", track["c_pos"], track["lo"][0], track["hi"][0]),
                                      ("neg", track["c_neg"], track["lo"][1], track["hi"][1])):
            width_est, r2, n_rows = est[name]
            key = f"width_{name}"
            track_tel[f"cf/track_width_{name}_raw"] = round(width_est, 4) if width_est is not None else -1.0
            track_tel[f"cf/track_r2_{name}"] = round(r2, 3) if r2 is not None else -1.0
            track_tel[f"cf/track_n_{name}"] = n_rows
            if width_est is not None:
                target = min(max(c_k * width_est, lo_k), hi_k)
                _TRACK[key] = min(max((1.0 - track["alpha"]) * _TRACK[key] + track["alpha"] * target, lo_k), hi_k)
            track_tel[f"cf/track_updated_{name}"] = int(width_est is not None)
        _TRACK["step"] += 1
        _track_state_save()
        width_pos, width_neg = float(_TRACK["width_pos"]), float(_TRACK["width_neg"])
        track_tel.update({"cf/track_step": _TRACK["step"], "cf/track_c_pos": track["c_pos"], "cf/track_c_neg": track["c_neg"],
                          "cf/track_alpha": track["alpha"], "cf/width_pos_init": width_init_pos, "cf/width_neg_init": width_init_neg})
    widths = torch.where(corr, torch.full((B,), width_pos, device=adv.device),
                         torch.full((B,), width_neg, device=adv.device))

    def _kernel(qq, width_rows):
        """Kernel weights before the floor. qq: [R,*] ranks in (0,1]; width_rows: [R]."""
        if kernel == "hard":
            return (qq <= hard_top).float()
        out = torch.pow(2.0, -qq / width_rows.unsqueeze(1))
        if top_gain_q > 0.0:
            out = torch.where(qq <= top_gain_q, out * top_gain, out)
        return out

    base = _kernel(q, widths)
    row_sign = torch.sign(adv.sum(dim=1))          # the advantage is constant along a response
    floor = torch.where(row_sign > 0,
                        torch.full((B,), floor_pos, device=adv.device),
                        torch.full((B,), floor_neg, device=adv.device))

    # class-mass equalisation: both classes are rescaled to the larger of the two class masses
    mass_eq = int(_envf("CF_MASS_EQUALIZE", 0))
    mt_pos_cfg = _envf("CF_MASS_TARGET_POS", 0.0)
    mt_neg_cfg = _envf("CF_MASS_TARGET_NEG", 0.0)
    if mass_eq not in (0, 1):
        raise RuntimeError(f"[creditfilter] CF_MASS_EQUALIZE={mass_eq} must be 0 or 1")
    if mass_eq == 1 and (mt_pos_cfg > 0.0 or mt_neg_cfg > 0.0):
        raise RuntimeError("[creditfilter] CF_MASS_EQUALIZE=1 excludes CF_MASS_TARGET_POS/NEG")
    if mass_eq == 1 and rank_scope != "response":
        raise RuntimeError("[creditfilter] CF_MASS_EQUALIZE=1 needs CF_RANK_SCOPE=response")
    mass_nat_pos = mass_nat_neg = mass_tgt_eff = -1.0
    if mass_eq == 1:
        qs_cm = (torch.arange(1, 2001, device=adv.device).float()) / 2000.0
        width_cls = torch.tensor([width_pos, width_neg], device=adv.device, dtype=torch.float32)
        grid_cls = _kernel(qs_cm.unsqueeze(0).expand(2, -1), width_cls)

        def _cmass(k, e):
            return float(torch.clamp(grid_cls[k], min=float(e)).mean())

        mass_nat_pos, mass_nat_neg = _cmass(0, floor_pos), _cmass(1, floor_neg)
        mass_tgt_eff = max(mass_nat_pos, mass_nat_neg)
        mt_pos_cfg = mt_neg_cfg = mass_tgt_eff
    w = torch.maximum(base, floor.unsqueeze(1))
    w = torch.where(mask, w, torch.ones_like(w))

    # fallback: mix toward uniform weights as the batch-mean token entropy crosses the band
    n_tok = mask.float().sum().clamp(min=1.0)
    Hbar = float((ent * mask.float()).sum() / n_tok)

    def _H_of(rows):
        mk = mask & rows.unsqueeze(1)
        nt = float(mk.float().sum())
        return float((ent * mk.float()).sum() / nt) if nt > 0 else float("nan")

    H_pos, H_neg = _H_of(corr), _H_of(~corr)
    m_mix = min(1.0, max(0.0, (h_hi - Hbar) / (h_hi - h_lo)))
    m_row = torch.full((B,), m_mix, device=adv.device)

    # rescale each row's window so that its mass E_q[max(kernel(q), floor)] equals the target of its class
    mt_pos, mt_neg = mt_pos_cfg, mt_neg_cfg
    mass_scale = torch.ones(B, device=adv.device)
    if mt_pos > 0.0 or mt_neg > 0.0:
        if rank_scope != "response":
            raise RuntimeError("[creditfilter] CF_MASS_TARGET_POS/NEG need CF_RANK_SCOPE=response")
        qs_mm = (torch.arange(1, 2001, device=adv.device).float()) / 2000.0
        grid_mm = _kernel(qs_mm.unsqueeze(0).expand(B, -1), widths)
        mass_row_mm = torch.maximum(grid_mm, floor.unsqueeze(1)).mean(dim=1)
        tgt_mm = torch.where(row_sign > 0, torch.full((B,), mt_pos, device=adv.device),
                             torch.full((B,), mt_neg, device=adv.device))
        mass_scale = torch.where(tgt_mm > 0, tgt_mm / mass_row_mm.clamp(min=1e-6), torch.ones_like(tgt_mm))
        w = torch.where(mask, w * mass_scale.unsqueeze(1), torch.ones_like(w))
    wt = m_row.unsqueeze(1) * w + (1.0 - m_row.unsqueeze(1))
    wt = torch.where(mask, wt, torch.ones_like(wt))

    # difficulty damping
    uids = [getattr(r, "prompt_uid", None) for r in records]
    if any(u is None for u in uids):
        raise RuntimeError("[creditfilter] records lack prompt_uid")
    acc_by = {}
    for u, r in zip(uids, records):
        acc_by.setdefault(u, []).append(1.0 if getattr(r, "answer_correct", False) else 0.0)
    p_by = {u: sum(v) / len(v) for u, v in acc_by.items()}
    p_row = torch.tensor([p_by[u] for u in uids], device=adv.device, dtype=torch.float32)
    one_m_p = (1.0 - p_row).clamp(min=0.0)
    g = torch.where(row_sign > 0, torch.pow(one_m_p, damp_pos),
                    torch.where(row_sign < 0, torch.pow(one_m_p, damp_neg),
                                torch.ones_like(p_row)))

    # telemetry and the mass check
    with torch.no_grad():
        d = {"route/mode": "creditfilter",
             "cf/kernel": kernel,
             "cf/rank_scope": rank_scope, "cf/rank_key": rank_key,
             "cf/hard_top": hard_top if kernel == "hard" else -1.0,
             "cf/width_pos": width_pos, "cf/width_neg": width_neg,
             "cf/width_mode": width_mode, **track_tel,
             "cf/floor_pos": floor_pos, "cf/floor_neg": floor_neg,
             "cf/damping_pos": damp_pos, "cf/damping_neg": damp_neg,
             "cf/fallback_h_lo": h_lo, "cf/fallback_h_hi": h_hi, "cf/batch_H": round(Hbar, 4),
             "cf/fallback_mix": round(m_mix, 4),
             "cf/H_pos": round(H_pos, 4) if H_pos == H_pos else -1.0, "cf/H_neg": round(H_neg, 4) if H_neg == H_neg else -1.0,
             "cf/mass_target_pos": mt_pos, "cf/mass_target_neg": mt_neg,
             "cf/top_gain_q": top_gain_q, "cf/top_gain": top_gain,
             "cf/mass_equalize": mass_eq,
             "cf/mass_nat_pos": round(mass_nat_pos, 6), "cf/mass_nat_neg": round(mass_nat_neg, 6), "cf/mass_target_eff": round(mass_tgt_eff, 6),
             "cf/n_groups": len(p_by), "cf/p_mean": round(float(p_row.mean()), 4)}
        qs = (torch.arange(1, 2001, device=adv.device).float()) / 2000.0
        # expected mass of each row on a 2000-point grid of q
        base_grid = _kernel(qs.unsqueeze(0).expand(B, -1), widths)
        e_row_uniform = torch.maximum(base_grid, floor.unsqueeze(1)).mean(dim=1) * mass_scale
        if rank_scope == "batch":
            # under the batch rank a row's q is not uniform: its expected mass is the mean of its own weights
            e_row = (w * mask.float()).sum(dim=1) / n_valid.float()
            qb = q[mask]
            top_frac = float((qb <= 0.2).float().mean())
            d["cf/qb_mean"] = round(float(qb.mean()), 4)
            d["cf/qb_top20_frac"] = round(top_frac, 4)
            row_top = ((q <= 0.2) & mask).float().sum(dim=1) / n_valid.float()
            for pct in (10, 50, 90):
                d[f"cf/row_top20_p{pct}"] = round(float(torch.quantile(row_top, pct / 100.0)), 4)
            if abs(top_frac - 0.2) > 0.01:
                raise RuntimeError(f"[creditfilter] CF_RANK_SCOPE=batch: the share of q <= 0.2 in the batch is {top_frac:.4f}, not 0.2")
        else:
            e_row = e_row_uniform
        # mass check: the measured class mass must match the expected one within CF_MASS_GUARD_TOL; rows shorter than
        # CF_MASS_GUARD_MINLEN tokens are left out of the comparison
        guard_minlen = int(_envf("CF_MASS_GUARD_MINLEN", 0))
        guard_tol = _envf("CF_MASS_GUARD_TOL", 0.03)
        for tag, s in (("pos", row_sign > 0), ("neg", row_sign < 0)):
            k = int(s.sum())
            d[f"cf/n_{tag}"] = k
            if k == 0:
                continue
            wm = (wt * mask.float())[s].sum(dim=1) / n_valid[s].float()
            d[f"cf/mass_{tag}"] = round(float(wm.mean()), 4)
            exp_row = m_row[s] * e_row[s] + (1.0 - m_row[s])
            exp_mass = float(exp_row.mean())
            d[f"cf/mass_{tag}_expect"] = round(exp_mass, 4)
            if rank_scope == "batch":
                d[f"cf/mass_{tag}_expect_uniform"] = round(float((m_row[s] * e_row_uniform[s] + (1.0 - m_row[s])).mean()), 4)
            mass_meas, mass_exp = float(wm.mean()), exp_mass
            if guard_minlen > 0:
                long_rows = n_valid[s] >= guard_minlen
                d[f"cf/guard_{tag}_rows_short"] = int((~long_rows).sum())
                if int(long_rows.sum()) >= 4:
                    mass_meas, mass_exp = float(wm[long_rows].mean()), float(exp_row[long_rows].mean())
            d[f"cf/guard_{tag}_dev"] = round(abs(mass_meas - mass_exp), 4)
            if k >= 4 and abs(mass_meas - mass_exp) > guard_tol:
                raise RuntimeError(f"[creditfilter] mass check: {tag} class measured {mass_meas:.3f}, expected {mass_exp:.3f} (tolerance {guard_tol})")
            d[f"cf/g_{tag}_mean"] = round(float(g[s].mean()), 4)
            d[f"cf/g_{tag}_min"] = round(float(g[s].min()), 4)
            d[f"cf/mass_scale_{tag}_mean"] = round(float(mass_scale[s].mean()), 4)
    adv.mul_(wt * g.unsqueeze(1))
    return d
