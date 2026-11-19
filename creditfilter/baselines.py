"""Token-weighting rules compared with CreditFilter, applied to the advantage tensor in place.

    apply_entropy_mask   80/20 rule (arXiv 2506.01939): the top CF_MASK_TOP fraction of tokens by entropy keep their
                         gradient, per response or over the whole batch (CF_MASK_SCOPE); the rest are zeroed
    apply_stare          STARE (arXiv 2606.19236): surprisal-quantile reweighting behind a batch-entropy gate
    apply_scgrpo         SC-GRPO (arXiv 2606.18810): weight f = D / (D + c) from a self-conditioned teacher KL, which
                         token_signals.py computes before the advantage step
"""
import math
import os

import torch


def _envf(name, default):
    v = os.environ.get(name, "")
    return float(v) if v.strip() else default


# under the per-response cut a response keeps at least MIN_KEPT_TOKENS tokens (or all of them if it is shorter)
MIN_KEPT_TOKENS = 4


def top_entropy_mask(entropys: torch.Tensor, response_mask: torch.Tensor, top: float, min_kept: int, scope: str = "response"):
    """Bool mask [B,T] of the top fraction `top` of the response tokens by entropy, and the kept fraction of each row [B]."""
    mask = response_mask.to(bool)
    H = torch.where(mask, entropys, torch.full_like(entropys, float("-inf")))
    n_valid = mask.sum(dim=1).clamp(min=1)
    if scope == "batch":
        flat = H[mask]
        k = max(1, int(math.ceil(top * flat.numel())))
        thr = torch.kthvalue(flat, flat.numel() - k + 1).values
    else:
        k = (n_valid.float() * top).ceil().long().clamp(min=min_kept)
        k = torch.minimum(k, n_valid)
        srt = torch.sort(H, dim=1, descending=True).values
        thr = srt.gather(1, (k - 1).unsqueeze(1))
    kept = (H >= thr) & mask
    return kept, kept.sum(dim=1).float() / n_valid.float()


def apply_entropy_mask(data, records, adv, mask, ent) -> dict:
    """80/20 rule: the top CF_MASK_TOP entropy tokens keep weight 1, every other response token gets 0, without
    renormalisation.  CF_MASK_SCOPE=response ranks within each response, batch over all valid tokens of the batch."""
    top = _envf("CF_MASK_TOP", 0.2)
    scope = os.environ.get("CF_MASK_SCOPE", "response").strip().lower()
    if scope not in ("response", "batch"):
        raise RuntimeError(f"[entropy_mask] CF_MASK_SCOPE={scope!r} not in (response|batch)")
    kept, kept_frac = top_entropy_mask(ent, mask, top, MIN_KEPT_TOKENS, scope)
    w = kept.float()
    adv.mul_(torch.where(mask, w, torch.ones_like(w)))
    return {
        "route/mode": "entropy_mask",
        "route/mask_top": top,
        "route/mask_scope": scope,
        "route/kept_frac": (kept.sum() / mask.sum()).item(),
        "route/kept_frac_row_mean": kept_frac.mean().item(),
        # under the batch cut a response of low-entropy tokens can keep none
        "route/rows_without_kept_tokens": int((kept.sum(dim=1) == 0).sum().item()),
    }


def apply_stare(data, records, adv, mask, ent) -> dict:
    """STARE (arXiv 2606.19236) with the batch-level entropy gate and fixed weights: when the batch-mean entropy is below
    CF_STARE_H_TARGET, the positive-advantage tokens in the top P of surprisal are multiplied by W; with M < 1 the
    negative-advantage tokens in the top P are multiplied by M."""
    W = _envf("CF_STARE_W", 1.1)
    M = _envf("CF_STARE_M", 1.0)
    P = _envf("CF_STARE_P", 0.10)
    htgt = _envf("CF_STARE_H_TARGET", -1.0)
    if htgt <= 0.0:
        raise RuntimeError("[stare] set CF_STARE_H_TARGET > 0 (nats of batch-mean token entropy)")
    if not (W >= 1.0 and 0.0 < M <= 1.0 and 0.0 < P < 1.0):
        raise RuntimeError(f"[stare] need W >= 1, 0 < M <= 1 and 0 < P < 1 (got W={W}, M={M}, P={P})")
    if "old_log_probs" not in data.batch.keys():
        raise RuntimeError("[stare] CF_ROUTE=stare needs 'old_log_probs' in the batch")
    s = -data.batch["old_log_probs"].to(adv.device)
    n_tok = mask.float().sum().clamp(min=1.0)
    Hbar = float((ent * mask.float()).sum() / n_tok)
    gate_on = Hbar < htgt
    row_sign = torch.sign(adv.sum(dim=1))
    d = {"route/mode": "stare", "stare/W": W, "stare/M": M, "stare/P": P,
         "stare/htgt": htgt, "stare/batch_H": round(Hbar, 4),
         "stare/gate_on": int(gate_on)}
    if not gate_on:
        return d
    w = torch.ones_like(adv)
    for tag, sgn, wt in (("pos", 1.0, W), ("neg", -1.0, M)):
        if abs(wt - 1.0) < 1e-9:
            continue
        tok = (row_sign == sgn).unsqueeze(1) & mask
        k = int(tok.sum())
        d[f"stare/n_{tag}_tok"] = k
        if k == 0:
            continue
        thr = torch.quantile(s[tok].float(), 1.0 - P)
        selected = tok & (s >= thr)
        w = torch.where(selected, torch.full_like(w, wt), w)
        d[f"stare/thr_{tag}"] = round(float(thr), 4)
        d[f"stare/n_{tag}_sel"] = int(selected.sum())
        d[f"stare/frac_{tag}_sel"] = round(float(selected.sum()) / k, 4)
    with torch.no_grad():
        for tag, sgn in (("pos", 1.0), ("neg", -1.0)):
            rows = row_sign == sgn
            if int(rows.sum()) == 0:
                continue
            nv = mask[rows].float().sum(dim=1).clamp(min=1.0)
            d[f"stare/mass_{tag}"] = round(
                float(((w * mask.float())[rows].sum(dim=1) / nv).mean()), 4)
        # check: only the top P of the positive side is reweighted, so its mean weight is about 1 + P * (W - 1)
        mp = d.get("stare/mass_pos")
        if mp is not None and not (1.0 <= mp <= 1.0 + 2.0 * P * (W - 1.0) + 0.01):
            raise RuntimeError(f"[stare] mean positive-side weight {mp} outside "
                               f"[1, {1.0 + 2.0 * P * (W - 1.0) + 0.01:.4f}]")
    adv.mul_(w)
    return d


def apply_scgrpo(data, records, adv, mask, ent) -> dict:
    """SC-GRPO (arXiv 2606.18810): weight f = D / (D + c) with D the teacher KL of the token and c = max(percentile
    CF_SCGRPO_PCT of the KL over the weighted tokens, CF_SCGRPO_CMIN).  Rows without a teacher KL (NaN) keep f = 1."""
    if "sc_kl" not in data.batch.keys():
        raise RuntimeError("[scgrpo] CF_ROUTE=scgrpo needs 'sc_kl' in the batch (the token-signal stage)")
    pct = _envf("CF_SCGRPO_PCT", 75.0) / 100.0
    cmin = _envf("CF_SCGRPO_CMIN", 1e-4)
    if not (0.0 < pct < 1.0):
        raise RuntimeError(f"[scgrpo] CF_SCGRPO_PCT={pct * 100} must be in (0,100)")
    D = data.batch["sc_kl"].to(adv.device).float()
    act = mask & D.isfinite()
    d = {"route/mode": "scgrpo", "sc/pct": pct, "sc/cmin": cmin}
    if act.any():
        vals = D[act]
        k = max(1, min(vals.numel(), int(math.ceil(pct * vals.numel()))))
        c = float(torch.kthvalue(vals, k).values)
        c = max(c, cmin)
        f = torch.where(act, D / (D + c), torch.ones_like(D))
        adv.mul_(torch.where(mask, f, torch.ones_like(f)))
        fw = f[act]
        for q, tag in ((0.5, "p50"), (0.75, "p75"), (0.95, "p95")):
            kk = max(1, min(vals.numel(), int(math.ceil(q * vals.numel()))))
            d[f"sc/kl_{tag}"] = round(float(torch.kthvalue(vals, kk).values), 6)
        d.update({
            "sc/c": round(c, 6),
            "sc/w_mean": round(float(fw.mean()), 4),
            "sc/frac_w_gt_half": round(float((fw > 0.5).float().mean()), 4),
            "sc/kl_var": round(float(vals.var()), 6),
            "sc/coverage": round(float(act.float().sum() / mask.float().sum().clamp(min=1)), 4),
        })
        if "sc_row_correct" in data.batch.keys():
            rc = data.batch["sc_row_correct"].to(adv.device)
            for v, tag in ((1.0, "correct"), (0.0, "incorrect")):
                sel = (rc == v).unsqueeze(1) & act
                if sel.any():
                    d[f"sc/w_mean_{tag}"] = round(float(f[sel].mean()), 4)
    else:
        d["sc/coverage"] = 0.0
    st = data.meta_info.get("sc_stats", {}) if hasattr(data, "meta_info") else {}
    for k2, v2 in st.items():
        d[f"sc/{k2}"] = v2
    return d
