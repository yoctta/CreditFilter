"""CPU unit tests of the CreditFilter coefficient (creditfilter/credit_filter.py).
Run:  python -m pytest tests/"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creditfilter import credit_filter as cf  # noqa: E402

PASSED = []


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    PASSED.append(name)


def mkbatch(B=8, T=32, seed=0, groups=2, pos_frac=0.5):
    """Synthetic batch: adv has a constant sign per row (group-standardised semantics), ent >= 0, ragged mask, grouped by prompt_uid."""
    g = torch.Generator().manual_seed(seed)
    ent = torch.rand(B, T, generator=g) * 2.0
    mask = torch.ones(B, T, dtype=torch.bool)
    for i in range(B):
        n = T - (i % 4) * 3
        mask[i, n:] = False
    sign = torch.tensor([1.0 if i < int(B * pos_frac) else -1.0 for i in range(B)])
    adv = sign.unsqueeze(1) * torch.ones(B, T) * 0.5
    adv = torch.where(mask, adv, torch.zeros_like(adv))
    recs = [SimpleNamespace(prompt_uid=f"u{i % groups}", answer_correct=bool(sign[i] > 0)) for i in range(B)]
    return SimpleNamespace(batch={}), recs, adv, mask, ent


def setenv(**kw):
    for k in list(os.environ):
        if k.startswith("CF_"):
            del os.environ[k]
    for k, v in kw.items():
        os.environ[k] = str(v)


def window_reference(adv, mask, ent, recs, width_pos=0.10, width_neg=0.14, floor_pos=0.1, floor_neg=0.0):
    """Closed-form window on the within-row entropy rank: w = max(2^(-q/width_c), floor_s)."""
    B, T = adv.shape
    H = torch.where(mask, ent, torch.full_like(ent, float("-inf")))
    order = torch.argsort(H, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, T + 1).unsqueeze(0).expand(B, -1).contiguous())
    q = ranks.float() / mask.sum(1, keepdim=True).clamp(min=1).float()
    corr = torch.tensor([r.answer_correct for r in recs])
    widths = torch.where(corr, torch.full((B,), width_pos), torch.full((B,), width_neg)).unsqueeze(1)
    sign = torch.sign(adv.sum(1))
    floor = torch.where(sign > 0, torch.full((B,), floor_pos), torch.full((B,), floor_neg)).unsqueeze(1)
    w = torch.maximum(torch.pow(2.0, -q / widths), floor)
    return torch.where(mask, w, torch.ones_like(w))


INERT = dict(CF_FALLBACK_H_LO=5.0, CF_FALLBACK_H_HI=9.0, CF_DAMPING_POS=0.0, CF_DAMPING_NEG=0.0)   # fallback and damping inert
BASE = dict(CF_FALLBACK_H_LO=1.5, CF_FALLBACK_H_HI=3.0, CF_WIDTH_POS=0.10, CF_WIDTH_NEG=0.14,
            CF_FLOOR_POS=0.1, CF_FLOOR_NEG=0.0, CF_DAMPING_POS=0.0, CF_DAMPING_NEG=0.0)


def test_fallback_band_required():
    setenv()
    data, recs, adv, mask, ent = mkbatch()
    try:
        cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
        check("missing H_LO/H_HI hard error", False)
    except RuntimeError as e:
        check("missing H_LO/H_HI hard error", "H_LO/H_HI" in str(e))


def test_window_matches_closed_form():
    setenv(**INERT)
    data, recs, adv, mask, ent = mkbatch()
    a = adv.clone()
    cf.apply_credit_filter(data, recs, a, mask, ent)
    w = window_reference(adv, mask, ent, recs)
    check("fallback inert, damping 0: coefficient == max(2^(-q/width), floor) elementwise", torch.allclose(a, adv * w, atol=1e-6))


def test_fallback_endpoints_and_interpolation():
    setenv(CF_FALLBACK_H_LO=0.01, CF_FALLBACK_H_HI=0.02, CF_DAMPING_POS=0.0, CF_DAMPING_NEG=0.0)
    data, recs, adv, mask, ent = mkbatch()
    a = adv.clone()
    d = cf.apply_credit_filter(data, recs, a, mask, ent)
    check("batch entropy above H_HI -> fallback_mix 0 -> uniform weights (adv unchanged)",
          torch.allclose(a, adv, atol=1e-6) and d["cf/fallback_mix"] == 0.0)
    setenv(CF_FALLBACK_H_LO=0.5, CF_FALLBACK_H_HI=1.5, CF_DAMPING_POS=0.0, CF_DAMPING_NEG=0.0)
    data, recs, adv, mask, ent = mkbatch()
    d = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
    mg, Hb = d["cf/fallback_mix"], d["cf/batch_H"]
    check("fallback linear inside the band: mix = (H_HI - Hbar)/(H_HI - H_LO)", abs(mg - (1.5 - Hb) / 1.0) < 1e-3 and 0.0 < mg < 1.0)


def test_difficulty_damping():
    setenv(CF_FALLBACK_H_LO=5.0, CF_FALLBACK_H_HI=9.0, CF_DAMPING_POS=1.0, CF_DAMPING_NEG=0.0)
    data, recs, adv, mask, ent = mkbatch(groups=2, pos_frac=0.5)     # 2 groups x 4 rows, 2 correct each -> p_g = 0.5
    a = adv.clone()
    d = cf.apply_credit_filter(data, recs, a, mask, ent)
    w = window_reference(adv, mask, ent, recs)
    pos = torch.tensor([r.answer_correct for r in recs])
    check("p_mean = 0.5", abs(d["cf/p_mean"] - 0.5) < 1e-6)
    check("positive rows = window x (1 - p_g)^1 = x0.5", torch.allclose(a[pos], (adv * w)[pos] * 0.5, atol=1e-6))
    check("negative rows with damping exponent 0 = pure window", torch.allclose(a[~pos], (adv * w)[~pos], atol=1e-6))
    check("telemetry g_pos_mean = 0.5", abs(d["cf/g_pos_mean"] - 0.5) < 1e-6)
    setenv(CF_FALLBACK_H_LO=5.0, CF_FALLBACK_H_HI=9.0, CF_DAMPING_POS=1.0, CF_DAMPING_NEG=1.0)
    data, recs, adv, mask, ent = mkbatch(groups=2, pos_frac=0.5)
    a = adv.clone()
    cf.apply_credit_filter(data, recs, a, mask, ent)
    check("symmetric damping also damps negative rows", torch.allclose(a[~pos], (adv * w)[~pos] * 0.5, atol=1e-6))


def test_mass_guard_and_determinism():
    setenv(CF_FALLBACK_H_LO=0.5, CF_FALLBACK_H_HI=1.5, CF_DAMPING_POS=1.0, CF_DAMPING_NEG=0.0)
    data, recs, adv, mask, ent = mkbatch(B=16, T=64, groups=4)
    d = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
    for tag in ("pos", "neg"):
        check(f"{tag} class mass within the guard tolerance of the closed form", abs(d[f"cf/mass_{tag}"] - d[f"cf/mass_{tag}_expect"]) <= 0.03)
    setenv(CF_FALLBACK_H_LO=0.5, CF_FALLBACK_H_HI=1.5)
    data, recs, adv, mask, ent = mkbatch(B=12, T=48, groups=3)
    a1, a2 = adv.clone(), adv.clone()
    cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a1, mask, ent)
    cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a2, mask, ent)
    check("determinism", torch.equal(a1, a2))


def test_batch_rank_scope():
    setenv(**INERT, CF_FLOOR_POS=0.1, CF_FLOOR_NEG=0.0)
    data, recs, adv, mask, ent = mkbatch(B=8, T=32)
    ent_b = ent.clone()
    ent_b[0] = 3.0 + torch.rand(32, generator=torch.Generator().manual_seed(1))       # row 0 all high entropy
    ent_b[7] = 0.01 * torch.rand(32, generator=torch.Generator().manual_seed(2))      # row 7 all low entropy (negative row, floor 0)
    a_resp = adv.clone()
    d_resp = cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a_resp, mask, ent_b)
    os.environ["CF_RANK_SCOPE"] = "batch"
    a_bat = adv.clone()
    d_bat = cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a_bat, mask, ent_b)
    w_resp = (a_resp / adv.abs().clamp(min=1e-9)).abs(); w_bat = (a_bat / adv.abs().clamp(min=1e-9)).abs()
    check("telemetry rank_scope", d_bat["cf/rank_scope"] == "batch" and d_resp["cf/rank_scope"] == "response")
    check("batch rank: whole-batch q<=0.2 share = 0.2", abs(d_bat["cf/qb_top20_frac"] - 0.2) <= 0.01 and abs(d_bat["cf/qb_mean"] - 0.5) < 0.03)
    _q0 = (torch.arange(1, 33).float() / 220.0); _cf0 = float(torch.pow(2.0, -_q0 / 0.10).mean())
    check("batch rank: the all-high-entropy row holds batch ranks 1..32, row mean == closed form",
          float(w_bat[0][mask[0]].min()) > 0.35 and abs(float(w_bat[0][mask[0]].mean()) - _cf0) < 0.01 and _cf0 > 0.6)
    check("batch rank: the all-low-entropy negative row drops to the floor (0)", float(w_bat[7][mask[7]].mean()) < 0.05)
    check("response rank: both rows carry the closed-form class mass",
          abs(float(w_resp[0][mask[0]].mean()) - d_resp["cf/mass_pos_expect"]) < 0.03
          and abs(float(w_resp[7][mask[7]].mean()) - d_resp["cf/mass_neg_expect"]) < 0.03)
    check("batch rank: per-row selected fraction uneven", d_bat["cf/row_top20_p10"] < 0.05 and d_bat["cf/row_top20_p90"] > d_bat["cf/row_top20_p50"] > -1)
    check("batch rank: guard uses the actual q, uniform reference reported separately",
          abs(d_bat["cf/mass_pos"] - d_bat["cf/mass_pos_expect"]) < 1e-3 and "cf/mass_pos_expect_uniform" in d_bat)
    a_bat2 = adv.clone(); cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a_bat2, mask, ent_b)
    check("batch rank: determinism", torch.allclose(a_bat, a_bat2))
    os.environ["CF_RANK_SCOPE"] = "corpus"
    try:
        cf.apply_credit_filter(SimpleNamespace(batch={}), recs, adv.clone(), mask, ent)
        check("illegal CF_RANK_SCOPE hard error", False)
    except RuntimeError as e:
        check("illegal CF_RANK_SCOPE hard error", "CF_RANK_SCOPE" in str(e))


def _profile_batch(B=64, T=200, lamH=0.10, lamC=0.12, seed=13):
    """Entropies with profile 2*2^(-q/lamH) (positions shuffled); fork probability C = 0.5*2^(-q/lamC) on the entropy rank."""
    g = torch.Generator().manual_seed(seed)
    q = (torch.arange(T).float() + 0.5) / T
    prof = 2.0 * torch.pow(2.0, -q / lamH)
    ent = torch.stack([prof[torch.randperm(T, generator=g)] for _ in range(B)])
    mask = torch.ones(B, T, dtype=torch.bool)
    sign = torch.tensor([1.0 if i < B // 2 else -1.0 for i in range(B)])
    adv = sign.unsqueeze(1) * torch.ones(B, T) * 0.5
    recs = [SimpleNamespace(prompt_uid=f"u{i % 4}", answer_correct=bool(sign[i] > 0)) for i in range(B)]
    order = torch.argsort(ent, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order); ranks.scatter_(1, order, torch.arange(1, T + 1).unsqueeze(0).expand(B, -1).contiguous())
    qr = (ranks.float() - 0.5) / T
    C = 0.5 * torch.pow(2.0, -qr / lamC)
    return SimpleNamespace(batch={"sum_pi2": 1.0 - C}), recs, adv, mask, ent


TRACK = dict(CF_FALLBACK_H_LO=50.0, CF_FALLBACK_H_HI=90.0, CF_DAMPING_POS=0.0, CF_DAMPING_NEG=0.0,
             CF_WIDTH_POS=0.10, CF_WIDTH_NEG=0.14, CF_WIDTH_MODE="tracked",
             CF_TRACK_C_POS=1.0, CF_TRACK_C_NEG=1.0, CF_TRACK_ALPHA=0.5,
             CF_TRACK_MIN_POS=0.05, CF_TRACK_MAX_POS=0.30, CF_TRACK_MIN_NEG=0.07, CF_TRACK_MAX_NEG=0.40, CF_TRACK_MIN_ROWS=8)


def test_tracked_widths():
    state = os.path.join(tempfile.mkdtemp(), "track_state.json")
    setenv(**TRACK, CF_TRACK_STATE_PATH=state)
    cf.reset_tracker()
    data, recs, adv, mask, ent = _profile_batch()
    d1 = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
    check("telemetry reports the tracked mode", d1["cf/width_mode"] == "tracked")
    check("the estimate recovers the width 0.12 of the fork-probability profile within 5% (both classes)",
          abs(d1["cf/track_width_pos_raw"] - 0.12) < 0.006 and abs(d1["cf/track_width_neg_raw"] - 0.12) < 0.006 and d1["cf/track_r2_pos"] > 0.99)
    check("running average, step 1 = 0.5*init + 0.5*c*estimate (pos 0.11, neg 0.13)",
          abs(d1["cf/width_pos"] - 0.11) < 0.004 and abs(d1["cf/width_neg"] - 0.13) < 0.004
          and d1["cf/width_pos_init"] == 0.10 and d1["cf/track_step"] == 1)
    for _ in range(12):
        d2 = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
    check("EMA converges to c*estimate (both classes)",
          abs(d2["cf/width_pos"] - 1.0 * d2["cf/track_width_pos_raw"]) < 0.002 and abs(d2["cf/width_neg"] - 1.0 * d2["cf/track_width_neg_raw"]) < 0.002)
    check("telemetry step", d2["cf/track_step"] == 13)
    check("mass telemetry uses the smoothed widths", abs(d2["cf/mass_pos"] - d2["cf/mass_pos_expect"]) < 0.02)
    os.environ["CF_TRACK_C_POS"] = "10.0"; os.environ["CF_TRACK_C_NEG"] = "10.0"
    for _ in range(12):
        d3 = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
    check("clamp at MAX (0.30/0.40) within the EMA transient",
          d3["cf/width_pos"] <= 0.30 + 1e-9 and d3["cf/width_neg"] <= 0.40 + 1e-9 and abs(d3["cf/width_pos"] - 0.30) < 1e-4 and abs(d3["cf/width_neg"] - 0.40) < 1e-4)
    check("state file written", os.path.exists(state) and json.load(open(state))["step"] == 25)
    cf.reset_tracker()
    d4 = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
    check("resume from the state file (step continues, widths continue near MAX)", d4["cf/track_step"] == 26 and abs(d4["cf/width_pos"] - 0.30) < 1e-4)
    cf.reset_tracker(); os.environ["CF_TRACK_STATE_PATH"] = ""
    os.environ["CF_TRACK_C_POS"] = "1.0"; os.environ["CF_TRACK_C_NEG"] = "1.0"; os.environ["CF_TRACK_MIN_ROWS"] = "40"
    d5 = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)   # 32 rows per class < 40
    check("classes below MIN_ROWS are not updated (widths stay at init)", d5["cf/track_updated_pos"] == 0 and abs(d5["cf/width_pos"] - 0.10) < 1e-9)
    os.environ["CF_RANK_SCOPE"] = "batch"
    try:
        cf.apply_credit_filter(SimpleNamespace(batch={"sum_pi2": data.batch["sum_pi2"]}), recs, adv.clone(), mask, ent)
        check("tracked + batch rank hard error", False)
    except RuntimeError as e:
        check("tracked + batch rank hard error", "CF_RANK_SCOPE=response" in str(e))
    cf.reset_tracker(); setenv(**TRACK, CF_TRACK_STATE_PATH="")
    try:
        cf.apply_credit_filter(SimpleNamespace(batch={}), recs, adv.clone(), mask, ent)
        check("tracked without sum_pi2 -> hard error", False)
    except RuntimeError as e:
        check("tracked without sum_pi2 -> hard error", "sum_pi2" in str(e))
    cf.reset_tracker()


def test_mass_targets_and_equalisation():
    setenv(**BASE)
    data, recs, adv, mask, ent = mkbatch(B=64, T=200, seed=5)
    a_base = adv.clone(); d_base = cf.apply_credit_filter(data, recs, a_base, mask, ent)
    check("mass targets 0 (default): mass_scale 1, target keys 0",
          d_base["cf/mass_target_pos"] == 0.0 and abs(d_base["cf/mass_scale_pos_mean"] - 1.0) < 1e-9 and abs(d_base["cf/mass_scale_neg_mean"] - 1.0) < 1e-9)
    setenv(**{**BASE, "CF_MASS_TARGET_POS": 0.1964, "CF_MASS_TARGET_NEG": 0.2003})
    a_mm = adv.clone(); d_mm = cf.apply_credit_filter(data, recs, a_mm, mask, ent)
    check("CreditFilter's widths with their own class masses as targets -> scale ~1, adv identical within 0.3%",
          abs(d_mm["cf/mass_scale_pos_mean"] - 1.0) < 2e-3 and abs(d_mm["cf/mass_scale_neg_mean"] - 1.0) < 2e-3
          and float((a_mm - a_base).abs().max()) < 3e-3 * float(a_base.abs().max()))
    setenv(**{**BASE, "CF_WIDTH_POS": 0.05, "CF_WIDTH_NEG": 0.07, "CF_MASS_TARGET_POS": 0.1964, "CF_MASS_TARGET_NEG": 0.2003})
    a_nar = adv.clone(); d_nar = cf.apply_credit_filter(data, recs, a_nar, mask, ent)
    check("narrow (0.05/0.07) rescaled: measured masses hit the targets and the guard passes",
          abs(d_nar["cf/mass_pos"] - 0.1964) < 0.01 and abs(d_nar["cf/mass_neg"] - 0.2003) < 0.01
          and abs(d_nar["cf/mass_pos_expect"] - 0.1964) < 1e-3 and abs(d_nar["cf/mass_neg_expect"] - 0.2003) < 1e-3)
    check("narrow rescaled: scale > 1 and the top-rank weight exceeds that of the base window",
          d_nar["cf/mass_scale_pos_mean"] > 1.2 and d_nar["cf/mass_scale_neg_mean"] > 1.5 and float(a_nar.abs().max()) > float(a_base.abs().max()) * 1.2)
    setenv(**{**BASE, "CF_WIDTH_POS": 0.20, "CF_WIDTH_NEG": 0.28, "CF_MASS_TARGET_POS": 0.1964, "CF_MASS_TARGET_NEG": 0.2003})
    a_wid = adv.clone(); d_wid = cf.apply_credit_filter(data, recs, a_wid, mask, ent)
    check("wide (0.20/0.28) rescaled: masses hit the targets, scale < 1",
          abs(d_wid["cf/mass_pos"] - 0.1964) < 0.01 and abs(d_wid["cf/mass_neg"] - 0.2003) < 0.01
          and d_wid["cf/mass_scale_pos_mean"] < 0.8 and d_wid["cf/mass_scale_neg_mean"] < 0.7)

    def row_mass(a):
        return (a.abs() * mask.float()).sum(dim=1) / mask.float().sum(dim=1)
    rm_p, rm_n, rm_w = row_mass(a_base), row_mass(a_nar), row_mass(a_wid)
    check("per-row mean |adv| equal across the base / narrow / wide windows (max rel diff < 1.5%)",
          float(((rm_n - rm_p).abs() / rm_p).max()) < 0.015 and float(((rm_w - rm_p).abs() / rm_p).max()) < 0.015)
    setenv(**{**BASE, "CF_RANK_SCOPE": "batch", "CF_MASS_TARGET_POS": 0.1964})
    try:
        cf.apply_credit_filter(SimpleNamespace(batch={}), recs, adv.clone(), mask, ent)
        check("mass target + batch rank hard error", False)
    except RuntimeError as e:
        check("mass target + batch rank hard error", "CF_MASS_TARGET_POS" in str(e))
    # equalisation: both classes rescaled to the larger closed-form class mass
    setenv(**BASE)
    data, recs, adv, mask, ent = mkbatch(B=16, T=64, seed=3)
    d0 = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
    check("default: mass_equalize 0, no matched target", d0["cf/mass_equalize"] == 0 and d0["cf/mass_target_eff"] == -1.0)
    setenv(**{**BASE, "CF_WIDTH_POS": 0.05, "CF_WIDTH_NEG": 0.07, "CF_FLOOR_POS": 0.0, "CF_FLOOR_NEG": 0.0, "CF_MASS_EQUALIZE": 1})
    d_sc = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
    check("equalise: target = larger natural mass (neg), positive rows amplified (>1.2), masses equal",
          abs(d_sc["cf/mass_target_eff"] - d_sc["cf/mass_nat_neg"]) < 1e-9 and d_sc["cf/mass_scale_pos_mean"] > 1.2
          and abs(d_sc["cf/mass_scale_neg_mean"] - 1.0) < 1e-6 and abs(d_sc["cf/mass_pos_expect"] - d_sc["cf/mass_neg_expect"]) < 1e-3)
    setenv(**{**BASE, "CF_MASS_EQUALIZE": 1, "CF_MASS_TARGET_POS": 0.1964})
    try:
        cf.apply_credit_filter(SimpleNamespace(batch={}), recs, adv.clone(), mask, ent)
        check("equalise + numeric target hard error", False)
    except RuntimeError as e:
        check("equalise + numeric target hard error", "CF_MASS_EQUALIZE" in str(e))


def test_hard_kernel():
    setenv(**INERT, CF_KERNEL="hard", CF_HARD_TOP=0.2, CF_FLOOR_POS=0.0, CF_FLOOR_NEG=0.0)
    data, recs, adv, mask, ent = mkbatch(B=16, T=64, seed=4)
    a = adv.clone(); d = cf.apply_credit_filter(data, recs, a, mask, ent)
    w = torch.where(adv != 0, a / adv, torch.ones_like(adv))
    B, T = adv.shape
    H = torch.where(mask, ent, torch.full_like(ent, float("-inf")))
    order = torch.argsort(H, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order); ranks.scatter_(1, order, torch.arange(1, T + 1).unsqueeze(0).expand(B, -1).contiguous())
    q = ranks.float() / mask.sum(1, keepdim=True).float()
    exp = torch.where(mask, (q <= 0.2).float(), torch.ones_like(q))
    check("hard kernel: weight 1 on the top-20% entropy ranks, 0 elsewhere", torch.allclose(w, exp) and d["cf/kernel"] == "hard" and d["cf/hard_top"] == 0.2)
    check("hard kernel: closed-form class mass 0.2", abs(d["cf/mass_pos_expect"] - 0.2) < 1e-6)


def test_rank_key_controls():
    base = dict(CF_FALLBACK_H_LO=50.0, CF_FALLBACK_H_HI=90.0, CF_DAMPING_POS=0.0, CF_DAMPING_NEG=0.0,
                CF_WIDTH_POS=0.111, CF_WIDTH_NEG=0.155, CF_FLOOR_POS=0.1213, CF_FLOOR_NEG=0.0)
    setenv(**base)
    data, recs, adv, mask, ent = mkbatch(B=16, T=32, seed=31)
    a_e = adv.clone(); d_e = cf.apply_credit_filter(data, recs, a_e, mask, ent)
    setenv(**base, CF_RANK_KEY="random")
    torch.manual_seed(7)
    a_r = adv.clone(); d_r = cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a_r, mask, ent)
    w_e = torch.where(adv != 0, a_e / adv, torch.ones_like(adv)); w_r = torch.where(adv != 0, a_r / adv, torch.ones_like(adv))
    same_multiset = all(torch.allclose(torch.sort(w_e[i][mask[i]]).values, torch.sort(w_r[i][mask[i]]).values, atol=1e-6) for i in range(adv.shape[0]))
    check("random rank: per-row weight multiset = entropy rank (masses unchanged), assignment differs",
          same_multiset and not torch.allclose(w_e, w_r, atol=1e-6) and d_r["cf/rank_key"] == "random"
          and abs(d_r["cf/mass_pos"] - d_e["cf/mass_pos"]) < 1e-6 and abs(d_r["cf/mass_neg"] - d_e["cf/mass_neg"]) < 1e-6)
    setenv(**base, CF_RANK_KEY="reversed")
    a_b = adv.clone(); d_b = cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a_b, mask, ent)
    w_b = torch.where(adv != 0, a_b / adv, torch.ones_like(adv))
    rev_ok = True
    for i in range(adv.shape[0]):
        if adv[i].abs().sum() == 0:
            continue
        v = mask[i]; e = ent[i][v]; we = w_e[i][v]; wb = w_b[i][v]
        rev_ok = rev_ok and bool(we[e.argmax()] >= we.max() - 1e-6) and bool(wb[e.argmax()] <= wb.min() + 1e-6) and bool(wb[e.argmin()] >= wb.max() - 1e-6)
    check("reversed rank: top-entropy token -> smallest weight, bottom -> largest; multiset unchanged",
          rev_ok and d_b["cf/rank_key"] == "reversed"
          and all(torch.allclose(torch.sort(w_e[i][mask[i]]).values, torch.sort(w_b[i][mask[i]]).values, atol=1e-6) for i in range(adv.shape[0])))
    try:
        setenv(**base, CF_RANK_KEY="shuffle"); cf.apply_credit_filter(SimpleNamespace(batch={}), recs, adv.clone(), mask, ent); bad1 = False
    except RuntimeError:
        bad1 = True
    try:
        setenv(**base, CF_RANK_KEY="random", CF_WIDTH_MODE="tracked"); cf.apply_credit_filter(SimpleNamespace(batch={}), recs, adv.clone(), mask, ent); bad2 = False
    except RuntimeError:
        bad2 = True
    check("rank-key guards: unknown key refused; random key refused with tracked widths", bad1 and bad2)
    cf.reset_tracker()


def test_mass_guard_short_rows():
    # 1-token rows sit at the floor (q = 1); CF_MASS_GUARD_MINLEN leaves them out of the guard's comparison
    setenv(**BASE, CF_MASS_GUARD_MINLEN=16, CF_MASS_GUARD_TOL=0.06)
    data, recs, adv, mask, ent = mkbatch(B=16, T=64, seed=9)
    mask[3, 1:] = False; mask[7, 2:] = False
    adv = torch.where(mask, adv, torch.zeros_like(adv))
    d = cf.apply_credit_filter(data, recs, adv.clone(), mask, ent)
    check("guard: short rows counted and excluded", d["cf/guard_pos_rows_short"] == 2 and d["cf/guard_pos_dev"] <= 0.06)


def test_outcome_and_advantage_sign_disagree():
    """A row's class is read from two places: the window WIDTH follows the verifier outcome (answer_correct), while the
    FLOOR, the DAMPING exponent and the mass target follow the sign of the row's advantage.  The two disagree when the
    overlong penalty moves a response across the mean reward of its group: row 2 is a penalised correct response below
    its group's mean (negative advantage), row 5 an unpenalised wrong response above the mean of a group whose other
    wrong responses carry the penalty (positive advantage)."""
    WP, WN, FP, FN, DP, DN = 0.10, 0.30, 0.20, 0.05, 1.0, 2.0
    knobs = dict(CF_FALLBACK_H_LO=5.0, CF_FALLBACK_H_HI=9.0, CF_WIDTH_POS=WP, CF_WIDTH_NEG=WN, CF_FLOOR_POS=FP, CF_FLOOR_NEG=FN,
                 CF_DAMPING_POS=DP, CF_DAMPING_NEG=DN)                       # fallback inert, every class knob distinct
    B, T = 8, 256
    ent = torch.rand(B, T, generator=torch.Generator().manual_seed(17)) * 2.0
    mask = torch.ones(B, T, dtype=torch.bool)
    # terminal rewards: +1 / -1 from the grader minus an overlong penalty in [0, 1]; two prompt groups of four responses
    correct = [True, True, True, False, True, False, False, False]
    reward = [1.0, 1.0, 0.0, -1.0, 0.0, -1.0, -2.0, -2.0]
    uid = ["u0"] * 4 + ["u1"] * 4
    r = torch.tensor(reward).view(2, 4)
    mu = r.mean(dim=1, keepdim=True)
    sd = ((r - mu) ** 2).mean(dim=1, keepdim=True).sqrt()
    adv_row = ((r - mu) / (sd + 1e-6)).reshape(B)                            # the group standardisation of the terminal reward
    sign = torch.sign(adv_row)
    check("rows 2 and 5 are the ones whose outcome and advantage sign disagree",
          sign.tolist() == [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0] and correct[2] and not correct[5])
    adv = adv_row.unsqueeze(1).expand(B, T).contiguous()
    recs = [SimpleNamespace(prompt_uid=u, answer_correct=c) for u, c in zip(uid, correct)]
    setenv(**knobs)
    a = adv.clone()
    d = cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a, mask, ent)
    coef = a / adv
    # damping g = (1 - p_g)^exponent: p_g is the accuracy of the row's group (3/4 in u0, 1/4 in u1), the exponent follows the sign
    one_m_p = torch.tensor([0.25] * 4 + [0.75] * 4)
    g_by_sign = torch.where(sign > 0, one_m_p ** DP, one_m_p ** DN)
    w_ref = window_reference(adv, mask, ent, recs, width_pos=WP, width_neg=WN, floor_pos=FP, floor_neg=FN)   # width by outcome, floor by sign
    check("coefficient = (1 - p_g)^damping(sign) * max(2^(-q / width(outcome)), floor(sign)) on every row",
          torch.allclose(coef, g_by_sign.unsqueeze(1) * w_ref, atol=1e-6))
    win = coef / g_by_sign.unsqueeze(1)
    top_pos, top_neg = 2.0 ** (-(1.0 / T) / WP), 2.0 ** (-(1.0 / T) / WN)    # the kernel at the top-entropy token, q = 1/T
    check("correct response with a negative advantage: width CF_WIDTH_POS, floor CF_FLOOR_NEG, damping CF_DAMPING_NEG",
          abs(float(win[2].max()) - top_pos) < 1e-5 and abs(float(win[2].min()) - FN) < 1e-6
          and abs(float(coef[2].max()) - 0.25 ** DN * top_pos) < 1e-5)
    check("wrong response with a positive advantage: width CF_WIDTH_NEG, floor CF_FLOOR_POS, damping CF_DAMPING_POS",
          abs(float(win[5].max()) - top_neg) < 1e-5 and abs(float(win[5].min()) - FP) < 1e-6
          and abs(float(coef[5].max()) - 0.75 ** DP * top_neg) < 1e-5)
    # control: the swapped reading (width by sign, floor by outcome) gives another window on the two disagreeing rows only
    by_sign = [SimpleNamespace(prompt_uid=u, answer_correct=bool(s > 0)) for u, s in zip(uid, sign)]
    outcome_sign = torch.tensor([1.0 if c else -1.0 for c in correct]).unsqueeze(1).expand(B, T)
    w_swapped = window_reference(outcome_sign, mask, ent, by_sign, width_pos=WP, width_neg=WN, floor_pos=FP, floor_neg=FN)
    check("the swapped reading (width by sign, floor by outcome) differs on exactly the two disagreeing rows",
          all(torch.allclose(win[i], w_swapped[i], atol=1e-6) == (i not in (2, 5)) for i in range(B)))
    check("telemetry counts the classes by advantage sign and reports the damping by sign",
          d["cf/n_pos"] == 4 and d["cf/n_neg"] == 4 and abs(d["cf/g_pos_mean"] - 0.5) < 1e-4 and abs(d["cf/g_neg_min"] - 0.25 ** DN) < 1e-4)
    MP, MN = 0.30, 0.20
    setenv(**knobs, CF_MASS_TARGET_POS=MP, CF_MASS_TARGET_NEG=MN)
    a2 = adv.clone()
    cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a2, mask, ent)
    mass = ((a2 / adv) / g_by_sign.unsqueeze(1)).mean(dim=1)                 # per-row mean of the rescaled window
    check("the mass target follows the sign as well (row 2 -> CF_MASS_TARGET_NEG, row 5 -> CF_MASS_TARGET_POS)",
          all(abs(float(mass[i]) - (MP if sign[i] > 0 else MN)) < 0.01 for i in range(B)))


def test_top_rank_gain():
    """CF_TOP_GAIN_Q / CF_TOP_GAIN: the exp kernel is multiplied by the gain on the ranks q <= Q, before the floor; off by default."""
    B, T = 4, 800
    g = torch.Generator().manual_seed(11)
    ent = torch.rand(B, T, generator=g) * 2.0
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 400:] = False            # 400 tokens: exactly one position with q <= 0.0025
    mask[3, 200:] = False            # 200 tokens: no position in the top 0.25%
    sign = torch.tensor([1.0, 1.0, -1.0, -1.0])
    adv = torch.where(mask, sign.unsqueeze(1) * torch.ones(B, T) * 0.5, torch.zeros(B, T))
    recs = [SimpleNamespace(prompt_uid=f"u{i % 2}", answer_correct=bool(sign[i] > 0)) for i in range(B)]
    w_ref = window_reference(adv, mask, ent, recs)
    H = torch.where(mask, ent, torch.full_like(ent, float("-inf")))
    order = torch.argsort(H, dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, T + 1).unsqueeze(0).expand(B, -1).contiguous())
    q = ranks.float() / mask.sum(1, keepdim=True).float()
    top = (q <= 0.0025) & mask

    setenv(**INERT)
    a0 = adv.clone()
    d0 = cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a0, mask, ent)
    check("top-rank gain off by default: the plain window, telemetry 0 / 1", torch.allclose(a0, adv * w_ref, atol=1e-6)
          and d0["cf/top_gain_q"] == 0.0 and d0["cf/top_gain"] == 1.0)

    setenv(**INERT, CF_TOP_GAIN_Q=0.0025, CF_TOP_GAIN=3.0)
    a1 = adv.clone()
    d1 = cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a1, mask, ent)
    widths = torch.tensor([0.10, 0.10, 0.14, 0.14]).unsqueeze(1)
    floor = torch.tensor([0.1, 0.1, 0.0, 0.0]).unsqueeze(1)
    kern = torch.pow(2.0, -q / widths)
    w_gain = torch.where(mask, torch.maximum(torch.where(top, kern * 3.0, kern), floor), torch.ones_like(kern))
    check("top-rank gain: coefficient == max(gain * 2^(-q/width) on q <= Q, floor) elementwise", torch.allclose(a1, adv * w_gain, atol=1e-6))
    check("positions in the top 0.25%: 2 / 1 / 2 / 0 for rows of 800 / 400 / 800 / 200 tokens", top.sum(1).tolist() == [2, 1, 2, 0])
    check("a row without a position in the top 0.25% keeps the plain window", torch.allclose(a1[3], a0[3], atol=1e-7))
    check("telemetry reports the knobs", d1["cf/top_gain_q"] == 0.0025 and d1["cf/top_gain"] == 3.0)

    # with class-mass targets the closed-form row mass includes the gain: every long row lands on its target
    setenv(**INERT, CF_TOP_GAIN_Q=0.0025, CF_TOP_GAIN=3.0, CF_MASS_TARGET_POS=0.1964, CF_MASS_TARGET_NEG=0.2003)
    a2 = adv.clone()
    d2 = cf.apply_credit_filter(SimpleNamespace(batch={}), recs, a2, mask, ent)
    row_mass = ((a2 / 0.5).abs() * mask.float()).sum(1) / mask.sum(1).float()
    check("with mass targets the rows of 800 tokens carry the class targets 0.1964 / 0.2003",
          abs(float(row_mass[0]) - 0.1964) < 2e-3 and abs(float(row_mass[2]) - 0.2003) < 2e-3)
    check("the rescale factor is below one (the gain adds mass)", d2["cf/mass_scale_pos_mean"] < 1.0 and d2["cf/mass_scale_neg_mean"] < 1.0)

    setenv(**INERT, CF_KERNEL="hard", CF_FLOOR_POS=0, CF_FLOOR_NEG=0, CF_TOP_GAIN_Q=0.0025, CF_TOP_GAIN=3.0)
    try:
        cf.apply_credit_filter(SimpleNamespace(batch={}), recs, adv.clone(), mask, ent)
        check("the gain with the hard window is refused", False)
    except RuntimeError as e:
        check("the gain with the hard window is refused", "CF_KERNEL=exp" in str(e))


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    setenv()
    print(f"ALL {len(PASSED)} CHECKS PASSED")
    for n in PASSED:
        print("  ok", n)


if __name__ == "__main__":
    main()
