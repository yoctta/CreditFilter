"""CPU unit tests of STARE (baselines.apply_stare through routing.apply_token_weights)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASSED = []


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL: {name} {detail}".rstrip())
    PASSED.append(name)
    print(f"[PASS] {name} {detail}", flush=True)


class _Batch(dict):
    def keys(self):
        return set(super().keys())


class _Data:
    def __init__(self, batch, uids=None):
        self.batch = batch
        self.non_tensor_batch = {} if uids is None else {"uid": uids}


class _Rec:
    def __init__(self, correct):
        self.answer_correct = correct


def _env(mode, **kv):
    for k in list(os.environ):
        if k.startswith("CF_"):
            del os.environ[k]
    os.environ["CF_ROUTE"] = mode
    for k, v in kv.items():
        os.environ[k] = str(v)


def test_stare():
    from creditfilter import routing as rt
    torch.manual_seed(3)
    B, T = 16, 50
    g = torch.Generator().manual_seed(7)
    ent = torch.rand(B, T, generator=g) * 0.2          # batch mean entropy ~0.10
    olp = -(0.1 + torch.rand(B, T, generator=g) * 3.0)  # surprisal in (0.1, 3.1), almost surely pairwise distinct
    mask = torch.ones(B, T, dtype=torch.bool)
    sign = torch.tensor([1.0] * 8 + [-1.0] * 6 + [0.0] * 2)
    adv0 = sign.unsqueeze(1).expand(B, T).contiguous()
    recs = [_Rec(s > 0) for s in sign]

    # CF_STARE_H_TARGET not set: refused
    _env("stare", CF_STARE_W=1.1)
    try:
        rt.apply_token_weights(_Data(_Batch(advantages=adv0.clone(), response_mask=mask.float(), entropys=ent, old_log_probs=olp)), recs)
        check("stare/H_TARGET unset hard error", False, "no error raised")
    except RuntimeError as e:
        check("stare/H_TARGET unset hard error", "H_TARGET" in str(e), "")

    # gate closed (batch-mean entropy 0.10 >= H_TARGET 0.05): advantages unchanged
    _env("stare", CF_STARE_W=1.1, CF_STARE_H_TARGET=0.05)
    adv = adv0.clone()
    d = rt.apply_token_weights(_Data(_Batch(advantages=adv, response_mask=mask.float(), entropys=ent, old_log_probs=olp)), recs)
    check("stare/gate closed leaves advantages unchanged", d["stare/gate_on"] == 0 and torch.equal(adv, adv0), "")

    # gate open: the top 10% of surprisal on the positive side is multiplied by 1.1, the rest is unchanged
    _env("stare", CF_STARE_W=1.1, CF_STARE_M=1.0, CF_STARE_P=0.10, CF_STARE_H_TARGET=0.12)
    adv = adv0.clone()
    d = rt.apply_token_weights(_Data(_Batch(advantages=adv, response_mask=mask.float(), entropys=ent, old_log_probs=olp)), recs)
    s = -olp
    pos_tok = (sign > 0).unsqueeze(1) & mask
    n_pos = int(pos_tok.sum())
    amp = adv != adv0          # modified positions (zero-advantage rows have adv==adv0==0, never mis-flagged)
    check("stare/gate open marker", d["stare/gate_on"] == 1, f"batch_H={d['stare/batch_H']}")
    check("stare/amplification only on positive side", bool((amp & ~pos_tok).sum() == 0), f"out_of_bounds={int((amp & ~pos_tok).sum())}")
    n_sel = int(amp.sum())
    check("stare/selected count ~ P*N", abs(n_sel - 0.10 * n_pos) <= 2, f"{n_sel} vs {0.10 * n_pos:.0f}")
    smin_sel = float(s[amp].min()) if n_sel else -1
    smax_un = float(s[pos_tok & ~amp].max())
    check("stare/selected = highest-surprisal segment on positive side", smin_sel >= smax_un - 1e-6,
          f"min_sel={smin_sel:.3f} max_unsel={smax_un:.3f}")
    check("stare/amplification factor=W", bool(torch.allclose(adv[amp] / adv0[amp], torch.full((n_sel,), 1.1), atol=1e-6)), "")
    check("stare/negative/zero side elementwise unchanged", bool(torch.equal(adv[sign <= 0], adv0[sign <= 0])), "")
    check("stare/mass telemetry ~ 1+P(W-1)", abs(d["stare/mass_pos"] - 1.01) < 0.005, f"{d['stare/mass_pos']}")

    # with M = 0.9 the top 10% of surprisal on the negative side is multiplied by 0.9
    _env("stare", CF_STARE_W=1.1, CF_STARE_M=0.9, CF_STARE_P=0.10, CF_STARE_H_TARGET=0.12)
    adv = adv0.clone()
    d = rt.apply_token_weights(_Data(_Batch(advantages=adv, response_mask=mask.float(), entropys=ent, old_log_probs=olp)), recs)
    neg_tok = (sign < 0).unsqueeze(1) & mask
    att = ((adv / adv0.where(adv0 != 0, torch.ones_like(adv0))) - 0.9).abs() < 1e-6
    att = att & neg_tok
    check("stare/negative-side attenuated count ~ P*N", abs(int(att.sum()) - 0.10 * int(neg_tok.sum())) <= 2, f"{int(att.sum())}")
    check("stare/negative-side mass telemetry < 1", d["stare/mass_neg"] < 1.0, f"{d['stare/mass_neg']}")


if __name__ == "__main__":
    test_stare()
    print(f"ALL {len(PASSED)} CHECKS PASSED")
