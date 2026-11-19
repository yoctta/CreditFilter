"""CPU unit tests of the 80/20 rule and of SC-GRPO (creditfilter/baselines.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from creditfilter.baselines import apply_entropy_mask, apply_scgrpo  # noqa: E402


class _FakeData:
    def __init__(self, tensors, meta=None):
        self.batch = tensors
        self.meta_info = meta or {}


def test_entropy_mask():
    torch.manual_seed(0)
    B, T = 6, 40
    ent = torch.rand(B, T)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 30:] = False
    os.environ["CF_MASK_TOP"] = "0.2"
    for scope in ("response", "batch"):
        os.environ["CF_MASK_SCOPE"] = scope
        adv = torch.ones(B, T)
        d = apply_entropy_mask(None, None, adv, mask, ent)
        kept = (adv == 1.0) & mask
        assert torch.equal(adv[~mask], torch.ones(B, T)[~mask]), "padding changed"
        assert bool(((adv == 0.0) | (adv == 1.0)).all())
        if scope == "response":
            # 20% of the 30 response tokens of every row, and they are the tokens of highest entropy
            assert kept.sum(dim=1).tolist() == [6] * B
            for b in range(B):
                assert ent[b, :30][kept[b, :30]].min() > ent[b, :30][~kept[b, :30]].max()
        else:
            assert int(kept.sum()) == 36
            assert ent[mask & kept].min() > ent[mask & ~kept].max()
        assert abs(d["route/kept_frac"] - 0.2) < 1e-6 and d["route/mask_scope"] == scope
    print("[entropy_mask] OK")


def test_scgrpo():
    torch.manual_seed(1)
    B, T = 8, 30
    adv = torch.randn(B, T)
    adv0 = adv.clone()
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 25:] = False
    D = torch.rand(B, T) * 0.2
    D[4:, :] = float("nan")  # fallback rows
    data = _FakeData({"sc_kl": D, "sc_row_correct": torch.tensor([1., 0., 1., 0., -1., -1., -1., -1.])},
                     {"sc_stats": {"n_weighted": 4}})
    os.environ["CF_SCGRPO_PCT"] = "75"
    os.environ["CF_SCGRPO_CMIN"] = "1e-4"
    d = apply_scgrpo(data, None, adv, mask, None)
    # fallback rows exactly untouched
    assert torch.allclose(adv[4:], adv0[4:]), "fallback rows modified"
    # weighted rows: adv = adv0 * D/(D+c), verify against the reported c
    c = d["sc/c"]
    f_ref = D[:4, :25] / (D[:4, :25] + c)
    assert torch.allclose(adv[:4, :25], adv0[:4, :25] * f_ref, atol=1e-4), "f mismatch"
    # c is the 75th percentile of active KL (>= cmin)
    vals = D[:4, :25].flatten()
    k = int(torch.tensor(0.75 * vals.numel()).ceil())
    assert abs(c - float(vals.kthvalue(k).values)) < 1e-6
    # weights in (0,1); the fraction above 1/2 must be < 0.5
    assert 0 < d["sc/w_mean"] < 1 and d["sc/frac_w_gt_half"] <= 0.5
    assert bool(((adv * adv0) >= 0).all()), "sign flip"
    print("[scgrpo] OK", {k2: d[k2] for k2 in ("sc/c", "sc/w_mean", "sc/frac_w_gt_half", "sc/coverage")})


def test_scgrpo_all_fallback():
    B, T = 3, 10
    adv = torch.randn(B, T)
    adv0 = adv.clone()
    data = _FakeData({"sc_kl": torch.full((B, T), float("nan"))})
    d = apply_scgrpo(data, None, adv, torch.ones(B, T, dtype=torch.bool), None)
    assert torch.allclose(adv, adv0) and d["sc/coverage"] == 0.0
    print("[scgrpo all-fallback] OK")


if __name__ == "__main__":
    test_entropy_mask()
    test_scgrpo()
    test_scgrpo_all_fallback()
    print("ALL PASS")
