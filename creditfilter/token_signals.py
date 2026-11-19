"""Teacher KL of SC-GRPO (arXiv 2606.18810), computed on the actor workers before the advantage step.

The teacher is the policy itself with a correct rollout of the same prompt group placed in its context as a
demonstration; the per-token signal is the forward KL teacher || student over the full vocabulary at every response
position.  Groups with CF_SCGRPO_NCMIN <= n_correct <= G - 1 are weighted, the others keep uniform weights.
"""
import os

import numpy as np
import torch

from verl.protocol import DataProto


def _envi(name, default):
    v = os.environ.get(name, "").strip()
    return int(v) if v else int(default)


_GUIDE = (
    "You are given a reference solution that correctly solves the problem below. "
    "Use it only as a high-level guide for your own reasoning about the same "
    "problem; do not copy it verbatim.\n\n### Reference Correct Solution\n"
)


def prepare_signal_request(batch, tokenizer, global_steps=0):
    """Driver side: the request for the actor workers, or None when no row qualifies."""
    resp = batch.batch["responses"]
    T = resp.shape[1]
    att = batch.batch["attention_mask"]
    rmask_full = batch.batch["response_mask"] if "response_mask" in batch.batch else att[:, -T:]
    prompts = batch.batch["input_ids"][:, :-T]
    p_att = att[:, :-T]
    uid = batch.non_tensor_batch["uid"]
    # a rollout counts as correct when its reward is positive, as in dynamic sampling
    scores = batch.batch["token_level_scores"].sum(dim=-1)
    correct = (scores > 0).tolist()
    groups = {}
    for i, u in enumerate(uid):
        groups.setdefault(u, []).append(i)
    nc_min = _envi("CF_SCGRPO_NCMIN", 2)
    demo_cap = _envi("CF_SCGRPO_DEMO_MAX_TOK", 8192)
    kl_resp_cap = _envi("CF_SCGRPO_KL_MAX_RESP", 12288)  # longer responses keep uniform weights (memory of the full-vocabulary logits)
    rng = np.random.default_rng(int(global_steps) + 17)
    guide_ids = tokenizer("<|im_start|>system\n" + _GUIDE, add_special_tokens=False)["input_ids"]
    guide_tail = tokenizer("\n<|im_end|>\n", add_special_tokens=False)["input_ids"]
    rows, teacher_lists, resp_lens, corr_flags = [], [], [], []
    n_weighted = n_fallback = n_demo_overflow = 0
    for u, idxs in groups.items():
        G = len(idxs)
        corr_idx = [j for j in idxs if correct[j]]
        n_c = len(corr_idx)
        if not (nc_min <= n_c <= G - 1):
            n_fallback += G
            continue
        # the demonstration is drawn uniformly from the correct rollouts of at most CF_SCGRPO_DEMO_MAX_TOK tokens;
        # a row without such a demonstration keeps uniform weights
        demo_pool = {}
        for j in corr_idx:
            Lr = int(rmask_full[j].sum().item())
            if Lr <= demo_cap:
                demo_pool[j] = Lr
        for i in idxs:
            cands = [j for j in demo_pool if j != i]
            if not cands:
                n_demo_overflow += 1
                n_fallback += 1
                continue
            demo_j = int(rng.choice(cands))
            demo_txt = tokenizer.decode(resp[demo_j, : demo_pool[demo_j]].tolist(), skip_special_tokens=True)
            demo_ids = tokenizer(demo_txt, add_special_tokens=False)["input_ids"]
            p_valid = prompts[i][p_att[i].bool()].tolist()
            r_len = int(rmask_full[i].sum().item())
            if r_len == 0 or r_len > kl_resp_cap:
                n_fallback += 1
                continue
            teacher = guide_ids + demo_ids + guide_tail + p_valid + resp[i, :r_len].tolist()
            rows.append(i)
            teacher_lists.append(teacher)
            resp_lens.append(r_len)
            corr_flags.append(bool(correct[i]))
            n_weighted += 1
    if not rows:
        return None
    Lmax = max(len(t) for t in teacher_lists)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    t_ids = torch.full((len(rows), Lmax), pad_id, dtype=torch.long)
    t_att = torch.zeros((len(rows), Lmax), dtype=torch.long)
    for k, t in enumerate(teacher_lists):
        t_ids[k, Lmax - len(t):] = torch.tensor(t, dtype=torch.long)  # left pad
        t_att[k, Lmax - len(t):] = 1
    sub = batch.select_idxs(rows)
    req = DataProto.from_dict(
        tensors={
            "input_ids": sub.batch["input_ids"],
            "attention_mask": sub.batch["attention_mask"],
            "position_ids": sub.batch["position_ids"],
            "responses": sub.batch["responses"],
            "sc_teacher_ids": t_ids,
            "sc_teacher_att": t_att,
            "sc_resp_len": torch.tensor(resp_lens, dtype=torch.long),
        },
        meta_info={
            "sc_rows": rows,
            "sc_corr_flags": corr_flags,
            "sc_stats": {"n_weighted": n_weighted, "n_fallback": n_fallback, "n_demo_overflow": n_demo_overflow},
        },
    )
    return req


def merge_signal_output(batch, req, out):
    """Write the worker output into batch.batch for the route (NaN on rows without a teacher KL); returns the statistics."""
    B, T = batch.batch["responses"].shape
    D = torch.full((B, T), float("nan"))
    corr = torch.full((B,), -1.0)
    for k, i in enumerate(req.meta_info["sc_rows"]):
        D[i] = out.batch["sc_kl"][k].float()
        corr[i] = 1.0 if req.meta_info["sc_corr_flags"][k] else 0.0
    batch.batch["sc_kl"] = D
    batch.batch["sc_row_correct"] = corr
    return dict(req.meta_info["sc_stats"])


def _logits_suffix(model, input_ids, attention_mask, position_ids, keep):
    """Logits of the last `keep` positions."""
    kw = dict(input_ids=input_ids, attention_mask=attention_mask,
              position_ids=position_ids, use_cache=False)
    try:
        return model(**kw, logits_to_keep=keep).logits[:, -keep:]
    except TypeError:
        pass
    try:
        return model(**kw, num_logits_to_keep=keep).logits[:, -keep:]
    except TypeError:
        return model(**kw).logits[:, -keep:]


def compute_token_signals(model, data: DataProto) -> DataProto:
    """Worker side, called on every actor rank with its shard of the request: the teacher KL of every response token."""
    device = torch.cuda.current_device()
    ids = data.batch["input_ids"]
    att = data.batch["attention_mask"]
    pos = data.batch["position_ids"]
    t_ids = data.batch["sc_teacher_ids"]
    t_att = data.batch["sc_teacher_att"]
    rlen = data.batch["sc_resp_len"]
    B, L = ids.shape
    T = data.batch["responses"].shape[1]
    P = L - T
    chunk = 512
    out = torch.zeros((B, T), dtype=torch.float32)
    with torch.no_grad():
        for i in range(B):  # one row at a time: two full-vocabulary logit tensors of one response in memory
            Ti = int(rlen[i].item())
            if Ti == 0:
                continue
            # teacher context: left-padded, the response is its suffix
            tv = int(t_att[i].sum().item())
            ti = t_ids[i:i + 1, -tv:].to(device)
            ta = t_att[i:i + 1, -tv:].to(device)
            tp = torch.clamp(torch.cumsum(ta, dim=-1) - 1, min=0)
            lg_t = _logits_suffix(model, ti, ta, tp, Ti + 1)[0, :-1]  # [Ti,V]
            # student context: the original prompt and response
            L_eff = P + Ti
            si = ids[i:i + 1, :L_eff].to(device)
            sa = att[i:i + 1, :L_eff].to(device)
            sp = pos[i:i + 1, :L_eff].to(device)
            lg_s = _logits_suffix(model, si, sa, sp, Ti + 1)[0, :-1]  # [Ti,V]
            for lo in range(0, Ti, chunk):
                hi = min(Ti, lo + chunk)
                lp_t = torch.log_softmax(lg_t[lo:hi].float(), dim=-1)
                lp_s = torch.log_softmax(lg_s[lo:hi].float(), dim=-1)
                out[i, lo:hi] = torch.sum(lp_t.exp() * (lp_t - lp_s), dim=-1).cpu()
            del lg_t, lg_s
            torch.cuda.empty_cache()
    return DataProto.from_dict(tensors={"sc_kl": out})
