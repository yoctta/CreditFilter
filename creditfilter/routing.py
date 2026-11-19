"""CF_ROUTE selects the token weighting applied to the advantages:
off (uniform weights) | entropy_mask | creditfilter | stare | scgrpo."""
import os

ROUTES = ("off", "entropy_mask", "creditfilter", "stare", "scgrpo")
SIGNAL_ROUTES = ("scgrpo",)
# batch keys written by the token-signal stage; the trainer drops them after the advantage step
SIGNAL_BATCH_KEYS = ("sc_kl", "sc_row_correct")


def route_mode() -> str:
    m = os.environ.get("CF_ROUTE", "off").strip().lower()
    if m not in ROUTES:
        raise RuntimeError(f"[routing] CF_ROUTE={m!r} not in {ROUTES}")
    return m


def needs_entropy() -> bool:
    """The routes read the per-token entropies of the old-log-prob pass."""
    return route_mode() != "off"


def needs_sum_pi2() -> bool:
    """The tracked widths of CreditFilter-D need the per-token sum(pi^2) from the actor."""
    return route_mode() == "creditfilter" and os.environ.get("CF_WIDTH_MODE", "fixed").strip().lower() == "tracked"


def apply_token_weights(data, records) -> dict:
    """Multiply data.batch['advantages'] in place by the token weights of the route. Returns its telemetry dict."""
    mode = route_mode()
    if mode == "off":
        return {}
    if "entropys" not in data.batch.keys():
        raise RuntimeError(f"[routing] CF_ROUTE={mode} needs the per-token entropies ('entropys') in the batch")
    adv = data.batch["advantages"]
    mask = data.batch["response_mask"].to(bool)
    ent = data.batch["entropys"].to(adv.device)
    if ent.shape != adv.shape:
        raise RuntimeError(f"[routing] entropies {tuple(ent.shape)} != advantages {tuple(adv.shape)}")
    if mode == "creditfilter":
        from .credit_filter import apply_credit_filter
        return apply_credit_filter(data, records, adv, mask, ent)
    from . import baselines
    fn = {"entropy_mask": baselines.apply_entropy_mask, "stare": baselines.apply_stare, "scgrpo": baselines.apply_scgrpo}[mode]
    return fn(data, records, adv, mask, ent)


def attach_token_signals(batch, tokenizer, worker_group, global_steps):
    """SC-GRPO: build the teacher-KL request, run it on the actor workers and write the per-token result into the batch.
    Returns the stage statistics, or None when the route needs no such stage or no row qualifies."""
    mode = route_mode()
    if mode not in SIGNAL_ROUTES:
        return None
    from verl import DataProto

    from .token_signals import merge_signal_output, prepare_signal_request

    req = prepare_signal_request(batch, tokenizer, global_steps)
    if req is None:
        print(f"[token-signal] mode={mode}: no row qualifies", flush=True)
        return None
    ws = worker_group.world_size
    n = len(req)
    pad = (-n) % ws
    req_p = DataProto.concat([req, req.select_idxs([0] * pad)]) if pad else req
    out = worker_group.compute_token_signals(req_p)
    if pad:
        out = out.select_idxs(list(range(n)))
    stats = merge_signal_output(batch, req, out)
    print(f"[token-signal] mode={mode} rows={n} pad={pad} {stats}", flush=True)
    return stats
