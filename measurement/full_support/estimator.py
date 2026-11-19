"""The full-support estimator of the forking value and the rank strata of the instrument.

At a response position with next-token law pi and branch values Q(x) = P(correct | prefix, x) the forking value is
    nu = Var_pi(Q) = 1/2 sum_{x,y} pi_x pi_y (Q_x - Q_y)^2 = C * kappa,   C = 1 - sum_x pi_x^2,
with kappa = 1/2 E[(Q_X - Q_Y)^2 | X != Y] the fork consequence.  The branches are the head (the most probable tokens, in
descending order until their mass reaches COVER, at least KMIN and at most KMAX of them) with their probabilities p_k, and two
independent draws y1, y2 from the remaining law, whose mass is r.  Every branch has m continuations, S of them correct.  With
U_a = S_a (S_a - 1) / (m_a (m_a - 1)), unbiased for Q_a^2, and D_ab = U_a + U_b - 2 Qhat_a Qhat_b, unbiased for (Q_a - Q_b)^2,
    nu_hat = sum_{a<b in head} p_a p_b D_ab + r sum_{k in head} p_k (D_{k,y1} + D_{k,y2}) / 2 + 1/2 r^2 D_{y1,y2} [y1 != y2]
is unbiased for nu under the full law.  Without tail draws (r below RMIN) the tail terms are dropped and |bias| <= r.
"""
import bisect
import hashlib

FORMAT_TOKENS = {151643, 151645, 151667, 151668}     # Qwen3 special tokens, never ranked or sampled
TAIL_EXCLUDED = 5                                    # the last five response tokens are never ranked or sampled
RANK_EDGES = [0, .0025, .005, .01, .015, .025, .0375, .05, .075, .1, .15, .2, .3, .4, .6, .8, 1.0000001]
ENTROPY_QUANTILES = [0.001, 0.0025, 0.005, 0.01, 0.025]   # top strata of the token entropy over the whole corpus


def rank_stratum(q, edges=RANK_EDGES):
    return bisect.bisect_right(edges, q) - 1


def entropy_stratum(H, thresholds):
    """thresholds: the entropies at ENTROPY_QUANTILES, descending; returns len(thresholds) below all of them."""
    for k, thr in enumerate(thresholds):
        if H >= thr:
            return k
    return len(thresholds)


def seed_of(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def pair_D(Sa, ma, Sb, mb):
    """Unbiased for (Q_a - Q_b)^2 from independent continuations of the two branches."""
    Ua = Sa * (Sa - 1) / (ma * (ma - 1))
    Ub = Sb * (Sb - 1) / (mb * (mb - 1))
    return Ua + Ub - 2.0 * (Sa / ma) * (Sb / mb)


def nu_hat(head, tail, r):
    """head: [(p, S, m)]; tail: [] or [(token, S, m), (token, S, m)]; r: tail mass.
    Returns (nu, nu_head, C_head): the full-law estimate, and the estimate and fork probability under the head-normalised law."""
    nu = 0.0
    for a in range(len(head)):
        for b in range(a + 1, len(head)):
            nu += head[a][0] * head[b][0] * pair_D(head[a][1], head[a][2], head[b][1], head[b][2])
    if len(tail) == 2 and r > 0:
        (ya, Sa, ma), (yb, Sb, mb) = tail
        for p, S, m in head:
            nu += r * p * 0.5 * (pair_D(S, m, Sa, ma) + pair_D(S, m, Sb, mb))
        if ya != yb:
            nu += 0.5 * r * r * pair_D(Sa, ma, Sb, mb)
    ph = sum(p for p, S, m in head)
    nuh = 0.0
    for a in range(len(head)):
        for b in range(a + 1, len(head)):
            nuh += (head[a][0] / ph) * (head[b][0] / ph) * pair_D(head[a][1], head[a][2], head[b][1], head[b][2])
    Ch = 1.0 - sum((p / ph) ** 2 for p, S, m in head)
    return nu, nuh, Ch


def record_nu(rec):
    """nu_hat of a position record: rec['br'] = [{'t': token, 'p': probability, 'k': 'h' or 't', 'S', 'm'}, ...], rec['r']."""
    head = [(b["p"], b["S"], b["m"]) for b in rec["br"] if b["k"] == "h"]
    tail = [(b["t"], b["S"], b["m"]) for b in rec["br"] if b["k"] == "t"]
    return nu_hat(head, tail, rec["r"])
