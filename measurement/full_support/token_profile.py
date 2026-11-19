"""Teacher-forced profile of the policy with HF transformers.

  python3 token_profile.py light <corpus.jsonl> <model> <out.jsonl>
      every response token: entropy H, fork probability C = 1 - sum pi^2, probability of the sampled token; grade of the response
  python3 token_profile.py law <tasks.jsonl> <model> <out.jsonl> COVER KMIN KMAX RMIN
      the selected positions: the head branches (most probable tokens until their mass reaches COVER, between KMIN and KMAX of
      them) and two draws from the remaining law when its mass is at least RMIN

Resumable: rows already in <out> are skipped."""
import json
import os
import sys

import ray

HERE = os.path.dirname(os.path.abspath(__file__))


@ray.remote(num_gpus=1)
class P:
    def __init__(self, idx, model, paths):
        import time
        time.sleep(idx * 10)
        sys.path[:0] = paths
        import torch
        from transformers import AutoModelForCausalLM
        from grader import is_correct
        from estimator import seed_of
        self.torch, self.is_correct, self.seed_of = torch, is_correct, seed_of
        self.m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda().eval()

    def _logits(self, ids):
        torch = self.torch
        with torch.no_grad():
            return self.m(torch.tensor([ids], device="cuda")).logits[0]

    def light(self, rec):
        torch = self.torch
        ids = rec["prompt_ids"] + rec["resp_ids"]
        off = len(rec["prompt_ids"])
        lg = self._logits(ids)
        ent, C, pobs = [], [], []
        with torch.no_grad():
            for i in range(off - 1, len(ids) - 1, 512):
                end = min(i + 512, len(ids) - 1)  # rows off-1 .. len-2 predict the response tokens
                lp = torch.log_softmax(lg[i:end].float(), dim=-1)
                p = lp.exp()
                ent.extend((-(p * lp).sum(-1)).tolist())
                C.extend((1.0 - (p * p).sum(-1)).tolist())
                tgt = torch.tensor(ids[i + 1:end + 1], device="cuda")
                pobs.extend(p[torch.arange(p.shape[0], device="cuda"), tgt].tolist())
        del lg
        acc = self.is_correct(rec["text"], rec["gt"])
        assert len(ent) == len(rec["resp_ids"]) == len(pobs), (len(ent), len(rec["resp_ids"]), len(pobs))
        return {"uid": rec["uid"], "qid": rec["uid"].rsplit("_op", 1)[0], "acc": acc, "finish": rec.get("finish"), "gt": rec["gt"],
                "prompt_ids": rec["prompt_ids"], "resp_ids": rec["resp_ids"],
                "ent": [round(e, 4) for e in ent], "C": [round(c, 4) for c in C], "pobs": [round(x, 5) for x in pobs]}

    def law(self, rec, cover, kmin, kmax, rmin):
        torch = self.torch
        ids = rec["prompt_ids"] + rec["resp_ids"]
        off = len(rec["prompt_ids"])
        lg = self._logits(ids)
        out = []
        with torch.no_grad():
            for ps in rec["positions"]:
                row = off + ps["i"] - 1
                p = torch.softmax(lg[row].float(), dim=-1)
                pv, pi = torch.topk(p, 64)
                cum = torch.cumsum(pv, 0)
                K = int((cum < cover).sum().item()) + 1
                K = max(kmin, min(kmax, K))
                head = [[int(pi[k]), float(pv[k])] for k in range(K)]
                r = max(0.0, 1.0 - float(cum[K - 1]))
                tail = []
                if r >= rmin:
                    pt = p.clone()
                    pt[pi[:K]] = 0.0
                    pt = pt / pt.sum()
                    g = torch.Generator(device="cuda").manual_seed(self.seed_of("tail", rec["uid"], ps["i"]))
                    ys = torch.multinomial(pt, 2, replacement=True, generator=g)
                    tail = [[int(y), float(pt[y])] for y in ys]
                H = float(-(p * torch.log(p.clamp_min(1e-30))).sum())
                Cf = float(1.0 - (p * p).sum())
                out.append(dict(ps, H_law=round(H, 4), C_law=round(Cf, 5), pobs=round(float(p[rec["resp_ids"][ps["i"]]]), 5),
                                head=[[t, round(x, 6)] for t, x in head], tail=[[t, round(x, 6)] for t, x in tail], r=round(r, 6)))
        del lg
        return {"uid": rec["uid"], "positions": out}


def main():
    mode, src, model, out = sys.argv[1:5]
    extra = sys.argv[5:]
    rows = [json.loads(l) for l in open(src)]
    done = set()
    try:
        for l in open(out):
            done.add(json.loads(l)["uid"])
    except FileNotFoundError:
        pass
    rows = [r for r in rows if r["uid"] not in done]
    print(f"[profile:{mode}] {len(rows)} rows to do (skip {len(done)}) extra={extra}", flush=True)
    ray.init(address="auto", namespace="full_support")
    nw = max(1, int(ray.cluster_resources().get("GPU", 1)))
    print(f"[profile] {nw} engines", flush=True)
    ws = [P.remote(i, model, [HERE, os.path.dirname(HERE)]) for i in range(nw)]
    futs = {}
    for k, r in enumerate(rows):
        w = ws[k % nw]
        if mode == "light":
            futs[w.light.remote(r)] = r["uid"]
        else:
            cover, kmin, kmax, rmin = float(extra[0]), int(extra[1]), int(extra[2]), float(extra[3])
            futs[w.law.remote(r, cover, kmin, kmax, rmin)] = r["uid"]
    n = 0
    with open(out, "a") as f:
        pending = list(futs)
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            f.write(json.dumps(ray.get(ready[0])) + "\n")
            f.flush()
            n += 1
            if n % 200 == 0:
                print(f"[profile:{mode}] {n}/{len(rows)}", flush=True)
    print(f"[profile:{mode}] done {n}", flush=True)


if __name__ == "__main__":
    main()
