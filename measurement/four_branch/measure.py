"""The four-branch measurement.

  python3 measure.py select <entropy.jsonl> <corpus.jsonl> <selection.jsonl> [NRESP=640]
  python3 measure.py run <selection.jsonl> <records.jsonl> <model> [J=8]

select   NRESP responses, half correct and half wrong while both last (prompt plus response within MAXTOT tokens, at least 200
         tokens that are not format tokens), and 24 positions per response: 3, 3, 3, 2, 2, 2, 3, 2, 2, 2 uniform draws from the
         ten bands of the entropy rank (edges 0, .025, .05, .075, .1, .15, .2, .3, .4, .6, .8); ranks beyond 0.8 are not sampled
run      at every position the branches are the sampled token and the three most probable alternatives; every branch is
         continued J times with the training sampler (temperature 1, top-p 1) inside a context of MAXLEN tokens, and every
         continuation is graded.  One record per position, in the format of metrics/four_branch; a rerun skips the responses
         already written.

Environment: MAXTOT=4160, MAXLEN=4608, GPU_UTIL=0.85, FORMAT_TOKENS=<ids, comma separated> (default: the special tokens of Qwen3)."""
import hashlib
import json
import math
import os
import random
import sys

import ray

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT_TOKENS = {151643, 151645, 151667, 151668}
if os.environ.get("FORMAT_TOKENS"):
    FORMAT_TOKENS = {int(x) for x in os.environ["FORMAT_TOKENS"].split(",")}
BRANCHES = 4
ALLOCATION = [(0, .025, 3), (.025, .05, 3), (.05, .075, 3), (.075, .1, 2), (.1, .15, 2),
              (.15, .2, 2), (.2, .3, 3), (.3, .4, 2), (.4, .6, 2), (.6, .8, 2)]


def seed_of(*parts):
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def select(stats_file, corpus_file, out_file, nresp=640):
    gt = {}
    for l in open(corpus_file):
        r = json.loads(l)
        gt[r["uid"]] = r["gt"]
    maxtot = int(os.environ.get("MAXTOT", "4160"))
    rows = []
    for l in open(stats_file):
        r = json.loads(l)
        nf = [i for i, t in enumerate(r["resp_ids"]) if t not in FORMAT_TOKENS]
        if len(nf) >= 200 and len(r["prompt_ids"]) + len(r["resp_ids"]) <= maxtot:
            rows.append(r)
    rng = random.Random(41)
    right = [r for r in rows if r["acc"] == 1]
    wrong = [r for r in rows if r["acc"] == 0]
    rng.shuffle(right)
    rng.shuffle(wrong)
    pick = right[:nresp // 2] + wrong[:nresp // 2]
    for pool in (right[nresp // 2:], wrong[nresp // 2:]):
        while len(pick) < nresp and pool:
            pick.append(pool.pop())
    out = []
    for r in pick:
        ids_len = len(r["prompt_ids"]) + len(r["resp_ids"])
        off = len(r["prompt_ids"])
        ent = r["ent"]
        nf = [i for i, t in enumerate(r["resp_ids"]) if t not in FORMAT_TOKENS]
        order = sorted(nf, key=lambda i: (-ent[i], i))
        L = len(nf)
        rg = random.Random(seed_of("positions", r["uid"]))
        pos = []
        for qlo, qhi, n in ALLOCATION:
            seg = [(rk, i) for rk, i in list(enumerate(order))[int(qlo * L):int(qhi * L)]
                   if off + i + 5 <= ids_len]
            pos += [dict(i=i, q=round((rk + 0.5) / L, 5), H=round(ent[i], 4)) for rk, i in rg.sample(seg, min(n, len(seg)))]
        out.append(dict(uid=r["uid"], acc=r["acc"], gt=gt[r["uid"]],
                        prompt_ids=r["prompt_ids"], resp_ids=r["resp_ids"], positions=pos))
    with open(out_file, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    na = sum(1 for o in out if o["acc"] == 1)
    npos = sum(len(o["positions"]) for o in out)
    print(f"[select] {len(out)} responses (correct {na}, wrong {len(out) - na}), {npos} positions", flush=True)


@ray.remote(num_gpus=1)
class G:
    def __init__(self, idx, model, maxlen, gpu_util, paths):
        import time
        time.sleep(idx * 15)
        sys.path[:0] = paths
        from vllm import LLM, SamplingParams
        from grader import is_correct
        self.SP, self.is_correct, self.maxlen = SamplingParams, is_correct, int(maxlen)
        self.llm = LLM(model=model, gpu_memory_utilization=float(gpu_util), max_model_len=self.maxlen)
        self.tok = self.llm.get_tokenizer()

    def run(self, rec, J):
        ids = rec["prompt_ids"] + rec["resp_ids"]
        off = len(rec["prompt_ids"])
        sp0 = self.SP(max_tokens=1, prompt_logprobs=BRANCHES, temperature=0)
        plp = self.llm.generate([{"prompt_token_ids": ids}], sp0, use_tqdm=False)[0].prompt_logprobs
        # branches of every position: the sampled token first, then the most probable alternatives
        recs = []
        for ps in rec["positions"]:
            p = off + ps["i"]
            d = plp[p]
            obs = ids[p]
            cand = sorted(d, key=lambda t: -d[t].logprob)[:BRANCHES]
            if obs not in cand:
                cand[-1] = obs
            j = cand.index(obs)
            cand[0], cand[j] = cand[j], cand[0]
            w = [math.exp(d[c].logprob) for c in cand]
            recs.append((ps, cand, w))
        # the continuations of all positions of the response in one batch
        prompts, sps, meta = [], [], []
        for pi, (ps, cand, w) in enumerate(recs):
            p = off + ps["i"]
            for ci, a in enumerate(cand):
                pin = ids[:p] + [a]
                mx = max(64, self.maxlen - 108 - len(pin))
                prompts.append({"prompt_token_ids": pin})
                sps.append(self.SP(n=J, temperature=1.0, top_p=1.0, max_tokens=mx, seed=seed_of(rec["uid"], ps["i"], ci) % 2 ** 31))
                meta.append((pi, ci))
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        byp = {}
        for (pi, ci), o in zip(meta, outs):
            byp.setdefault(pi, {})[ci] = o
        out = []
        for pi, (ps, cand, w) in enumerate(recs):
            cover = sum(w)
            rp = rec["resp_ids"][:ps["i"]]
            grades, fins, boxes = [], [], []
            for ci, a in enumerate(cand):
                o = byp[pi][ci]
                grades.append([self.is_correct(self.tok.decode(rp + [a] + list(c.token_ids)), rec["gt"]) for c in o.outputs])
                fins.append(sum(1 for c in o.outputs if c.finish_reason == "length"))
                boxes.append(sum(1 for c in o.outputs if "\\boxed" in self.tok.decode(list(c.token_ids)[-80:])))
            out.append(dict(uid=rec["uid"], acc=rec["acc"], i=ps["i"], q=ps["q"], H=ps["H"], cover=round(cover, 4),
                            w=[round(x / cover, 5) for x in w], g=grades, fin=fins, box=boxes))
        return out


def run(selection_file, out_file, model, J=8):
    sels = [json.loads(l) for l in open(selection_file)]
    done = set()
    try:
        for l in open(out_file):
            done.add(json.loads(l)["uid"])
    except FileNotFoundError:
        pass
    sels = [s for s in sels if s["uid"] not in done]
    maxlen, gpu_util = int(os.environ.get("MAXLEN", "4608")), float(os.environ.get("GPU_UTIL", "0.85"))
    print(f"[measure] {len(sels)} responses to do (skip {len(done)}) maxlen={maxlen} gpu_util={gpu_util} J={J}", flush=True)
    if not sels:
        return
    ray.init(address="auto", namespace="four_branch")
    nw = max(1, min(int(ray.cluster_resources().get("GPU", 1)), len(sels)))
    ws = [G.remote(i, model, maxlen, gpu_util, [os.path.dirname(HERE)]) for i in range(nw)]
    pend = {ws[i % nw].run.remote(s, J): s["uid"] for i, s in enumerate(sels)}
    fin = 0
    with open(out_file, "a") as f:
        while pend:
            rd, _ = ray.wait(list(pend), num_returns=1)
            uid = pend.pop(rd[0])
            try:
                for row in ray.get(rd[0]):
                    f.write(json.dumps(row) + "\n")
                f.flush()
                fin += 1
                if fin % 8 == 0:
                    print(f"[measure] {fin}/{len(sels)} responses", flush=True)
            except Exception as e:
                print(f"[measure] FAIL {uid}: {type(e).__name__} {str(e)[:180]}", flush=True)
    print(f"[measure] done {fin}/{len(sels)} -> {out_file}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "select":
        select(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 640)
    elif len(sys.argv) >= 5 and sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 8)
    else:
        sys.exit(__doc__)
