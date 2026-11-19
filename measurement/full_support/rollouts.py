"""Policy rollouts for the full-support instrument: N responses per training prompt with the training sampler (temperature 1,
top-p 1, at most MAXNEW new tokens), one request per rollout with its own seed.  Resumable per prompt.

  MODEL=<model dir> OUT=<corpus.jsonl> [NQ=512 N=16 MAXNEW=5800 GEN_MAXLEN=8192 SEED=13 NW=8] python3 rollouts.py"""
import json
import os
import sys

import ray

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from estimator import seed_of  # noqa: E402
from prompts import load_prompts  # noqa: E402


@ray.remote(num_gpus=1)
class G:
    def __init__(self, idx, model, maxnew, maxlen, n, seed):
        import time
        time.sleep(idx * 15)
        from vllm import LLM, SamplingParams
        self.SP, self.n, self.maxnew, self.seed = SamplingParams, n, maxnew, seed
        self.llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=int(maxlen), enable_prefix_caching=True, seed=0)

    def gen(self, batch):
        prompts, sps, meta = [], [], []
        for b in batch:
            for k in range(self.n):
                prompts.append({"prompt_token_ids": b["prompt_ids"]})
                sps.append(self.SP(n=1, temperature=1.0, top_p=1.0, max_tokens=self.maxnew, seed=seed_of("gen", self.seed, b["qid"], k)))
                meta.append((b, k))
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        rows = []
        for (b, k), o in zip(meta, outs):
            c = o.outputs[0]
            rows.append({"uid": f"{b['qid']}_op{k}", "prompt_ids": b["prompt_ids"], "resp_ids": list(c.token_ids), "text": c.text,
                         "gt": b["gt"], "finish": "stop" if c.finish_reason == "stop" else c.finish_reason})
        return rows


def main():
    model, out = os.environ["MODEL"], os.environ["OUT"]
    nq, n = int(os.environ.get("NQ", "512")), int(os.environ.get("N", "16"))
    maxnew, maxlen, seed, nw = int(os.environ.get("MAXNEW", "5800")), int(os.environ.get("GEN_MAXLEN", "8192")), int(os.environ.get("SEED", "13")), int(os.environ.get("NW", "8"))
    qs = load_prompts(model, 0, nq)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                done.add(json.loads(l)["uid"].rsplit("_op", 1)[0])
            except Exception:
                pass
    qs = [q for q in qs if q["qid"] not in done]
    print(f"[rollouts] {len(qs)} prompts to do (skip {len(done)}) N={n} MAXNEW={maxnew} max_model_len={maxlen} SEED={seed}", flush=True)
    ray.init(address="auto", namespace="full_support")
    nw = max(1, min(nw, int(ray.cluster_resources().get("GPU", 1))))
    print(f"[rollouts] {nw} engines", flush=True)
    ws = [G.remote(i, model, maxnew, maxlen, n, seed) for i in range(nw)]
    chunks = [qs[i::nw] for i in range(nw)]
    futs = [ws[i].gen.remote(chunks[i]) for i in range(nw) if chunks[i]]
    with open(out, "a") as f:
        for fu in futs:
            for row in ray.get(fu):
                f.write(json.dumps(row) + "\n")
            f.flush()
    print("[rollouts] done", flush=True)


if __name__ == "__main__":
    main()
