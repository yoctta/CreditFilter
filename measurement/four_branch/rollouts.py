"""Policy rollouts for the four-branch instrument: N responses per training prompt with the training sampler (temperature 1,
top-p 1, at most MAXNEW new tokens).  Resumable per prompt.

  MODEL=<model dir> OUT=<corpus.jsonl> [NQ=288 QOFFSET=224 N=8 MAXNEW=5800 GEN_MAXLEN=8192 NW=8] python3 rollouts.py"""
import json
import os
import sys

import ray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import load_prompts  # noqa: E402


@ray.remote(num_gpus=1)
class G:
    def __init__(self, idx, model, maxnew, n, maxlen):
        import time
        time.sleep(idx * 20)
        from vllm import LLM, SamplingParams
        self.SP, self.n, self.maxnew = SamplingParams, n, maxnew
        self.llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=int(maxlen))

    def gen(self, batch):
        sp = self.SP(n=self.n, temperature=1.0, top_p=1.0, max_tokens=self.maxnew)
        outs = self.llm.generate([{"prompt_token_ids": b["prompt_ids"]} for b in batch], sp)
        rows = []
        for b, o in zip(batch, outs):
            for i, c in enumerate(o.outputs):
                rows.append({"uid": f"{b['qid']}_op{i}", "prompt_ids": b["prompt_ids"],
                             "resp_ids": list(c.token_ids), "text": c.text, "gt": b["gt"],
                             "finish": "stop" if c.finish_reason == "stop" else c.finish_reason})
        return rows


def main():
    model, out = os.environ["MODEL"], os.environ["OUT"]
    nq, off, n = int(os.environ.get("NQ", "288")), int(os.environ.get("QOFFSET", "224")), int(os.environ.get("N", "8"))
    maxnew, maxlen, nw = int(os.environ.get("MAXNEW", "5800")), int(os.environ.get("GEN_MAXLEN", "8192")), int(os.environ.get("NW", "8"))
    qs = load_prompts(model, off, nq)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try:
                done.add(json.loads(l)["uid"].rsplit("_op", 1)[0])
            except Exception:
                pass
    qs = [q for q in qs if q["qid"] not in done]
    print(f"[rollouts] {len(qs)} prompts to do (skip {len(done)}) N={n} MAXNEW={maxnew} max_model_len={maxlen}", flush=True)
    ray.init(address="auto", namespace="four_branch")
    ws = [G.remote(i, model, maxnew, n, maxlen) for i in range(nw)]
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
