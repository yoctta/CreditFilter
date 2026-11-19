"""Per-token policy entropy (teacher-forced, HF transformers) and grade of the rollouts that can be measured: finished responses
with at least MINTOK tokens that are not format tokens and at most MAXTOT tokens of prompt and response.

  python3 entropy.py <corpus.jsonl> <model> <out.jsonl>        [MINTOK=200 MAXTOT=4160 FORMAT_TOKENS=<ids, comma separated>]

FORMAT_TOKENS defaults to the special tokens of Qwen3; another model family needs its own.  Resumable."""
import json
import os
import sys

import ray

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT_TOKENS = {151643, 151645, 151667, 151668}
if os.environ.get("FORMAT_TOKENS"):
    FORMAT_TOKENS = {int(x) for x in os.environ["FORMAT_TOKENS"].split(",")}


@ray.remote(num_gpus=1)
class E:
    def __init__(self, idx, model, paths):
        import time
        time.sleep(idx * 15)
        sys.path[:0] = paths
        import torch
        from transformers import AutoModelForCausalLM
        from grader import is_correct
        self.torch, self.is_correct = torch, is_correct
        self.m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).cuda().eval()

    def run(self, rec):
        torch = self.torch
        ids = rec["prompt_ids"] + rec["resp_ids"]
        off = len(rec["prompt_ids"])
        with torch.no_grad():
            lg = self.m(torch.tensor([ids], device="cuda")).logits[0]
            ent = []
            for i in range(off - 1, len(ids) - 1, 512):
                lp = torch.log_softmax(lg[i:min(i + 512, len(ids) - 1)].float(), dim=-1)  # rows off-1 .. len-2 predict the response tokens
                ent.extend((-(lp.exp() * lp).sum(-1)).tolist())
            del lg
        return {"uid": rec["uid"], "acc": self.is_correct(rec["text"], rec["gt"]), "prompt_ids": rec["prompt_ids"],
                "resp_ids": rec["resp_ids"], "ent": [round(e, 4) for e in ent]}


def main():
    corpus, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
    mintok, maxtot = int(os.environ.get("MINTOK", "200")), int(os.environ.get("MAXTOT", "4160"))
    rows = []
    for l in open(corpus):
        r = json.loads(l)
        nf = sum(1 for t in r["resp_ids"] if t not in FORMAT_TOKENS)
        if r.get("finish") == "stop" and nf >= mintok and len(r["prompt_ids"]) + len(r["resp_ids"]) <= maxtot:
            rows.append(r)
    done = set()
    try:
        for l in open(out):
            done.add(json.loads(l)["uid"])
    except FileNotFoundError:
        pass
    rows = [r for r in rows if r["uid"] not in done]
    print(f"[entropy] {len(rows)} rollouts to do (skip {len(done)}), format tokens {sorted(FORMAT_TOKENS)}", flush=True)
    if not rows:
        return
    ray.init(address="auto", namespace="four_branch")
    nw = max(1, int(ray.cluster_resources().get("GPU", 1)))
    ws = [E.remote(i, model, [os.path.dirname(HERE)]) for i in range(nw)]
    pend = {ws[i % nw].run.remote(r): r["uid"] for i, r in enumerate(rows)}
    fin = 0
    with open(out, "a") as f:
        while pend:
            rd, _ = ray.wait(list(pend), num_returns=1)
            uid = pend.pop(rd[0])
            try:
                f.write(json.dumps(ray.get(rd[0])) + "\n")
                f.flush()
                fin += 1
                if fin % 50 == 0:
                    print(f"[entropy] {fin}/{len(rows)}", flush=True)
            except Exception as e:
                print(f"[entropy] FAIL {uid}: {type(e).__name__} {e}", flush=True)
    print(f"[entropy] done {fin}/{len(rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
