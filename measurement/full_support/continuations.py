"""Sampled continuations of every branch at every selected position.

  python3 continuations.py <law.jsonl> <tasks.jsonl> <out.jsonl> <model> M MAXNEW MAXLEN GPU_UTIL MAX_NUM_SEQS BATCH NGRADE

One vLLM engine per GPU; whole responses are batched (all positions and branches; the M continuations of a branch share the prefix
through prefix caching); the continuations are graded in a process pool of the driver while the engines run.  The continuation
budget of a branch at response index i is MAXNEW - (i + 1) tokens, the cap of training on the whole response, and at least 16.
Output: one record per position; a rerun skips the positions already written."""
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import ray

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from estimator import seed_of  # noqa: E402


def _grade_init():
    sys.path.insert(0, os.path.dirname(HERE))
    global _is_correct
    from grader import is_correct
    _is_correct = is_correct


def _grade(args):
    return _is_correct(*args)


@ray.remote(num_gpus=1)
class F:
    def __init__(self, idx, model, maxlen, gpu_util, max_num_seqs):
        time.sleep(idx * 15)
        from vllm import LLM, SamplingParams
        self.SP = SamplingParams
        self.maxlen = int(maxlen)
        self.llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=float(gpu_util), max_model_len=self.maxlen,
                       max_num_seqs=int(max_num_seqs), enable_prefix_caching=True, seed=0)
        self.tok = self.llm.get_tokenizer()

    def run(self, batch, m, maxnew):
        prompts, sps, meta = [], [], []
        for r in batch:
            for ps in r["positions"]:
                branches = [t for t, p in ps["head"]] + [t for t, p in ps["tail"]]
                for bi, t in enumerate(branches):
                    pin = r["prompt_ids"] + r["resp_ids"][:ps["i"]] + [t]
                    mx = max(16, min(maxnew - (ps["i"] + 1), self.maxlen - len(pin) - 1))
                    seed = (seed_of("fork", r["uid"], ps["i"], bi) % (2 ** 31 // 256)) * 256  # the seeds s .. s+m-1 of the continuations never overlap across branches
                    prompts.append({"prompt_token_ids": pin})
                    sps.append(self.SP(n=int(m), temperature=1.0, top_p=1.0, max_tokens=mx, seed=seed, detokenize=False))
                    meta.append((r["uid"], ps["i"], bi, len(r["prompt_ids"])))
        order = sorted(range(len(prompts)), key=lambda k: -sps[k].max_tokens)  # longest budgets first: the engine serves requests in order
        prompts = [prompts[k] for k in order]
        sps = [sps[k] for k in order]
        meta = [meta[k] for k in order]
        t0 = time.time()
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        dt = time.time() - t0
        res, ntok = [], 0
        for (uid, i, bi, off), o, pr in zip(meta, outs, prompts):
            resp_prefix = pr["prompt_token_ids"][off:]
            conts = []
            for c in o.outputs:
                ids = list(c.token_ids)
                ntok += len(ids)
                txt = self.tok.decode(resp_prefix + ids)
                conts.append((len(ids), 1 if c.finish_reason == "length" else 0, 1 if "\\boxed" in self.tok.decode(ids[-80:]) else 0, txt))
            res.append((uid, i, bi, conts))
        return dict(res=res, dt=dt, ntok=ntok, nreq=len(prompts))


def main():
    law, tasks, out, model = sys.argv[1:5]
    m, maxnew, maxlen = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    gpu_util, max_num_seqs, batch_n, ngrade = float(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10]), int(sys.argv[11])
    T = {json.loads(l)["uid"]: json.loads(l) for l in open(tasks)}
    done = set()
    try:
        for l in open(out):
            r = json.loads(l)
            done.add((r["uid"], r["i"]))
    except FileNotFoundError:
        pass
    work = []
    for l in open(law):
        r = json.loads(l)
        t = T[r["uid"]]
        pos = [p for p in r["positions"] if (r["uid"], p["i"]) not in done]
        if pos:
            work.append(dict(uid=r["uid"], qid=t["qid"], acc=t["acc"], gt=t["gt"], prate=t["prate"], L=t["L"],
                             prompt_ids=t["prompt_ids"], resp_ids=t["resp_ids"], positions=pos))
    nbr = sum(len(p["head"]) + len(p["tail"]) for w in work for p in w["positions"])
    npos = sum(len(w["positions"]) for w in work)
    print(f"[continuations] responses {len(work)} positions {npos} branches {nbr} continuations {nbr * m} (skip {len(done)} positions) "
          f"M={m} MAXNEW={maxnew} MAXLEN={maxlen} GPU_UTIL={gpu_util} MAX_NUM_SEQS={max_num_seqs} BATCH={batch_n} NGRADE={ngrade}", flush=True)
    if not work:
        return
    pool = ProcessPoolExecutor(max_workers=ngrade, mp_context=mp.get_context("forkserver"), initializer=_grade_init)
    ray.init(address="auto", namespace="full_support")
    nw = max(1, int(ray.cluster_resources().get("GPU", 1)))
    print(f"[continuations] {nw} engines", flush=True)
    ws = [F.remote(i, model, maxlen, gpu_util, max_num_seqs) for i in range(nw)]
    work.sort(key=lambda r: -sum(max(16, maxnew - (p["i"] + 1)) * (len(p["head"]) + len(p["tail"])) for p in r["positions"]))  # heaviest responses first
    pending = list(work)
    nbatches_est = (len(work) + batch_n - 1) // batch_n

    units = []  # at the end: parts of a response (a subset of its positions), so that the last responses spread over all engines

    def take():
        rem = len(pending)
        if rem == 0 and not units:
            return None
        if units:
            return [units.pop(0)]
        if rem <= nw:  # fewer responses than engines: split their positions into about nw parts
            tot = sum(len(r["positions"]) for r in pending)
            csz = max(1, -(-tot // nw))
            for r in pending:
                for k in range(0, len(r["positions"]), csz):
                    units.append(dict(r, positions=r["positions"][k:k + csz]))
            pending.clear()
            return [units.pop(0)]
        size = batch_n if rem > 2 * batch_n * nw else max(1, min(batch_n, rem // nw))  # smaller batches towards the end: every engine keeps a share of the last work
        b = pending[:size]
        del pending[:size]
        return b

    inflight = {}
    for w in ws:
        b = take()
        if b:
            inflight[w.run.remote(b, m, maxnew)] = (w, b)
    t_start = time.time()
    stat = dict(ncont=0, ntok=0, nb=0)
    q = queue.Queue(maxsize=6)  # grading and writing run in a thread, so that an engine never waits for the grader

    def writer():
        with open(out, "a") as f:
            while True:
                item = q.get()
                if item is None:
                    return
                batch, res = item
                _grade_and_write(batch, res, pool, f, m, stat, t_start, nbatches_est)

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    while inflight:
        ready, _ = ray.wait(list(inflight), num_returns=1)
        fut = ready[0]
        actor, batch = inflight.pop(fut)
        res = ray.get(fut)
        b = take()
        if b:
            inflight[actor.run.remote(b, m, maxnew)] = (actor, b)
        q.put((batch, res))
    q.put(None)
    th.join()
    pool.shutdown()
    print(f"[continuations] done: {stat['ncont']} continuations, {stat['ntok']} tokens, {(time.time() - t_start) / 3600:.2f} h", flush=True)


def _grade_and_write(batch, res, pool, f, m, stat, t_start, nbatches):
    byp = {}
    for uid, i, bi, conts in res["res"]:
        byp.setdefault((uid, i), {})[bi] = conts
    jobs = []
    for r in batch:
        for ps in r["positions"]:
            brs = byp[(r["uid"], ps["i"])]
            for bi in range(len(ps["head"]) + len(ps["tail"])):
                jobs += [(txt, r["gt"]) for (_, _, _, txt) in brs[bi]]
    grades = list(pool.map(_grade, jobs, chunksize=64))
    gi = 0
    for r in batch:
        for ps in r["positions"]:
            brs = byp[(r["uid"], ps["i"])]
            toks = [(t, p, "h") for t, p in ps["head"]] + [(t, p, "t") for t, p in ps["tail"]]
            br = []
            for bi, (t, p, kind) in enumerate(toks):
                conts = brs[bi]
                g = grades[gi:gi + len(conts)]
                gi += len(conts)
                br.append(dict(t=t, p=p, k=kind, S=sum(g), m=len(conts), len=[c[0] for c in conts],
                               nlen=sum(c[1] for c in conts), nbox=sum(c[2] for c in conts)))
            rec = dict(uid=r["uid"], qid=r["qid"], acc=r["acc"], prate=r["prate"], L=r["L"], i=ps["i"], q=ps["q"], H=ps["H"], C=ps["C"],
                       H_law=ps.get("H_law"), C_law=ps.get("C_law"), pobs=ps.get("pobs"), s=ps["s"], g=ps["g"], Ns=ps["Ns"], Ng=ps["Ng"],
                       pi=ps["pi"], r=ps["r"], cover=round(1.0 - ps["r"], 6), br=br)
            f.write(json.dumps(rec) + "\n")
    f.flush()
    stat["nb"] += 1
    stat["ncont"] += res["nreq"] * m
    stat["ntok"] += res["ntok"]
    el = time.time() - t_start
    print(f"[continuations] batch {stat['nb']}/{nbatches} responses {len(batch)} requests {res['nreq']} engine {res['dt']:.0f}s "
          f"cum {stat['ncont']} continuations {stat['ntok'] / 1e6:.1f}M tokens in {el / 60:.1f} min = {stat['ncont'] / el:.1f} cont/s {stat['ntok'] / el / 1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
