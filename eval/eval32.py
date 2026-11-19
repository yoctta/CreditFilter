#!/usr/bin/env python3
"""Evaluation of checkpoints with the protocol of the paper's tables.

k = 32 samples per problem at temperature 1.0 and top-p 0.7, at most 20480 new tokens, on MATH500 (500 problems),
AIME2024 (30) and AIME2025 (30).  One node: every GPU holds one vLLM copy of the model and takes a share of the problems.

  python3 eval/eval32.py --plan plan.json --out results/<name> [--k 32 --temp 1.0 --top-p 0.7 --max-new 20480 --ngpu 8 --seed 0]

plan.json: [{"tag": <name>, "hf": <dir with safetensors>} or {"tag": ..., "actor": <global_step_N/actor>, "hf": <target dir>}];
an entry with "actor" is merged to HF safetensors first.

Per tag: samples.jsonl with one line per (problem, sample): set, qidx, s, acc (training grader), acc_mv (Math-Verify, when
installed), pred (normalised boxed answer), boxed_raw, ntok, stop, boxed; summary.json with the accuracy per set and a
problem-bootstrap interval.  Per run: runcard.json (arguments, package versions, md5 of the problem sets).
The run stops instead of writing zeros when the rendered prompt is not ChatML with the user message exactly once, when
more than 5% of the samples fail in the grader, or when a GPU already holds memory."""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
import socket
import subprocess
import sys
import time

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SETS_DEFAULT = ["MATH500", "AIME2024", "AIME2025"]


def load_tasks(bench_root, sets, limit):
    import pyarrow.parquet as pq
    tasks = []
    for s in sets:
        rows = pq.read_table(f"{bench_root}/{s}/test.parquet").to_pylist()
        if limit:
            rows = rows[:limit]
        for i, r in enumerate(rows):
            msgs = [dict(m) for m in list(r["prompt"])]
            tasks.append({"set": s, "qidx": i, "prompt": msgs, "gt": str(r["reward_model"]["ground_truth"])})
    return tasks


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_safetensors(d):
    return bool(glob.glob(os.path.join(d, "*.safetensors")))


def merge(actor_dir, hf_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = PKG_ROOT + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print(f"[merge] {actor_dir} -> {hf_dir}", flush=True)
    subprocess.run([sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
                    "--local_dir", actor_dir, "--target_dir", hf_dir], check=True, env=env, cwd=PKG_ROOT)
    assert has_safetensors(hf_dir), f"merge produced no safetensors in {hf_dir}"


TEMPLATE_KW = {}  # extra arguments of apply_chat_template (--enable-thinking)


def template_check(tok, first_prompt, allow_plain):
    """The rendered prompt must be ChatML and contain the user message exactly once."""
    txt = tok.apply_chat_template(first_prompt, tokenize=False, add_generation_prompt=True, **TEMPLATE_KW)
    user_msgs = [m["content"] for m in first_prompt if m.get("role") == "user"]
    ok_chatml = txt.startswith("<|im_start|>user") or txt.startswith("<|im_start|>system")
    ok_tail = txt.rstrip().endswith("<|im_start|>assistant") or (TEMPLATE_KW.get("enable_thinking") is False and txt.rstrip().endswith("<|im_start|>assistant\n<think>\n\n</think>"))
    n_user_turns = txt.count("<|im_start|>user")
    n_content = txt.count(user_msgs[-1]) if user_msgs else -1
    verdict = {"chatml": ok_chatml, "assistant_tail": ok_tail, "user_turns": n_user_turns, "user_content_copies": n_content,
               "rendered_prompt_head": txt[:400]}
    print("[template] rendered first prompt (head): " + json.dumps(txt[:400]), flush=True)
    print("[template] verdict: " + json.dumps({k: v for k, v in verdict.items() if k != "rendered_prompt_head"}), flush=True)
    if not (ok_chatml and ok_tail and n_user_turns == len(user_msgs) and n_content == 1):
        msg = f"template guard: chatml={ok_chatml} assistant_tail={ok_tail} user_turns={n_user_turns} user_content_copies={n_content}"
        if allow_plain:
            print("[template] WARNING (allowed by flag): " + msg, flush=True)
        else:
            raise RuntimeError(msg + " - refusing; pass --allow-plain-template only if this is intended")
    return verdict


def gpu_preflight(ngpu, max_used_mib=2048):
    """Every GPU to be used must be nearly empty: vLLM reserves most of the card."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except Exception as e:
        print(f"[preflight] nvidia-smi unavailable ({e}); skipping the GPU memory check", flush=True)
        return
    busy = []
    for line in out[:ngpu]:
        idx, used, total = [x.strip() for x in line.split(",")]
        if int(used) > max_used_mib:
            busy.append(f"GPU{idx} {used}/{total} MiB used")
    print(f"[preflight] {len(out)} GPU(s) visible, using {ngpu}; memory in use: " + "; ".join(l.strip() for l in out[:ngpu]), flush=True)
    if busy:
        raise RuntimeError("GPU(s) already hold memory: " + "; ".join(busy))


def worker(gpu, hf_dir, tasks, args, out_path, seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if PKG_ROOT not in sys.path:
        sys.path.insert(0, PKG_ROOT)
    # the training grader, loaded by path as verl does for training
    import importlib.util
    _spec = importlib.util.spec_from_file_location("custom_score", os.path.join(PKG_ROOT, "creditfilter", "custom_score.py"))
    _custom_score = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_custom_score)
    compute_score = _custom_score.compute_score
    from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed, normalize_final_answer
    # Math-Verify accepts equivalent forms that exact match rejects (\dfrac vs \frac, .5 vs 0.5, set order); it is applied
    # when installed, and the raw boxed answer is stored so that any grader can be run afterwards
    try:
        from math_verify import parse as _mv_parse, verify as _mv_verify
        _mv_gold = {}
        def grade_mv(gt, raw):
            if not raw: return 0
            key = str(gt)
            if key not in _mv_gold: _mv_gold[key] = _mv_parse("$" + key + "$")
            g = _mv_gold[key]; pr = _mv_parse("$" + raw + "$")
            return int(bool(_mv_verify(g, pr, timeout_seconds=3))) if (g and pr) else 0
    except Exception:  # math-verify not installed: acc_mv is null
        grade_mv = None
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
    if gpu == 0:
        template_check(tok, tasks[0]["prompt"], args.allow_plain_template)
    llm = LLM(model=hf_dir, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              dtype="bfloat16", trust_remote_code=True, seed=seed)
    sp = SamplingParams(n=args.k, temperature=args.temp, top_p=args.top_p, max_tokens=args.max_new, seed=seed)
    prompts = [tok.apply_chat_template(t["prompt"], tokenize=False, add_generation_prompt=True, **TEMPLATE_KW) for t in tasks]
    outs = llm.generate(prompts, sp)
    nerr = 0
    with open(out_path, "w") as f:
        for t, o in zip(tasks, outs):
            for si, c in enumerate(o.outputs):
                try:
                    r = compute_score(t["set"], c.text, t["gt"], None)
                    acc = int(bool(r.get("acc", r.get("score", 0) > 0))) if isinstance(r, dict) else int(float(r) > 0)
                except Exception:
                    nerr += 1
                    acc = 0
                bx = last_boxed_only_string(c.text)
                boxed = int(bx is not None)
                try:
                    pred = normalize_final_answer(remove_boxed(bx))[:64] if bx else ""
                except Exception:
                    pred = "?"
                try:
                    boxed_raw = remove_boxed(bx)[:2000] if bx else ""
                except Exception:
                    boxed_raw = "?"
                acc_mv = None
                if grade_mv is not None:
                    try:
                        acc_mv = grade_mv(t["gt"], boxed_raw if boxed_raw != "?" else "")
                    except Exception:
                        acc_mv = None
                f.write(json.dumps({"set": t["set"], "qidx": t["qidx"], "s": si, "acc": acc, "acc_mv": acc_mv, "pred": pred,
                                    "boxed_raw": boxed_raw, "ntok": len(c.token_ids), "stop": str(c.finish_reason), "boxed": boxed}) + "\n")
    total = sum(len(o.outputs) for o in outs)
    if total and nerr / total > 0.05:
        raise RuntimeError(f"grader error rate {nerr}/{total} > 5%")
    print(f"[worker {gpu}] done {len(tasks)} problems, {total} samples, grader_err={nerr}", flush=True)


def tokenizer_record(hf_dir):
    rec = {}
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"):
        p = os.path.join(hf_dir, fn)
        rec[fn] = md5(p) if os.path.exists(p) else None
    return rec


def eval_ckpt(tag, hf_dir, tasks, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    shards = [tasks[i::args.ngpu] for i in range(args.ngpu)]
    procs, parts = [], []
    for g in range(args.ngpu):
        if not shards[g]:
            continue
        part = os.path.join(out_dir, f"part{g}.jsonl")
        parts.append(part)
        p = ctx.Process(target=worker, args=(g, hf_dir, shards[g], args, part, args.seed * 1000 + args.shard_i * args.ngpu + g))   # distinct seeds per GPU and shard
        p.start()
        procs.append(p)
    while any(p.is_alive() for p in procs):
        time.sleep(5)
        if any(p.exitcode not in (None, 0) for p in procs):
            print("[eval32] a worker failed - terminating the others", flush=True)
            for p in procs:
                if p.is_alive():
                    p.terminate()
            break
    for p in procs:
        p.join()
    bad = [p.exitcode for p in procs if p.exitcode != 0]
    if bad:
        raise RuntimeError(f"{tag}: {len(bad)} worker(s) failed (exit codes {bad}) - see the log above")
    rows = []
    with open(os.path.join(out_dir, "samples.jsonl"), "w") as fo:
        for part in parts:
            for line in open(part):
                rows.append(json.loads(line))
                fo.write(line)
            os.remove(part)
    expect = len(tasks) * args.k
    if len(rows) < expect * 0.98:
        raise RuntimeError(f"{tag}: only {len(rows)}/{expect} samples - refusing to summarise")
    if args.shard_n > 1:   # one shard of a multi-node evaluation; eval/merge_shards.py assembles samples.jsonl + summary.json
        json.dump({"tag": tag, "hf_dir": hf_dir, "shard": f"{args.shard_i}/{args.shard_n}", "n_tasks": len(tasks), "n_rows": len(rows), "k": args.k,
                   "temp": args.temp, "top_p": args.top_p, "max_new": args.max_new, "seed": args.seed, "sets": args.sets, "limit": args.limit,
                   "tokenizer_md5": tokenizer_record(hf_dir)}, open(os.path.join(out_dir, "shard.json"), "w"), indent=1)
        print(f"[SHARD DONE] {tag}: shard {args.shard_i}/{args.shard_n}, {len(tasks)} problems, {len(rows)} samples", flush=True)
        return
    summ = summarize(tag, hf_dir, rows, args)
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    line = " | ".join(f"{s} {summ[s]['acc']:.4f} [{summ[s]['ci95'][0]:.3f},{summ[s]['ci95'][1]:.3f}] n={summ[s]['n_q']}" for s in args.sets)
    print(f"[VERDICT] {tag}: mean@{args.k}  {line}", flush=True)


def summarize(tag, hf_dir, rows, args):
    """summary.json content for one checkpoint: accuracy + 2000-replicate problem bootstrap per set (shared with merge_shards.py)."""
    summ = {"tag": tag, "hf_dir": hf_dir, "k": args.k, "temp": args.temp, "top_p": args.top_p, "max_new": args.max_new,
            "seed": args.seed, "n_samples": len(rows), "tokenizer_md5": tokenizer_record(hf_dir)}
    rnd = random.Random(0)
    for s in args.sets:
        rs = [r for r in rows if r["set"] == s]
        qacc = {}
        for r in rs:
            qacc.setdefault(r["qidx"], []).append(r["acc"])
        pq = [sum(v) / len(v) for v in qacc.values()]
        acc = sum(pq) / len(pq)
        bs = sorted(sum(pq[rnd.randrange(len(pq))] for _ in pq) / len(pq) for _ in range(2000))
        summ[s] = {"acc": round(acc, 4), "n_q": len(pq), "k": args.k, "ci95": [round(bs[50], 4), round(bs[1949], 4)],
                   "boxed_rate": round(sum(r["boxed"] for r in rs) / len(rs), 4),
                   "mean_ntok": round(sum(r["ntok"] for r in rs) / len(rs), 1),
                   "truncated_rate": round(sum(r["stop"] == "length" for r in rs) / len(rs), 4),
                   "zero_flag": acc == 0.0}
        if acc == 0.0:
            print(f"[VERDICT WARNING] {tag}/{s}: accuracy 0.0", flush=True)
    return summ


def versions():
    import importlib.metadata as im
    v = {}
    for pkg in ("vllm", "torch", "transformers", "tokenizers", "flashinfer-python", "triton"):
        try:
            v[pkg] = im.version(pkg)
        except Exception:
            v[pkg] = "missing"
    try:
        v["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[:1]
    except Exception as e:
        v["gpu"] = f"unavailable: {e}"
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=20480)
    ap.add_argument("--max-model-len", type=int, default=21504)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--ngpu", type=int, default=0, help="0 = all visible GPUs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default=",".join(SETS_DEFAULT))
    ap.add_argument("--limit", type=int, default=0, help="first N problems per set (smoke only)")
    ap.add_argument("--shard", default="0/1", help="I/N: evaluate problems I, I+N, I+2N, ... only (multi-node split; assemble with eval/merge_shards.py)")
    ap.add_argument("--merge-only", action="store_true", help="only merge FSDP shards to safetensors for the plan's checkpoints, then exit")
    ap.add_argument("--bench-root", default=os.path.join(PKG_ROOT, "data", "eval"))
    ap.add_argument("--allow-plain-template", action="store_true")
    ap.add_argument("--enable-thinking", default="default", choices=["default", "true", "false"], help="pass enable_thinking to the chat template of models that have the switch (default: not passed)")
    args = ap.parse_args()
    if args.enable_thinking != "default":
        TEMPLATE_KW["enable_thinking"] = (args.enable_thinking == "true")
        print(f"[template] apply_chat_template kwargs: {TEMPLATE_KW}", flush=True)
    args.sets = [s for s in args.sets.split(",") if s]
    args.shard_i, args.shard_n = (int(x) for x in args.shard.split("/")); assert 0 <= args.shard_i < args.shard_n, args.shard
    if args.ngpu <= 0:
        import torch
        args.ngpu = torch.cuda.device_count()
    assert args.ngpu > 0, "no GPU visible"
    os.makedirs(args.out, exist_ok=True)
    plan = json.load(open(args.plan))
    runcard = {"argv": sys.argv, "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
               "hostname": socket.gethostname(), "platform": platform.platform(), "python": sys.version.split()[0],
               "versions": versions(), "plan_md5": md5(args.plan), "plan": plan, "pkg_root": PKG_ROOT,
               "bench_md5": {s: md5(f"{args.bench_root}/{s}/test.parquet") for s in args.sets},
               "protocol": {"k": args.k, "temp": args.temp, "top_p": args.top_p, "max_new": args.max_new,
                            "max_model_len": args.max_model_len, "seed": args.seed, "ngpu": args.ngpu, "limit": args.limit, "shard": args.shard}}
    json.dump(runcard, open(os.path.join(args.out, "runcard.json"), "w"), indent=1)
    print("[eval32] runcard: " + json.dumps({k: runcard[k] for k in ("hostname", "versions", "protocol", "plan_md5", "bench_md5")}), flush=True)
    gpu_preflight(args.ngpu)
    if args.merge_only:   # prepare the safetensors once before several shard jobs share the checkpoint
        for e in plan:
            if not has_safetensors(e["hf"]):
                assert e.get("actor"), f"{e['tag']}: {e['hf']} has no safetensors and the plan gives no 'actor' dir to merge from"
                merge(e["actor"], e["hf"])
            print(f"[eval32] {e['tag']}: safetensors ready at {e['hf']}", flush=True)
        print("[eval32] MERGE DONE", flush=True)
        return
    tasks = load_tasks(args.bench_root, args.sets, args.limit)
    if args.shard_n > 1:
        tasks = tasks[args.shard_i::args.shard_n]
        print(f"[eval32] shard {args.shard_i}/{args.shard_n}: {len(tasks)} problems", flush=True)
    print(f"[eval32] {len(tasks)} problems x k={args.k} per tag; {len(plan)} tag(s); {args.ngpu} GPU worker(s)", flush=True)
    for e in plan:
        tag, hf = e["tag"], e["hf"]
        out_dir = os.path.join(args.out, tag)
        done_marker = "shard.json" if args.shard_n > 1 else "summary.json"
        if os.path.exists(os.path.join(out_dir, done_marker)):
            print(f"[eval32] {tag}: {done_marker} exists, skipping", flush=True)
            continue
        if not has_safetensors(hf):
            assert e.get("actor"), f"{tag}: {hf} has no safetensors and the plan gives no 'actor' dir to merge from"
            merge(e["actor"], hf)
        t0 = time.time()
        eval_ckpt(tag, hf, tasks, out_dir, args)
        print(f"[eval32] {tag}: {time.time() - t0:.0f} s", flush=True)
    runcard["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    json.dump(runcard, open(os.path.join(args.out, "runcard.json"), "w"), indent=1)
    print("[eval32] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
