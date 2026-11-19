#!/usr/bin/env python3
"""Builds plan.json for eval32.py from checkpoint directories.

  python3 eval/make_plan.py --out plan.json [--base <hf model dir>] [--runs-root runs --steps 350 375 400 [--arms creditfilter dapo ...]]
                            [--ckpt TAG=<global_step_N dir> ...]

  --base        the base model (tag base_step0)
  --runs-root   <runs-root>/<arm>_<YYYYMMDD_HHMMSS>/global_step_<N>/, the run directories of launchers/run_arm.sh;
                the latest run of each arm is taken
  --ckpt        an explicit checkpoint directory"""
import argparse
import glob
import json
import os
import re


def entry(tag, step_dir):
    hf = os.path.join(step_dir, "hf")
    actor = os.path.join(step_dir, "actor")
    if glob.glob(os.path.join(step_dir, "*.safetensors")):
        return {"tag": tag, "hf": step_dir}
    if not os.path.isdir(actor) and not glob.glob(os.path.join(hf, "*.safetensors")):
        raise SystemExit(f"{tag}: {step_dir} has neither actor/ shards nor hf/*.safetensors")
    return {"tag": tag, "actor": actor, "hf": hf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base")
    ap.add_argument("--runs-root")
    ap.add_argument("--steps", type=int, nargs="*", default=[350, 375, 400])
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--ckpt", nargs="*", default=[])
    a = ap.parse_args()
    plan = []
    if a.base:
        assert glob.glob(os.path.join(a.base, "*.safetensors")), f"--base {a.base} has no safetensors"
        plan.append({"tag": "base_step0", "hf": a.base})
    if a.runs_root:
        runs = {}
        for d in sorted(glob.glob(os.path.join(a.runs_root, "*"))):
            m = re.match(r"(.+?)_(\d{8}_\d{6})$", os.path.basename(d))
            if not m or "_smoke_" in os.path.basename(d):
                continue
            arm = m.group(1)
            if a.arms and arm not in a.arms:
                continue
            runs[arm] = d
        for arm, d in sorted(runs.items()):
            for s in a.steps:
                sd = os.path.join(d, f"global_step_{s}")
                if os.path.isdir(sd):
                    plan.append(entry(f"{arm}_{s}", sd))
                else:
                    print(f"[make_plan] {arm}: no global_step_{s} in {d} (skipped)")
    for kv in a.ckpt:
        tag, d = kv.split("=", 1)
        plan.append(entry(tag, d))
    assert plan, "empty plan"
    json.dump(plan, open(a.out, "w"), indent=1)
    print(f"[make_plan] {len(plan)} entries -> {a.out}: " + ", ".join(e["tag"] for e in plan))


if __name__ == "__main__":
    main()
