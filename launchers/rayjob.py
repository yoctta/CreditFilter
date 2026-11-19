#!/usr/bin/env python3
"""Ray Jobs helper over the dashboard's REST interface (standard library only).

  rayjob.py status <head_ip:8265> <submission_id>
  rayjob.py wait   <head_ip:8265> <submission_id> [timeout_s]        -> prints final status; exit 0 iff SUCCEEDED
  rayjob.py logs   <head_ip:8265> <submission_id> [outfile]          -> driver log (stdout or file)
  rayjob.py submit <head_ip:8265> <submission_id> <entrypoint> [env_vars_json] [entrypoint_gpus]  -> prints submission_id (small payloads only)
"""
import json
import sys
import time
import urllib.error
import urllib.request

TERMINAL = ("SUCCEEDED", "FAILED", "STOPPED")


def _req(url, data=None, timeout=60):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # direct connection, no HTTP proxy
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with opener.open(req, timeout=timeout) as r:
        return r.read().decode()


def status(base, sid):
    return json.loads(_req(f"{base}/api/jobs/{sid}")).get("status", "?")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    cmd, head, sid = sys.argv[1], sys.argv[2], sys.argv[3]
    base = f"http://{head}"
    if cmd == "status":
        print(status(base, sid))
    elif cmd == "wait":
        timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 7200.0
        t0 = time.time()
        errs = 0
        last_print = 0.0
        st = "?"
        while time.time() - t0 < timeout:
            try:
                st = status(base, sid)
                errs = 0
            except Exception as e:  # head busy / transient network error
                errs += 1
                st = f"?({type(e).__name__})"
                if errs > 30:
                    print(f"[rayjob] giving up after {errs} consecutive errors: {e}", file=sys.stderr)
                    print(st)
                    sys.exit(1)
            if st in TERMINAL:
                print(st)
                sys.exit(0 if st == "SUCCEEDED" else 1)
            if time.time() - last_print > 120:
                print(f"[rayjob] {sid}: {st} ({int(time.time()-t0)}s elapsed)", file=sys.stderr)
                last_print = time.time()
            time.sleep(20)
        print(f"TIMEOUT({st})")
        sys.exit(1)
    elif cmd == "logs":
        logs = json.loads(_req(f"{base}/api/jobs/{sid}/logs", timeout=600)).get("logs", "")
        if len(sys.argv) > 4:
            with open(sys.argv[4], "w") as f:
                f.write(logs)
            print(f"[rayjob] wrote {len(logs)} bytes to {sys.argv[4]}", file=sys.stderr)
        else:
            sys.stdout.write(logs)
    elif cmd == "submit":
        entrypoint = sys.argv[4]
        env_vars = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
        entry_gpus = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0   # GPUs reserved for the entrypoint
        body = json.dumps({"entrypoint": entrypoint, "submission_id": sid,
                           "runtime_env": {"env_vars": env_vars}, **({"entrypoint_num_gpus": entry_gpus} if entry_gpus > 0 else {})}).encode()
        resp = json.loads(_req(f"{base}/api/jobs/", data=body))
        print(resp.get("submission_id", resp))
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
