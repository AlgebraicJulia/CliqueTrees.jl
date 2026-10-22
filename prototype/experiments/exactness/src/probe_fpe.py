"""Probe the SIGFPE in cuTensorNet contract_path on one network: python probe_fpe.py <net.json>"""
import faulthandler, json, subprocess, sys, time
if len(sys.argv) > 3:
    faulthandler.enable()
    import cupy as cp, numpy as np
    from cuquantum.tensornet import Network, OptimizerOptions
    d = json.load(open(sys.argv[1])); size = {int(k): v for k, v in d["size"].items()}
    kw = json.loads(sys.argv[3])
    ops = []
    for t in d["inputs"]:
        ops += [cp.ones([size[i] for i in t], dtype=cp.complex64) * 0.1, list(t)]
    ops.append(list(d["output"]))
    with Network(*ops) as net:
        t0 = time.time(); _, info = net.contract_path(optimize=OptimizerOptions(**kw))
        print("OK", kw, f"{time.time()-t0:.1f}s", "slices", info.num_slices, flush=True)
    sys.exit(0)
for kw in [dict(samples=8), dict(samples=1), dict(samples=8, threads=4), dict()]:
    try:
        p = subprocess.run([sys.executable, "-I", __file__, sys.argv[1], "x", json.dumps(kw)], capture_output=True, text=True, timeout=300)
        print(kw, "rc", p.returncode, p.stdout.strip()[-200:], "|", " ".join(l.strip() for l in p.stderr.splitlines() if "File" in l)[:300], flush=True)
    except subprocess.TimeoutExpired:
        print(kw, "TIMEOUT 300s", flush=True)
