"""cuTensorNet's own path finder on the B networks under the same 2^30-element memory cap, ~60 s of search, all job threads.
    python cutn_trees.py <srcdir with <net>.json> <outdir> [threads] [budget_s]
Per network (child process per net, crash/timeout-proof): hyper-sample ladder S=1,2,4,.. (memf=2) until a run takes >= budget/2 s, then
a final run at S*budget/dt, then the same final S with memf 1 and 4.  memory_limit = 2^30 * 8 B (complex64) * memf (cuTensorNet's
workspace model counts several tensors, so memf in {1,2,4} as in baselines/tn/code/cutn_worker.py).  Every run is rescored with cotengra
(sliced modes removed); the tree with width <= 30 and minimal log10C among runs with search_s <= 1.25*budget is written as tag cutn60
(search_s = that run's time) into <outdir>/<net>.json (copy of the src json with trees ours60, ctg_hyper60, cutn60).
All candidate runs: <outdir>/cutn_paths.tsv.  A background thread keeps the GPU busy (HPG kills idle-GPU jobs)."""
import json, math, os, subprocess, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 30


def child(src, outdir, net, threads, budget):
    import cupy as cp
    import cotengra as ctg
    from cuquantum.tensornet import Network, NetworkOptions, OptimizerOptions
    js = json.load(open(f"{src}/{net}.json"))
    inputs, output = js["inputs"], js["output"]
    size = {int(k): v for k, v in js["size"].items()}
    sym = ctg.get_symbol
    cin = [tuple(sym(i) for i in t) for t in inputs]; cout = tuple(sym(i) for i in output)
    csize = {sym(k): v for k, v in size.items()}
    ops = [cp.ones([size[i] for i in t], dtype=cp.complex64) * 0.5 for t in inputs]
    inter = []
    for o, t in zip(ops, inputs):
        inter += [o, list(t)]
    inter.append(list(output))
    tsv = open(f"{outdir}/cutn_paths.tsv", "a")
    cands = []

    def one(S, memf):
        try:
            return one_(S, memf)
        except Exception as e:  # MemoryLimitExceeded, ALL_HYPER_SAMPLES_FAILED, ...
            print(net, S, memf, "FAILED", type(e).__name__, str(e)[:200], flush=True)
            return float("inf")

    def one_(S, memf):
        opts = NetworkOptions(memory_limit=int(2 ** TARGET * 8 * memf))
        with Network(*inter, options=opts) as n:
            t0 = time.time()
            path, info = n.contract_path(optimize=OptimizerOptions(samples=S, threads=threads, seed=1))
            dt = time.time() - t0
        sl = [int(m) for m, _ in info.slices] if info.slices else []
        tree = ctg.ContractionTree.from_path(cin, cout, csize, path=[tuple(p) for p in path])
        for m in sl:
            tree.remove_ind(sym(m), inplace=True)
        d = dict(tag="cutn60", path=[list(p) for p in path], sliced=sl,
                 log10C=tree.contraction_cost(log=10), W=tree.contraction_width(), search_s=dt)
        tsv.write(f"{net}\t{S}\t{memf}\t{d['log10C']:.4f}\t{d['W']:.2f}\t{tree.multiplicity}\t{dt:.2f}\n"); tsv.flush()
        print(net, S, memf, d["log10C"], d["W"], tree.multiplicity, dt, flush=True)
        cands.append(d)
        return dt

    S = 1
    while True:
        dt = one(S, 2)
        if dt >= budget / 2 or S >= 2 ** 30 or (S >= 1024 and dt < 0.2):
            break
        S *= 2
    Sf = int(min(2 ** 30, max(1, S * budget / max(dt, 1e-3) * 0.9))) if dt < budget else S
    if Sf != S:
        one(Sf, 2)
    for mf in (1, 4):
        one(Sf, mf)
    if not cands:
        print(net, "no cuTensorNet path", flush=True)
        return
    ok = [c for c in cands if c["search_s"] <= 1.25 * budget] or cands
    capped = [c for c in ok if c["W"] <= TARGET + 1e-9] or ok
    best = min(capped, key=lambda c: (c["W"] > TARGET, c["log10C"]))
    js["trees"] = [t for t in js["trees"] if t["tag"] in ("ours60", "ctg_hyper60")] + [best]
    json.dump(js, open(f"{outdir}/{net}.json.tmp", "w"), default=int); os.replace(f"{outdir}/{net}.json.tmp", f"{outdir}/{net}.json")


def keepalive():
    import cupy as cp
    x = cp.random.rand(4096, 4096, dtype=cp.float32)
    while True:
        t = time.time()
        for _ in range(40):
            x = (x @ x) * 1e-4
        cp.cuda.Stream.null.synchronize()
        time.sleep(max(0.0, 2 * (time.time() - t)))


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(*sys.argv[2:4], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]))
    else:
        src, outdir = sys.argv[1], sys.argv[2]
        threads = int(sys.argv[3]) if len(sys.argv) > 3 else 32
        budget = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
        os.makedirs(outdir, exist_ok=True)
        threading.Thread(target=keepalive, daemon=True).start()
        nets = [l.strip() for l in open(os.path.join(HERE, "..", "queueB.txt")) if l.strip() and os.path.exists(f"{src}/{l.strip()}.json")]
        for net in nets:
            if os.path.exists(f"{outdir}/{net}.json"):
                continue
            try:
                r = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--child", src, outdir, net, str(threads), str(budget)],
                                   timeout=1500, capture_output=True, text=True)
                print(net, "rc", r.returncode, (r.stderr or "")[-300:], flush=True)
            except subprocess.TimeoutExpired:
                print(net, "TIMEOUT", flush=True)
                open(f"{outdir}/cutn_paths.tsv", "a").write(f"{net}\t-\t-\tnan\tnan\tnan\ttimeout\n")
