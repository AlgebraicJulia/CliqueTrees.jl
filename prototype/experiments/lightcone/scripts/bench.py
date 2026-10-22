"""Per setting benchmark. usage: bench.py data/<setting>.pkl   -> results/<setting>.json
Stages: (1) line-graph sizes of all distinct nets; (2) quantile sample of S distinct nets: cotengra greedy/auto/auto-hq/hyper,
actual CPU (and GPU) contraction when cheap; (3) our methods via Julia + exact; trees built by the ftn bridge."""
import os, sys, json, math, time, pickle, subprocess, threading, collections
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import cotengra as ctg
sys.path.insert(0, "/home/itaykadosh/Projects/GATAS/tn-work/frontier/scripts")
import ftn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JULIA = os.path.expanduser("~/.juliaup/bin/julia"); MTENV = "/home/itaykadosh/Projects/GATAS/tn-work/mtenv"
S_CT = int(os.environ.get("S_CT", 8)); S_EX = int(os.environ.get("S_EX", 5)); EXACT_LIMIT = float(os.environ.get("EXACT_LIMIT", 12))
MAXT = 2500   # skip optimizers on nets with more tensors than this
try:
    import cupy as cp
except Exception:
    cp = None

def relabel(inputs, size):
    inputs = [t for t in inputs if t]
    ids = {}
    for t in inputs:
        for x in t:
            ids.setdefault(x, len(ids))
    ins = [tuple(ftn.sym(ids[x]) for x in t) for t in inputs]
    sz = {ftn.sym(ids[x]): size[x] for x in ids}
    return ins, sz, len(ids)

def linegraph(ins, sz):
    idx = {s: i for i, s in enumerate(sz)}
    E = set()
    for t in ins:
        u = [idx[x] for x in dict.fromkeys(t)]
        for a in range(len(u)):
            for b in range(a + 1, len(u)):
                E.add((min(u[a], u[b]), max(u[a], u[b])))
    return len(idx), E

def tstats(tree):
    nops = len(tree.children)
    return dict(logC=tree.contraction_cost(log=10), W=tree.contraction_width(), cost=tree.contraction_cost(), nops=nops)

import signal
CAP = float(os.environ.get("CT_CAP", 90))
class Cap(BaseException): pass
def _alarm(*a): raise Cap()
signal.signal(signal.SIGALRM, _alarm)
def run_ct(ins, sz, how):
    t = time.time()
    if how in ("auto", "auto-hq"):
        # presets with parallel="auto" would spawn a process pool; keep the search single-threaded (like our Julia runs)
        opt = (ctg.AutoOptimizer if how == "auto" else ctg.AutoHQOptimizer)(parallel=False)
        signal.setitimer(signal.ITIMER_REAL, CAP)
        try:
            tr = ctg.array_contract_tree(ins, (), sz, optimize=opt)
        except Cap:
            return None, time.time() - t
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return tr, time.time() - t
    if how == "hyper":
        tr = ctg.array_contract_tree(ins, (), sz, optimize=ctg.HyperOptimizer(max_time=0.5, parallel=False, progbar=False))
    else:
        tr = ctg.array_contract_tree(ins, (), sz, optimize=how)
    dt = time.time() - t
    return tr, dt

def contract_time(tree, ins, sz, gpu=False):
    rng = np.random.default_rng(0)
    arrs = [(rng.random(tuple(sz[x] for x in t)) + 1j * rng.random(tuple(sz[x] for x in t))) * 0.5 for t in ins]
    if gpu:
        arrs = [cp.asarray(a) for a in arrs]
        tree.contract(arrs, backend="cupy"); cp.cuda.Device().synchronize()   # warmup
        t = time.time(); tree.contract(arrs, backend="cupy"); cp.cuda.Device().synchronize(); return time.time() - t
    t = time.time(); tree.contract(arrs); return time.time() - t

def run_julia(wd):
    p = subprocess.Popen([JULIA, "-t", "2", f"--project={MTENV}", HERE + "/scripts/ours.jl", wd + "/list.txt", wd, str(EXACT_LIMIT)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = []; t0 = time.time(); seen_to = None
    import select
    while True:
        r, _, _ = select.select([p.stdout], [], [], 1.0)
        if r:
            line = p.stdout.readline()
            if not line: break
            log.append(line)
            if line.startswith("FIN"): break
            if "exact TIMEOUT" in line and seen_to is None: seen_to = time.time()
        if seen_to and time.time() - seen_to > 20: break   # a timed-out exact task can block process exit (non-allocating loop): kill
        if time.time() - t0 > 1800: break
        if p.poll() is not None and not r: break
    p.kill()
    if not any(l.startswith("FIN") or "exact TIMEOUT" in l for l in log) : raise RuntimeError("julia failed: " + "".join(log[-20:]))

def main(path):
    d = pickle.load(open(path, "rb"))
    name = os.path.basename(path)[:-4]
    out = dict(name=name, kind=d["kind"], N=d["N"], p=d["p"], nterms=d["nedges"], nsampled=d["nsampled"], ndistinct=d["ndistinct"],
               extract_per_term=d["per_term"], has_gpu=cp is not None)
    counts = d["counts"]
    # 1. sizes of all distinct nets
    nets = []
    for h, (inputs, size) in d["nets"].items():
        ins, sz, n = relabel(inputs, size)
        if not ins:
            out["ntrivial"] = out.get("ntrivial", 0) + counts[h]; continue   # network is a pure scalar: nothing to optimise
        nv, E = linegraph(ins, sz)
        nets.append(dict(h=h, ins=ins, sz=sz, n=nv, m=len(E), ntensors=len(ins), count=counts[h], E=E))
    nets.sort(key=lambda x: (x["n"], x["m"]))
    ns = np.array([x["n"] for x in nets]); ms = np.array([x["m"] for x in nets])
    out["lg_n"] = dict(min=int(ns.min()), med=float(np.median(ns)), max=int(ns.max()), le128=int((ns <= 128).sum()), le512=int((ns <= 512).sum()))
    out["lg_m"] = dict(min=int(ms.min()), med=float(np.median(ms)), max=int(ms.max()))
    # chao1-style estimate of distinct nets in all terms when only a subset of terms was sampled
    c = collections.Counter(counts.values()); f1, f2 = c.get(1, 0), c.get(2, 0)
    unseen = f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2
    out["ndistinct_est"] = d["ndistinct"] if d["nsampled"] >= d["nedges"] else int(round(d["ndistinct"] + min(unseen, d["nedges"] - d["nsampled"])))
    # 2. quantile sample
    S = min(S_CT, len(nets)); groups = np.array_split(np.arange(len(nets)), S)
    samp = []
    for g in groups:
        if len(g) == 0: continue
        i = int(g[len(g) // 2]); x = nets[i]
        x["w_dedup"] = len(g); x["w_all"] = sum(nets[j]["count"] for j in g)
        samp.append(x)
    rows = []
    for x in samp:
        r = dict(h=x["h"], n=x["n"], m=x["m"], ntensors=x["ntensors"], w_dedup=x["w_dedup"], w_all=x["w_all"])
        if x["ntensors"] > MAXT:
            r["skipped"] = "too large"; rows.append(r); continue
        trees = {}
        for how in ("greedy", "auto", "auto-hq", "hyper"):
            tr, dt = run_ct(x["ins"], x["sz"], how)
            if tr is None:   # censored at CT_CAP: search time is a lower bound; take the stats of the greedy tree (conservative for contraction)
                r[how] = dict(t=dt, timeout=True, **tstats(trees["greedy"])); trees[how] = trees["greedy"]
            else:
                r[how] = dict(t=dt, **tstats(tr)); trees[how] = tr
        # contraction measurements on cheap trees
        for how in ("auto", "auto-hq", "hyper"):
            s = r[how]
            if s["logC"] <= 9.3 and s["W"] <= 24:
                s["t_cpu"] = contract_time(trees[how], x["ins"], x["sz"])
                if cp is not None:
                    try: s["t_gpu"] = contract_time(trees[how], x["ins"], x["sz"], gpu=True)
                    except Exception as e: s["gpu_err"] = str(e)[:80]
        r["_trees"] = None
        rows.append(r)
        x["_tree_hq"] = trees["auto-hq"]
    out["ct"] = rows
    # 3. ours on the S_EX smallest-to-median... quantile subset of the sample (keeps size spread)
    ex = [x for x in samp if x["ntensors"] <= MAXT]
    ex = [ex[i] for i in np.unique(np.linspace(0, len(ex) - 1, min(S_EX, len(ex))).round().astype(int))] if ex else []
    wd = f"{HERE}/work/{name}"; os.makedirs(wd, exist_ok=True)
    import shutil; shutil.rmtree(wd); os.makedirs(wd)
    lines = []
    for i, x in enumerate(ex):
        gid = f"g{i}"
        gp = f"{wd}/{gid}.txt"
        with open(gp, "w") as f:
            f.write(f"{x['n']} {len(x['E'])}\n")
            for a, b in sorted(x["E"]): f.write(f"{a+1} {b+1}\n")
        lines.append((gid, gp))
    open(wd + "/list.txt", "w").write("\n".join(f"{a} {b}" for a, b in lines) + "\n")
    ours = []
    if ex:
        run_julia(wd)
        rows_tsv = [l.rstrip("\n").split("\t") for l in open(wd + "/rows.tsv")]
        for i, x in enumerate(ex):
            gid = f"g{i}"; rr = dict(h=x["h"], n=x["n"], m=x["m"])
            for (rid, meth, t, w, lb) in rows_tsv:
                if meth == "exact" and w == "TIMEOUT" and rid == gid: rr["exact"] = dict(timeout=True); rr.setdefault("lb", float(lb)); continue
                if rid != gid: continue
                if w == "TIMEOUT": rr["exact"] = dict(timeout=True); continue
                rr.setdefault("lb", float(lb))
                odf = f"{wd}/{gid}__{meth}.ord"
                order_ = [int(v) - 1 for v in open(odf).read().split()]
                t0 = time.time(); tr = ftn.tree_from_order(x["ins"], (), x["sz"], order_); tb = time.time() - t0
                st = tstats(tr); st["bag"] = float(w); st["t"] = float(t); st["t_tree"] = tb
                if meth == "exact": st["optimal"] = True
                rr[meth] = st
            ours.append(rr)
    out["ours"] = ours
    os.makedirs(f"{HERE}/results", exist_ok=True)
    json.dump(out, open(f"{HERE}/results/{name}.json", "w"))
    print(name, "done: distinct", out["ndistinct"], "lg_n", out["lg_n"], flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
