"""cuTensorNet path-finder baseline worker (cuquantum-python 26.9 / cutensornet 2.14, cupy).
usage: python cutn_worker.py <queue> <outdir> --threads K --tag W [--exec 1] [--keepalive 0]
queue line: <source: einsum|frontier> <network> <maxT seconds> <cap W log2 elements or 0> [memf, default 1]
For each task: ladder of hyper-sample counts S = 1,2,4,... (cuTensorNet has no time budget: search time is set by
OptimizerOptions.samples), each run timed and scored; the ladder stops once a run exceeds maxT. Also one run with the
DEFAULT OptimizerOptions() (samples=0). Budget T of the analysis = best (min logC) row with search_s <= T.
cap 0: slicing disabled (SlicerOptions(disable_slicing=True)): the UNSLICED path. cap W>0: NetworkOptions(memory_limit =
2^W * 4 B * memf), so cuTensorNet slices until its workspace model fits; its model counts workspace (several tensors), so the
recomputed per-slice width is typically below W: the queue runs memf=1,2,4 and the analysis keeps rows with width<=W ('cap_met').
Scoring (units = cotengra 0.8.2, same as einsumbench/frontier): the cuTensorNet path (+ its sliced modes) is rebuilt as a
cotengra ContractionTree; logC = log10 tree.contraction_cost() (scalar multiply-adds summed over all pairwise
contractions and all slices), width = log2 largest intermediate per slice, maxop = log2 largest pairwise contraction.
cuTensorNet's own opt_cost (FLOPs) and largest_intermediate are logged for reference (cutn_log10flops, cutn_log2maxint).
"""
import argparse, fcntl, math, os, sys, threading, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("queue"); ap.add_argument("outdir")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--tag", default=str(os.getpid()))
ap.add_argument("--exec", type=int, default=0)
ap.add_argument("--keepalive", type=int, default=0)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("CTGPLUGIN", os.path.join(HERE, "..", "ctgplugin")))
sys.path.insert(0, os.environ.get("FTN_DIR", os.path.join(HERE, "..", "frontier", "scripts")))
import cotengra as ctg  # noqa: E402
import cupy as cp  # noqa: E402
from cuquantum.tensornet import (Network, NetworkOptions, OptimizerOptions, SlicerOptions)  # noqa: E402
import tnc  # noqa: E402
import ftn  # noqa: E402

ITEM = 4  # float32 operands


def load(src, name):
    return tnc.load(name) if src == "einsum" else ftn.load(name)


def pop(path):
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path) as f:
            lines = f.readlines()
        if not lines:
            return None
        with open(path, "w") as f:
            f.writelines(lines[1:])
        return lines[0].split()


def keepalive():
    x = cp.random.rand(4096, 4096, dtype=cp.float32)
    while True:
        t = time.time()
        for _ in range(40):
            x = (x @ x) * 1e-4
        cp.cuda.Stream.null.synchronize()
        time.sleep(max(0.0, 3 * (time.time() - t)))


def build_operands(inputs, size):
    ops = []
    for t in inputs:
        n = int(np.prod([size[i] for i in t], dtype=np.float64))
        if n > 2 ** 27:
            raise MemoryError("operand too large")
        ops.append(cp.ones([size[i] for i in t], dtype=cp.float32) * 0.5)
    return ops


def ctg_score(inputs, output, size, path, slices, idmap):
    tree = ctg.ContractionTree.from_path(inputs, output, size, path=path)
    inv = {v: k for k, v in idmap.items()}
    for m in slices:
        tree.remove_ind(inv[m], inplace=True)
    mo = max((tree.get_flops(nd) for nd, _, _ in tree.traverse()), default=1)
    return tree.contraction_cost(log=10), tree.contraction_width(), math.log2(max(mo, 1)), tree.multiplicity


os.makedirs(os.path.join(a.outdir, "res"), exist_ok=True)
out = open(os.path.join(a.outdir, "res", f"cutn_t{a.threads}_{a.tag}.tsv"), "a")
COLS = "source network cap threads samples seed status search_s logC width maxop nslices nsliced cap_met cutn_log10flops cutn_log2maxint exec_s extra".split()
if out.tell() == 0:
    out.write("\t".join(COLS) + "\n")
if a.keepalive:
    threading.Thread(target=keepalive, daemon=True).start()


def row(**k):
    d = dict(source="", network="", cap=0, threads=a.threads, samples="", seed=1, status="ok", search_s="nan", logC="nan",
             width="nan", maxop="nan", nslices=1, nsliced=0, cap_met="", cutn_log10flops="nan", cutn_log2maxint="nan",
             exec_s="", extra="")
    d.update(k)
    out.write("\t".join(str(d[c]) for c in COLS) + "\n"); out.flush()
    print(time.strftime("%T"), {c: d[c] for c in ("network", "cap", "samples", "status", "search_s", "logC", "width", "nslices")}, flush=True)


def run_one(src, name, cap, memf, S, inputs, output, size, idmap, ops, T):
    modes = [[idmap[i] for i in t] for t in inputs]
    omodes = [idmap[i] for i in output]
    inter = []
    for o, ms in zip(ops, modes):
        inter += [o, ms]
    inter.append(omodes)
    nopt = {"device_id": 0}
    if cap:
        nopt["memory_limit"] = int(2 ** cap * ITEM * memf)
    kw = {} if cap else {"slicing": SlicerOptions(disable_slicing=True)}
    opt = OptimizerOptions(samples=S, threads=a.threads, seed=1, **kw) if S is not None else OptimizerOptions(**kw)
    with Network(*inter, options=NetworkOptions(**nopt)) as net:
        t0 = time.time()
        fb = ""
        try:
            path, info = net.contract_path(optimize=opt)
        except Exception as e:  # noqa: BLE001
            if cap or not any(k in repr(e) for k in ("MemoryLimit", "ALL_HYPER_SAMPLES_FAILED")):
                raise
            # unsliced path needs more workspace than the device has: retry with cuTensorNet's default slicing (flagged)
            fb = ";fallback_sliced=1"
            opt = OptimizerOptions(samples=S, threads=a.threads, seed=1) if S is not None else OptimizerOptions()
            path, info = net.contract_path(optimize=opt)
        dt = time.time() - t0
        sl = [m for m, _ in info.slices] if info.slices else []
        sl = [int(m) for m in sl]
        c, w, mo, ns = ctg_score(inputs, output, size, list(path), sl, idmap)
        ex = ""
        if a.exec and c < 11.5 and w < 29 and ns <= 8:
            try:
                net.autotune(iterations=1)
                cp.cuda.Stream.null.synchronize(); t1 = time.time()
                if ns == 1:
                    net.contract()
                else:
                    net.contract(slices=range(min(ns, 8)))   # time of 8 slices (extrapolate x ns/8)
                cp.cuda.Stream.null.synchronize(); ex = round(time.time() - t1, 4)
            except Exception as e:  # noqa: BLE001
                ex = "err:" + repr(e)[:60]
        row(source=src, network=name, cap=cap, samples="default" if S is None else S, search_s=round(dt, 4),
            logC=round(c, 4), width=round(w, 3), maxop=round(mo, 3), nslices=ns, nsliced=len(sl),
            cap_met=(int(w <= cap + 1e-9) if cap else ""), cutn_log10flops=round(math.log10(max(info.opt_cost, 1)), 4),
            cutn_log2maxint=round(math.log2(max(info.largest_intermediate, 1)), 3), exec_s=ex, extra=f"memf={memf}{fb}")
        return dt


MARK = os.path.join(a.outdir, f"cur_{a.tag}.txt")   # in-flight item; survives a native crash (SIGFPE) of the process
CAP_S = 2 ** 30                                      # cuTensorNet samples is int32: 2^31 overflows


def mark(src, name, cap, memf, S):
    with open(MARK, "w") as f:
        f.write(f"{src}\t{name}\t{cap}\t{memf}\t{'default' if S is None else S}\n")


if os.path.exists(MARK):   # previous incarnation of this worker died natively while on this item
    p = open(MARK).read().strip().split("\t")
    if len(p) == 5:
        row(source=p[0], network=p[1], cap=int(p[2]), samples=p[4], status="crash", extra=f"memf={p[3]};worker_died_after_samples={p[4]}")
    os.remove(MARK)

cache = {}
while True:
    t = pop(a.queue)
    if t is None:
        break
    src, name, maxT, cap = t[0], t[1], float(t[2]), int(t[3])
    memf = float(t[4]) if len(t) > 4 else 1.0
    try:
        if name not in cache:
            cache.clear()
            inputs, output, size = load(src, name)
            idmap = {}
            for tt in list(inputs) + [output]:
                for i in tt:
                    idmap.setdefault(i, len(idmap))
            cache[name] = (inputs, output, size, idmap, build_operands(inputs, size))
        inputs, output, size, idmap, ops = cache[name]
        S, nerr, prev_dt = None, 0, 0.0
        while True:
            mark(src, name, cap, memf, S)
            try:
                dt = run_one(src, name, cap, memf, S, inputs, output, size, idmap, ops, maxT)
                nerr = 0
            except Exception as e:  # noqa: BLE001
                row(source=src, network=name, cap=cap, samples="default" if S is None else S, extra=f"memf={memf}",
                    status=("error:" + repr(e))[:150].replace("\t", " "))
                dt, nerr = 0.0, nerr + 1
                if nerr >= 3:
                    break
            if S is not None and dt > maxT:
                break
            if S is not None and S >= 1024 and dt < 0.2:    # tiny network: search time does not grow with samples
                break
            if S is not None and S >= CAP_S:
                break
            if S is not None and S >= 256 and prev_dt > 0 and dt < 1.3 * prev_dt:   # search time stopped growing
                break
            if S is not None:
                prev_dt = dt
            S = 1 if S is None else min(S * (4 if dt < maxT / 16 else 2), CAP_S)
    except Exception as e:  # noqa: BLE001
        row(source=src, network=name, cap=cap, status=("error:" + repr(e))[:150].replace("\t", " "))
    if os.path.exists(MARK):
        os.remove(MARK)
