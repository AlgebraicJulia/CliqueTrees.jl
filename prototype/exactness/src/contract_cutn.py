"""Experiment B, cuTensorNet executor (fair executor: slicing loop runs inside cuTensorNet, on the GPU).

    python contract_cutn.py <pkldir with <net>.json> <out.tsv> [--maxc 13.5] [--reps 3] [--samples 8]
                            [--mem-gb 120] [--item-timeout 900] [--only net1,net2] [--own 1]

Crash-proof: this process is a driver; the real work runs in a child (`--child`) that appends rows to <out.tsv> and
announces every item in <out.tsv>.cur before starting it.  If the child dies (native SIGFPE / SIGSEGV inside
cuTensorNet) or an item exceeds --item-timeout, the driver writes status=crash:<signal> (or timeout) for that item
(with the last Python frame from faulthandler) and relaunches the child, which skips finished items (resumable).
Core dumps are disabled (RLIMIT_CORE=0).

Per network: complex128 host data (seeded by name, tensors scaled by prod d^(-1/(2k)) over summed indices) ->
(1) reference = cuTensorNet complex128 execution of the cheapest tree (C <= 12.5; falls back to the next cheapest
if that one crashes), (2) rescale so |ref| ~ 1 if it is outside (1e-6, 1e6) (avoids complex64 underflow, as in
contractB.py), (3) every tree: path + slicing handed to cuTensorNet (SLICED MODES ONLY, extent 1 per slice; the old
version passed (mode, full extent), i.e. no slicing at all, which gave MemoryLimitExceeded), complex64, autotune,
1 warm-up + `reps` timed runs (median), result vs complex128 reference.  Trees above 10^maxc are skipped.  `--own 1`
also runs cuTensorNet's own path search (+ its own slicing under the memory limit).
Columns: net tag log10C_pred W nslices search_s t_med t_min t_all relerr ref status
"""
import argparse
import faulthandler
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time

COLS = "net\ttag\tlog10C_pred\tW\tnslices\tsearch_s\tt_med\tt_min\tt_all\trelerr\tref\tstatus\n"


def scales(inputs, output, size):
    k = {}
    for t in inputs:
        for i in t:
            k[i] = k.get(i, 0) + 1
    outs = set(output)
    return [math.exp(sum(-math.log(size[i]) / (2 * k[i]) for i in t if i not in outs)) for t in inputs]


def nslices_of(t, size):
    return int(math.prod(size[m] for m in t["sliced"])) if t["sliced"] else 1


def load_items(a):
    items = []  # (net, tag, tree dict or None)
    only = set(a.only.split(",")) if a.only else None
    for f in sorted(os.listdir(a.pkldir)):
        if not f.endswith(".json") or f == "trees.json":
            continue
        d = json.load(open(os.path.join(a.pkldir, f)))
        if only and d["net"] not in only:
            continue
        ts = sorted(d["trees"], key=lambda t: t["log10C"])
        items.append((d, ts))
    items.sort(key=lambda x: x[1][0]["log10C"])
    return items


def done_set(out):
    s = set()
    if os.path.exists(out):
        for line in open(out):
            f = line.rstrip("\n").split("\t")
            if len(f) > 2 and f[0] != "net":
                s.add((f[0], f[1]))
    return s


# ----------------------------------------------------------------------------------------------- child
def child(a):
    faulthandler.enable(file=open(a.out + ".fault", "a"), all_threads=True)
    import cupy as cp
    import numpy as np
    try:
        from cuquantum.tensornet import Network, NetworkOptions, OptimizerOptions
    except ImportError:
        from cuquantum import Network, NetworkOptions, OptimizerOptions
    memopt = NetworkOptions(memory_limit=int(a.mem_gb * 2 ** 30))
    sync = lambda: cp.cuda.Device().synchronize()
    out = open(a.out, "a")
    work = a.out + ".work"
    os.makedirs(work, exist_ok=True)
    done = done_set(a.out)
    bad_ref = set(l.strip() for l in open(a.out + ".badref")) if os.path.exists(a.out + ".badref") else set()

    def mark(s):
        with open(a.out + ".cur", "w") as fh:
            fh.write(s + "\n")

    for d, ts in load_items(a):
        name = d["net"]
        size = {int(k): v for k, v in d["size"].items()}
        inputs, output = d["inputs"], d["output"]
        todo = [t for t in ts if (name, t["tag"]) not in done]
        want_own = a.own and (name, "cutn_own") not in done and ts[0]["log10C"] <= a.maxc
        if not todo and not want_own:
            continue
        if ts[0]["log10C"] > a.maxc:
            for t in todo:
                out.write("\t".join([name, t["tag"], f"{t['log10C']:.4f}", f"{t['W']:.2f}", str(nslices_of(t, size)),
                                     f"{t['search_s']:.3f}", "nan", "nan", "", "nan", "", "skip_cost"]) + "\n")
            out.flush()
            continue
        seed = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
        sc = scales(inputs, output, size)
        host = []
        for j, t in enumerate(inputs):
            rs = np.random.default_rng(seed + j)
            shp = tuple(size[i] for i in t)
            host.append((rs.standard_normal(shp) + 1j * rs.standard_normal(shp)) * (sc[j] / math.sqrt(2)))

        def inter_of(arrs):
            r = []
            for x, t in zip(arrs, inputs):
                r += [x, list(t)]
            r.append(list(output))
            return r

        def run(tree, arrs, tag_for_mark, reps):
            """execute one tree (None = cuTensorNet's own path) with complex arrays arrs; return (value, search, pred, tw, ts)"""
            with Network(*inter_of(arrs), options=memopt) as net:
                t0 = time.time()
                if tree is None:
                    _, info = net.contract_path(optimize=OptimizerOptions(samples=a.samples))
                    search = time.time() - t0
                    pred = math.log10(max(info.opt_cost, 1.0)) - math.log10(8.0)
                else:
                    opt = dict(path=[tuple(p) for p in tree["path"]])
                    if tree["sliced"]:
                        opt["slicing"] = list(tree["sliced"])  # modes only: each slice has extent 1
                    net.contract_path(optimize=OptimizerOptions(**opt))
                    search, pred = tree["search_s"], tree["log10C"]
                net.autotune(iterations=3)
                sync(); t0 = time.time(); r = net.contract(); sync(); tw = time.time() - t0
                tl = []
                for _ in range(reps if tw < 60 else 1):
                    sync(); t0 = time.time(); r = net.contract(); sync(); tl.append(time.time() - t0)
                v = cp.asnumpy(r).astype(np.complex128).ravel()
                return v, search, pred, tw, tl

        # reference (complex128, cheapest tree that runs)
        rf = os.path.join(work, name + ".npz")
        ref, nscale, refkind = None, 1.0, "none"
        if os.path.exists(rf):
            z = np.load(rf, allow_pickle=False)
            ref, nscale, refkind = z["ref"], float(z["nscale"]), str(z["kind"])
        else:
            for t in ts[:3]:
                if t["log10C"] > 12.5 or f"{name}:{t['tag']}" in bad_ref:
                    continue
                mark(f"REF\t{name}\t{t['tag']}")
                try:
                    arrs = [cp.asarray(x) for x in host]
                    v = run(t, arrs, t["tag"], 0)[0]
                    del arrs
                except Exception as e:  # noqa: BLE001  (python-level failure: try next tree)
                    print(name, "ref failed", t["tag"], repr(e)[:100], flush=True)
                    cp.get_default_memory_pool().free_all_blocks()
                    continue
                nr = float(np.linalg.norm(v))
                ref, refkind = v, "c128:" + t["tag"]
                if nr > 0 and not (1e-6 < nr < 1e6):
                    nscale = nr ** (-1.0 / len(inputs))
                    ref = v * nscale ** len(inputs)
                    print(name, f"rescaled |ref|={nr:.3e} factor {nscale:.4f}", flush=True)
                np.savez(rf, ref=ref, nscale=nscale, kind=refkind)
                cp.get_default_memory_pool().free_all_blocks()
                break
        arrs64 = [cp.asarray((x * nscale).astype(np.complex64)) for x in host]
        del host

        jobs = [(t["tag"], t) for t in todo] + ([("cutn_own", None)] if want_own else [])
        for tag, t in jobs:
            if t is not None and t["log10C"] > a.maxc:
                row = [name, tag, f"{t['log10C']:.4f}", f"{t['W']:.2f}", str(nslices_of(t, size)),
                       f"{t['search_s']:.3f}", "nan", "nan", "", "nan", refkind, "skip_cost"]
                out.write("\t".join(row) + "\n"); out.flush()
                continue
            mark(f"RUN\t{name}\t{tag}")
            row = [name, tag, f"{t['log10C']:.4f}" if t else "nan", f"{t['W']:.2f}" if t else "nan",
                   str(nslices_of(t, size)) if t else "nan"]
            try:
                v, search, pred, tw, tl = run(t, arrs64, tag, a.reps)
                if ref is None:
                    ref, refkind, err = v, f"c64:{tag}", 0.0
                else:
                    nr = np.linalg.norm(ref)
                    err = float(np.linalg.norm(v - ref) / nr) if nr > 0 else float(np.linalg.norm(v))
                status = "ok" if np.all(np.isfinite(v)) else "nonfinite"
                if t is None:
                    row[2] = f"{pred:.4f}"
                row += [f"{search:.3f}", f"{np.median(tl):.5f}", f"{min(tl):.5f}",
                        ",".join(f"{x:.4f}" for x in [tw] + tl), f"{err:.3e}", refkind, status]
            except Exception as e:  # noqa: BLE001
                row += ["nan"] * 3 + [""] + ["nan", refkind,
                        "error:" + type(e).__name__ + ":" + str(e)[:100].replace("\t", " ").replace("\n", " ")]
                if t is not None:
                    row[5] = f"{t['search_s']:.3f}"
            out.write("\t".join(row) + "\n"); out.flush()
            print(row, flush=True)
            cp.get_default_memory_pool().free_all_blocks()
        del arrs64
        cp.get_default_memory_pool().free_all_blocks()
    out.close()


# ----------------------------------------------------------------------------------------------- driver
def driver(a):
    if not os.path.exists(a.out):
        open(a.out, "w").write(COLS)
    items = load_items(a)
    meta = {}
    for d, ts in items:
        size = {int(k): v for k, v in d["size"].items()}
        for t in ts:
            meta[(d["net"], t["tag"])] = [f"{t['log10C']:.4f}", f"{t['W']:.2f}", str(nslices_of(t, size)), f"{t['search_s']:.3f}"]
    cur = a.out + ".cur"
    for attempt in range(a.max_restarts):
        if os.path.exists(cur):
            os.remove(cur)
        cmd = [sys.executable, "-I", os.path.abspath(__file__), a.pkldir, a.out, "--child", "--maxc", str(a.maxc),
               "--reps", str(a.reps), "--samples", str(a.samples), "--mem-gb", str(a.mem_gb), "--own", str(a.own)]
        if a.only:
            cmd += ["--only", a.only]
        p = subprocess.Popen(cmd, preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
        last, tstart = None, time.time()
        timed_out = False
        while p.poll() is None:
            time.sleep(2)
            s = open(cur).read().strip() if os.path.exists(cur) else None
            if s != last:
                last, tstart = s, time.time()
            if s and time.time() - tstart > (a.own_timeout if s.endswith('cutn_own') else a.item_timeout):
                timed_out = True
                p.kill(); p.wait()
        rc = p.returncode
        if rc == 0:
            print("driver: child finished", flush=True)
            return
        s = open(cur).read().strip() if os.path.exists(cur) else ""
        kind, net, tag = (s.split("\t") + ["", "", ""])[:3]
        sig = signal.Signals(-rc).name if rc < 0 else f"exit{rc}"
        fault = ""
        if os.path.exists(a.out + ".fault"):
            lines = [l.strip() for l in open(a.out + ".fault") if "File" in l]
            fault = lines[0].replace("\t", " ")[:140] if lines else ""
            open(a.out + ".fault", "w").close()
        status = "timeout" if timed_out else f"crash:{sig}" + (f":{fault}" if fault else "")
        print(f"driver: child died ({status}) at {s!r}; relaunch", flush=True)
        if kind == "REF":
            open(a.out + ".badref", "a").write(f"{net}:{tag}\n")  # next attempt tries the next tree as reference
        elif kind == "RUN":
            m = meta.get((net, tag), ["nan", "nan", "nan", "nan"])
            row = [net, tag] + m[:3] + [m[3], "nan", "nan", "", "nan", "", status]
            with open(a.out, "a") as fh:
                fh.write("\t".join(row) + "\n")
        elif not kind:
            print("driver: died before any item; aborting", flush=True)
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pkldir"); ap.add_argument("out")
    ap.add_argument("--maxc", type=float, default=13.5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--mem-gb", type=float, default=120.0)
    ap.add_argument("--item-timeout", type=float, default=900.0)
    ap.add_argument("--own-timeout", type=float, default=300.0)
    ap.add_argument("--max-restarts", type=int, default=400)
    ap.add_argument("--only", default=None)
    ap.add_argument("--own", type=int, default=1)
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    child(a) if a.child else driver(a)
