"""Experiment B, GPU side: execute every tree of every network (pickles from costB.py) on one GPU.

    python -I contractB.py <pkldir> <out.tsv> [--maxc 13.5] [--reps 3] [--backend torch|cupy] [--budget 30000]

Per network: one random complex64 data set (seeded by the network name; tensor t is scaled by
prod_i d_i^(-1/(2 k_i)) over its summed indices i, k_i = number of tensors on i, so a random sum over an index
keeps magnitudes ~1). Per tree: one warm-up run, then `reps` timed runs (cuda-synchronized wall, median), peak
device memory of a timed run, the result. Trees are skipped above 10^maxc predicted multiply-adds; a tree whose
warm-up took > 60 s gets 1 timed run. Agreement: relative 2-norm difference of each result to the complex128
result of the cheapest tree (when its log10 C <= 12.5), else to the first tree's complex64 result.
Columns: net tag log10C W nslices ncontract search_s t_med t_min t_all peak_GB relerr ref status
"""
import argparse
import hashlib
import math
import os
import pickle
import sys
import time

import numpy as np


def scales(inputs, output, size):
    k = {}
    for t in inputs:
        for i in t:
            k[i] = k.get(i, 0) + 1
    outs = set(output)
    sc = []
    for t in inputs:
        e = 0.0
        for i in t:
            if i not in outs:
                e -= math.log(size[i]) / (2 * k[i])
        sc.append(math.exp(e))
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkldir"); ap.add_argument("out")
    ap.add_argument("--maxc", type=float, default=13.5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--backend", default="torch")
    ap.add_argument("--budget", type=float, default=30000.0, help="stop starting new networks after this many s")
    ap.add_argument("--only", default=None, help="comma-separated network names; net:tag1+tag2 restricts to those trees")
    ap.add_argument("--device", default="cuda", help="torch device (cpu only for testing)")
    a = ap.parse_args()

    if a.backend == "torch":
        import torch
        dev = torch.device(a.device)
        oncuda = dev.type == "cuda"

        def sync():
            if oncuda:
                torch.cuda.synchronize()

        def randc(shape, seed, dtype):
            g = torch.Generator(device=dev); g.manual_seed(seed)
            return torch.randn(tuple(shape), dtype=dtype, device=dev, generator=g)

        C64, C128 = torch.complex64, torch.complex128
        reset_peak = torch.cuda.reset_peak_memory_stats if oncuda else (lambda: None)
        peak = (lambda: torch.cuda.max_memory_allocated() / 2 ** 30) if oncuda else (lambda: float("nan"))
        tonp = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
        cast = lambda x, dt: x.to(dt)
        free = torch.cuda.empty_cache if oncuda else (lambda: None)
        OOM = torch.cuda.OutOfMemoryError
        print("torch", torch.__version__, torch.cuda.get_device_name(0) if oncuda else "cpu", flush=True)
    else:
        import cupy as cp

        def sync():
            cp.cuda.Device().synchronize()

        def randc(shape, seed, dtype):
            rs = np.random.default_rng(seed)   # host-side draw (no curand needed), then copied
            x = (rs.standard_normal(tuple(shape)) + 1j * rs.standard_normal(tuple(shape))) / math.sqrt(2)
            return cp.asarray(x.astype(np.complex64 if dtype == cp.complex64 else np.complex128))

        C64, C128 = cp.complex64, cp.complex128
        pool = cp.get_default_memory_pool()
        reset_peak = lambda: None
        peak = lambda: pool.total_bytes() / 2 ** 30
        tonp = lambda x: cp.asnumpy(x)
        cast = lambda x, dt: x.astype(dt)
        free = pool.free_all_blocks
        OOM = cp.cuda.memory.OutOfMemoryError

    new = not os.path.exists(a.out)
    out = open(a.out, "a")
    if new:
        out.write("net\ttag\tlog10C\tW\tnslices\tncontract\tsearch_s\tt_med\tt_min\tt_all\tpeak_GB\trelerr\tref\tstatus\n")
    done = set()
    if not new:
        for line in open(a.out):
            f = line.split("\t")
            if len(f) > 2 and f[0] != "net":
                done.add((f[0], f[1]))

    # the pickles are our own (written by costB.py in this pipeline), so loading them is safe
    files = sorted(f for f in os.listdir(a.pkldir) if f.endswith(".pkl"))
    nets = []
    for f in files:
        with open(os.path.join(a.pkldir, f), "rb") as fh:
            d = pickle.load(fh)
        if a.only and d["net"] not in [x.split(":")[0] for x in a.only.split(",")]:
            continue
        cmin = min(t.contraction_cost(log=10) for _, t, _, _ in d["trees"])
        nets.append((cmin, d))
    nets.sort(key=lambda x: x[0])
    t_start = time.time()
    for cmin, d in nets:
        if time.time() - t_start > a.budget:
            print("budget reached; stop before", d["net"], flush=True)
            break
        net, inputs, output, size = d["net"], d["inputs"], d["output"], d["size"]
        seed = int(hashlib.md5(net.encode()).hexdigest()[:8], 16)
        sc = scales(inputs, output, size)
        arrays = [randc([size[i] for i in t], seed + j, C64) * sc[j] for j, t in enumerate(inputs)]
        trees = sorted(d["trees"], key=lambda x: x[1].contraction_cost())
        # reference: complex128 on the cheapest tree
        ref, refkind = None, "none"
        best = trees[0][1]
        if best.contraction_cost(log=10) <= 12.5:
            try:
                a128 = [cast(x, C128) for x in arrays]
                ref = tonp(best.contract(a128)).astype(np.complex128).ravel()
                refkind = "c128"
                del a128
            except OOM:
                ref = None
            free()
        # rescale so that |result| ~ 1 when the c128 reference is far from 1 (random loopy networks: product of many
        # random sums is log-normal, true values 1e-50..1e-80 underflow complex64 to exactly 0 -> relerr 1.0 artefact)
        nscale = 1.0
        if ref is not None and refkind == "c128":
            nr = float(np.linalg.norm(ref))
            if nr > 0 and not (1e-6 < nr < 1e6):
                nscale = nr ** (-1.0 / len(inputs))
                arrays = [x * nscale for x in arrays]
                ref = ref * nscale ** len(inputs)
                print(net, f"rescaled: |ref|={nr:.3e}, per-tensor factor {nscale:.4f}", flush=True)
        only_tags = None
        for x in (a.only or "").split(","):
            if x.split(":")[0] == net and ":" in x:
                only_tags = set(x.split(":")[1].split("+"))
        for tag, tree, search_s, info in d["trees"]:
            if (net, tag) in done or (only_tags is not None and tag not in only_tags):
                continue
            lc = tree.contraction_cost(log=10)
            row = [net, tag, f"{lc:.4f}", f"{tree.contraction_width():.2f}", str(tree.multiplicity),
                   str((tree.N - 1) * tree.multiplicity), f"{search_s:.3f}"]
            if lc > a.maxc:
                out.write("\t".join(row + ["nan"] * 5 + ["", "skip_cost"]) + "\n"); out.flush()
                continue
            run = lambda: tree.contract(arrays)
            fbtag = ""
            try:
                try:
                    sync(); t0 = time.time(); r = run(); sync(); tw = time.time() - t0
                except RuntimeError as e:
                    if "too many" not in str(e) or a.backend != "torch":
                        raise
                    # torch tensors/permutes are limited to 25 dims: execute this tree with cupy (cuTENSOR einsum, up to 64 modes)
                    import cupy as cp
                    carr = [cp.from_dlpack(x.contiguous()) for x in arrays]
                    run = lambda: torch.as_tensor(tree.contract(carr, backend="cupy"), device=dev)
                    fbtag = ";cupy"
                    free(); sync(); t0 = time.time(); r = run(); sync(); tw = time.time() - t0
                reps = a.reps if tw < 60 else 1
                ts = []
                for _ in range(reps):
                    del r; free(); reset_peak()
                    sync(); t0 = time.time(); r = run(); sync(); ts.append(time.time() - t0)
                pk = peak()
                v = tonp(r).astype(np.complex128).ravel()
                del r
                if ref is None:
                    ref, refkind = v, f"c64:{tag}"
                    err = 0.0
                else:
                    nr = np.linalg.norm(ref)
                    err = float(np.linalg.norm(v - ref) / nr) if nr > 0 else float(np.linalg.norm(v))
                status = "ok" if np.all(np.isfinite(v)) else "nonfinite"
                row += [f"{np.median(ts):.5f}", f"{min(ts):.5f}", ",".join(f"{x:.4f}" for x in [tw] + ts),
                        f"{pk:.3f}", f"{err:.3e}", refkind + fbtag, status]
            except OOM:
                row += ["nan"] * 5 + [refkind, "oom"]
            except Exception as e:  # noqa: BLE001
                row += ["nan"] * 5 + [refkind, "error:" + type(e).__name__ + ":" + str(e)[:80].replace("\t", " ")]
            free()
            out.write("\t".join(row) + "\n"); out.flush()
            print(net, tag, row[7:], flush=True)
        del arrays
        free()
    out.close()


if __name__ == "__main__":
    main()
