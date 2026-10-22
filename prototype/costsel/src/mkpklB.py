"""Pickles for the real B200 contraction (exactness contractB.py) of the networks where cost-aware selection
changed the tree.

    python -I mkpklB.py <queueB24> <pooldir> <gpudir> <rcdir> <resB trees dir> <outdir> [mindiff=0.01]

Per network: the old trees ours60, ctg_hyper60, td_exact/td_hyb300 (re-timed in the same job) plus
  csel      = br+ctg tree of the TD selected by br cost among all our TDs (widths K..K+2)
  csel_gpu  = the cheapest GPU-polished tree (+ctg)
Each new tree is sliced like costB (slice_and_reconfigure to 2^30 if wider). A network is written only if
min(csel, csel_gpu) differs from ours60 by more than mindiff log10 C (after slicing).
search_s of csel = generation time of the pool (t_cum of the winner's group); of csel_gpu = that + the GPU
job's time for the network (scoring + mass search + polish).
"""
import os
import pickle
import sys

sys.path.insert(0, os.environ.get("EXACT_SRC", "/blue/fairbanksj/itaykadosh/gpu/exactness/src"))
import cotengra as ctg  # noqa: E402
import ftn  # noqa: E402
import time  # noqa: E402


def post(tree, target):
    """= costB.post: cotengra subtree_reconfigure, then slice_and_reconfigure to 2^target if wider."""
    t0 = time.time()
    r = ftn.refine(tree)
    if r.contraction_width() > target:
        r = r.slice_and_reconfigure(target_size=2 ** target, minimize="flops")
    return r, time.time() - t0


def pairs(s):
    v = [int(x) for x in s.split()]
    return [(v[k], v[k + 1]) for k in range(0, len(v), 2)]


def main():
    qf, pooldir, gpudir, rcdir, bdir, outdir = sys.argv[1:7]
    mind = float(sys.argv[7]) if len(sys.argv) > 7 else 0.01
    os.makedirs(outdir, exist_ok=True)
    for net in [l.strip() for l in open(qf) if l.strip()]:
        pp = os.path.join(pooldir, net + ".pool"); bp = os.path.join(bdir, net + ".pkl")
        if not (os.path.exists(pp) and os.path.exists(bp)):
            print("skip (missing)", net); continue
        with open(bp, "rb") as f:   # our own pickle, written by costB.py in this pipeline (trusted)
            d = pickle.load(f)
        inputs, output, size, target = d["inputs"], d["output"], d["size"], d["target"]
        ls = open(pp).read().split("\n")
        ssa, sel = {}, None
        for i, l in enumerate(ls):
            x = l.split("\t")
            if x[0] == "SSA":
                ssa[x[1]] = pairs(ls[i + 1])
            elif x[0] == "SEL" and x[1] == "all":
                sel = x   # last = largest N
        if sel is None or sel[5] not in ssa:
            print("skip (no selection)", net); continue
        tgen = float(sel[4])
        new = []
        br = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=ssa[sel[5]])
        t, s = post(br, target)
        new.append(("csel", t, tgen + s, dict(td=sel[5])))
        gp = os.path.join(gpudir, net + ".gpu")
        if os.path.exists(gp):
            gl = open(gp).read().split("\n")
            pols = []
            for k, l in enumerate(gl):
                x = l.split("\t")
                if x[0] == "POL":
                    pols.append((float(x[4]), float(x[5]), x[1], pairs(gl[k + 1])))
            if pols:
                b = min(pols)
                t, s = post(ctg.ContractionTree.from_path(inputs, output, size, ssa_path=b[3]), target)
                gsec = sum(p[1] for p in pols) + 20.0   # polish of all three + mass search 10 s + scoring
                new.append(("csel_gpu", t, tgen + gsec + s, dict(pol=b[2])))
        old = [x for x in d["trees"] if x[0] in ("ours60", "ctg_hyper60", "td_exact", "td_hyb300")]
        o60 = [x[1].contraction_cost(log=10) for x in old if x[0] == "ours60"]
        cn = min(x[1].contraction_cost(log=10) for x in new)
        if o60 and abs(cn - o60[0]) <= mind:
            print(f"skip (same C as ours60) {net}: {cn:.3f} vs {o60[0]:.3f}"); continue
        print(f"{net}: new {[round(x[1].contraction_cost(log=10), 3) for x in new]} old "
              f"{[(x[0], round(x[1].contraction_cost(log=10), 3)) for x in old]}", flush=True)
        with open(os.path.join(outdir, net + ".pkl"), "wb") as f:
            pickle.dump(dict(net=net, inputs=inputs, output=output, size=size, target=target, trees=old + new), f)


if __name__ == "__main__":
    main()
