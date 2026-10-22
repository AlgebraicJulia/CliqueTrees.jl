"""Experiment B, CPU side: final contraction trees of one network for the GPU runs.

    python -I costB.py <tdfile (mode B)> <outdir> [target_log2=30]

Trees (all re-costed by cotengra; search_s = wall seconds of everything that made the tree):
  ours10, ours60      hybridcost(HybridTW(T), refine=branchrefine) (Julia, 1 thread) + cotengra
                      subtree_reconfigure (+ slice_and_reconfigure to 2^target if wider)
  td_MF1, td_hbt1, td_exact / td_hyb300
                      TD ladder: the ordering -> branchtree + branchrefine -> the same post-processing
  ctg_hyper10/60      cotengra HyperOptimizer(max_time=T, parallel=False, minimize='flops'); when the greedy
                      tree is wider than the target, slicing_reconf_opts={'target_size': 2^target} (cotengra's
                      own sliced search); else the same post-processing as ours
  ctg_greedy          cotengra greedy + the same post-processing
Writes <outdir>/<net>.pkl (inputs/output/size + trees) and appends one line per tree to <outdir>/trees.tsv.
"""
import math
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cotengra as ctg  # noqa: E402
import ftn  # noqa: E402
import tdfile  # noqa: E402
import tnc  # noqa: E402


def post(tree, target):
    t0 = time.time()
    r = ftn.refine(tree)
    if r.contraction_width() > target:
        r = r.slice_and_reconfigure(target_size=2 ** target, minimize="flops")
    return r, time.time() - t0


def main():
    tdpath, outdir = sys.argv[1], sys.argv[2]
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    td = tdfile.parse(tdpath)
    net = td["net"]["name"]
    inputs, output, size = tnc.load(net)
    trees = []   # (tag, tree, search_s, info)
    for d in td["ours"]:
        t = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=d["ssa"])
        r, s = post(t, target)
        trees.append((d["tag"], r, d["seconds"] + s, dict(source=d["source"], tdwidth=d["tdwidth"],
                                                          tdlower=d["tdlower"], certified=d["certified"])))
    for d in td["orders"]:
        t = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=d["ssa_br"])
        r, s = post(t, target)
        tag = "td_" + {"hbt1.0_s1": "hbt1"}.get(d["tag"], d["tag"])
        trees.append((tag, r, d["seconds"] + d["br_seconds"] + s, dict(tdwidth=d["width"])))
    t0 = time.time()
    g = ctg.array_contract_tree(inputs, output, size, optimize="greedy")
    gw = g.contraction_width()
    r, s = post(g, target)
    trees.append(("ctg_greedy", r, time.time() - t0, {}))
    for T in (10, 60):
        t0 = time.time()
        kw = dict(slicing_reconf_opts={"target_size": 2 ** target}) if gw > target else {}
        opt = ctg.HyperOptimizer(max_time=T, parallel=False, minimize="flops", progbar=False, max_repeats=10 ** 6, **kw)
        h = opt.search(inputs, output, size)
        if not kw:
            h, _ = post(h, target)
        trees.append((f"ctg_hyper{T}", h, time.time() - t0, dict(sliced_search=bool(kw))))
    with open(os.path.join(outdir, net + ".pkl.tmp"), "wb") as f:
        pickle.dump(dict(net=net, inputs=inputs, output=output, size=size, target=target, trees=trees), f)
    os.replace(os.path.join(outdir, net + ".pkl.tmp"), os.path.join(outdir, net + ".pkl"))
    # plain JSON copy for cuTensorNet (no cotengra needed to read it): integer labels, linear path, sliced labels
    import json
    inv = {ctg.get_symbol(i): i for i in range(1, len(size) + 2)}
    js = dict(net=net, inputs=[[inv[x] for x in t] for t in inputs], output=[inv[x] for x in output],
              size={inv[k]: v for k, v in size.items()},
              trees=[dict(tag=tag, path=[list(p) for p in t.get_path()], sliced=[inv[x] for x in t.sliced_inds],
                          log10C=t.contraction_cost(log=10), W=t.contraction_width(), search_s=s)
                     for tag, t, s, _ in trees])
    with open(os.path.join(outdir, net + ".json"), "w") as f:
        json.dump(js, f)
    with open(os.path.join(outdir, "trees.tsv"), "a") as f:
        for tag, t, s, info in trees:
            f.write("\t".join([net, tag, f"{t.contraction_cost(log=10):.6f}", f"{t.contraction_width():.3f}",
                               str(t.multiplicity), f"{s:.3f}", repr(info)]) + "\n")


if __name__ == "__main__":
    main()
