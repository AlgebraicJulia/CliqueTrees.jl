"""Experiment A, cotengra side: cost every TD-induced contraction tree of one network.

    python -I costA.py <tdfile> <out.tsv> [hyper_seconds=10]

Per ordering (tag) of the .td file, stages:
  bucket      cotengra bucket elimination on the ordering (ContractionTree.from_path(edge_path=...), greedy
              binarization of each bucket; open indices re-rooted last) = the TD's plain induced tree
  bt          Richard's clique-tree binarization (CliqueTrees branchtree)
  br          branchtree + branchrefine (our best conversion)
  bucket+ctg  bucket + cotengra subtree_reconfigure (subtree 8, to a fixed point, <= 8 passes)
  br+ctg      br + cotengra subtree_reconfigure
Baselines (tag ctg_greedy / ctg_hyper<T>): cotengra greedy and HyperOptimizer(max_time=T, parallel=False,
minimize='flops'), stages raw and +ctg.
Columns: net tag stage log10C W maxop seconds tdwidth lcost jl_log10C
(log10C = cotengra contraction_cost, W = log2 largest intermediate, maxop = log2 largest pairwise
contraction, jl_log10C = CliqueTrees branchstats cost for bt/br, to check the two agree).
"""
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cotengra as ctg  # noqa: E402
import ftn  # noqa: E402
import tdfile  # noqa: E402
import tnc  # noqa: E402

L2 = math.log10(2.0)


def row(out, net, tag, stage, st, secs, tdw="", lc="", jl=""):
    fmt = lambda x: x if isinstance(x, str) else f"{x:.6f}"
    out.write("\t".join([net, tag, stage, fmt(st["cost"]), fmt(st["width"]), fmt(st["maxop"]), fmt(secs),
                         fmt(tdw), fmt(lc), fmt(jl)]) + "\n")
    out.flush()


def main():
    tdpath, outpath = sys.argv[1], sys.argv[2]
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    td = tdfile.parse(tdpath)
    net = td["net"]["name"]
    inputs, output, size = tnc.load(net)
    done = set()
    if os.path.exists(outpath):
        for line in open(outpath):
            if line.startswith(("#", "net\t")):
                continue
            f = line.split("\t")
            done.add((f[1], f[2]))
    out = open(outpath, "a")
    if not done:
        out.write("net\ttag\tstage\tlog10C\tW\tmaxop\tseconds\ttdwidth\tlcost\tjl_log10C\n")
    for d in td["orders"]:
        tag = d["tag"]
        if (tag, "br+ctg") in done:
            continue
        t0 = time.time()
        bucket = ftn.tree_from_order(inputs, output, size, d["order"], optimize="greedy")
        row(out, net, tag, "bucket", ftn.stats(bucket), time.time() - t0, d["width"], d["lcost"])
        bt = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=d["ssa_bt"])
        row(out, net, tag, "bt", ftn.stats(bt), 0.0, d["width"], d["lcost"], d["bt_cost"] * L2)
        br = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=d["ssa_br"])
        row(out, net, tag, "br", ftn.stats(br), d["br_seconds"], d["width"], d["lcost"], d["br_cost"] * L2)
        t0 = time.time(); r = ftn.refine(bucket)
        row(out, net, tag, "bucket+ctg", ftn.stats(r), time.time() - t0, d["width"], d["lcost"])
        t0 = time.time(); r = ftn.refine(br)
        row(out, net, tag, "br+ctg", ftn.stats(r), time.time() - t0, d["width"], d["lcost"])
    if ("ctg_greedy", "raw") not in done:
        t0 = time.time()
        g = ctg.array_contract_tree(inputs, output, size, optimize="greedy")
        row(out, net, "ctg_greedy", "raw", ftn.stats(g), time.time() - t0)
        t0 = time.time(); r = ftn.refine(g)
        row(out, net, "ctg_greedy", "+ctg", ftn.stats(r), time.time() - t0)
    tagh = f"ctg_hyper{int(T)}"
    if (tagh, "raw") not in done:
        t0 = time.time()
        opt = ctg.HyperOptimizer(max_time=T, parallel=False, minimize="flops", progbar=False, max_repeats=10 ** 6)
        h = opt.search(inputs, output, size)
        row(out, net, tagh, "raw", ftn.stats(h), time.time() - t0)
        t0 = time.time(); r = ftn.refine(h)
        row(out, net, tagh, "+ctg", ftn.stats(r), time.time() - t0)
    out.write(f"#END\t{net}\n")
    out.close()


if __name__ == "__main__":
    main()
