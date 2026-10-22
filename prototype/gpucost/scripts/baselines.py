"""CPU baselines, one thread, writing tnio .tree files (re-costed later by verify.py) and a TSV row each.
usage: python -I baselines.py <orddir> <outdir> <methods comma> <name> [<name> ...]
methods:
  ctgref1   (a) at 1 s:  our hbt 1 s ordering -> ftn.tree_from_order -> ftn.refine (cotengra subtree reconf)
  ctgref10  (a) at 10 s: our hbt-nd 10 s ordering, same
  hypersa1, hypersa10:   cotengra HyperOptimizer(minimize=flops, simulated_annealing_opts={}), max_time 1 / 10 s
  hyper10:               cotengra HyperOptimizer (default: subtree reconfiguration of every trial), 10 s
  autohq:                cotengra 'auto-hq' preset (not time-limited), single-threaded
TSV columns: name method seconds log10C W"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/itaykadosh/Projects/GATAS/tn-work/frontier/scripts")
import cotengra as ctg  # noqa: E402
import tnio  # noqa: E402
import ftn  # noqa: E402


def ord_of(orddir, name, tag):
    with open(os.path.join(orddir, f"{name}__{tag}.ord")) as f:
        hdr = f.readline().split("\t")
        return float(hdr[2]), [int(x) for x in f.readline().split()]


def run(orddir, name, m, inputs, output, size):
    if m.startswith("ctgref"):
        tag = "hbtT1.0s1" if m == "ctgref1" else "hbtndT10.0s1"
        tord, order = ord_of(orddir, name, tag)
        t0 = time.time()
        tree = ftn.refine(ftn.tree_from_order(inputs, output, size, order))
        return tree, tord + time.time() - t0
    if m == "autohq":
        opt = ctg.AutoHQOptimizer(parallel=False)
        t0 = time.time()
        tree = opt.search(inputs, output, size)
        return tree, time.time() - t0
    T = float(m.replace("hypersa", "").replace("hyper", ""))
    kw = dict(simulated_annealing_opts={}) if m.startswith("hypersa") else {}
    opt = ctg.HyperOptimizer(minimize="flops", max_time=T, max_repeats=10**6, parallel=False, progbar=False, **kw)
    t0 = time.time()
    tree = opt.search(inputs, output, size)
    return tree, time.time() - t0


def main(orddir, outdir, methods, names):
    os.makedirs(outdir, exist_ok=True)
    for name in names:
        inputs, output, size, _ = tnio.load(name)
        for m in methods:
            f = os.path.join(outdir, f"{name}__{m}.tree")
            if os.path.exists(f):
                continue
            try:
                tree, dt = run(orddir, name, m, inputs, output, size)
            except Exception as e:  # noqa: BLE001
                print(name, m, "FAILED", repr(e), flush=True)
                continue
            c2 = tree.contraction_cost(log=2); w = tree.contraction_width(log=2)
            tnio.write_tree(f, len(inputs), tree.get_ssa_path(), meta={"log2C": f"{c2:.6f}", "W": f"{w:.4f}", "t": f"{dt:.3f}"})
            with open(os.path.join(outdir, "baselines.tsv"), "a") as fo:
                fo.write(f"{name}\t{m}\t{dt:.3f}\t{c2 * 0.30102999566:.4f}\t{w:.3f}\n")
            print(name, m, f"{dt:.2f}s", f"log10C {c2 * 0.30102999566:.4f}", f"W {w:.2f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4:])
