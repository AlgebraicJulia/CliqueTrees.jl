"""Batch seed trees: python mkseeds.py <list file> <ord dir> <seed dir> [rg seconds]
For each instance: hbt<T> / nd seeds from every matching ordering file <ord dir>/<name>__<m>__T<T>__s1.ord,
rg (best cotengra random-greedy within rg seconds, default 1), rand (random connected tree, seed 1).
Writes <seed dir>/<name>.<tag>.tree and prints a TSV row per seed: name tag secs log2C W."""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cotengra as ctg  # noqa: E402
import tnio  # noqa: E402
from seeds import from_order, random_tree  # noqa: E402


def emit(name, tag, inputs, output, size, ssa, dt, sdir):
    ssa = [tuple(map(int, p)) for p in ssa]
    st = tnio.tree_stats(tnio.ctg_tree(inputs, output, size, ssa))
    tnio.write_tree(os.path.join(sdir, f"{name}.{tag}.tree"), len(inputs), ssa, (), {"seed_secs": f"{dt:.3f}", "ctg_log2C": f"{st['log2C']:.6f}"})
    print(f"{name}\t{tag}\t{dt:.3f}\t{st['log2C']:.4f}\t{st['W']:.1f}", flush=True)


def main():
    lst, odir, sdir = sys.argv[1:4]
    rgs = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    os.makedirs(sdir, exist_ok=True)
    for name in open(lst).read().split():
        inputs, output, size, _ = tnio.load(name)
        for f in sorted(glob.glob(os.path.join(odir, f"{name}__*__s1.ord"))):
            m, T = os.path.basename(f)[len(name) + 2:].split("__")[:2]
            tag = m + T[1:] if m != "nd" else "nd"
            if os.path.exists(os.path.join(sdir, f"{name}.{tag}.tree")):
                continue
            with open(f) as fh:
                hdr = fh.readline().split("\t")
                order = [int(x) for x in fh.readline().split()]
            t0 = time.time()
            tree = from_order(inputs, output, size, order)
            emit(name, tag, inputs, output, size, tree.get_ssa_path(), float(hdr[4]) + time.time() - t0, sdir)
        if not os.path.exists(os.path.join(sdir, f"{name}.rg.tree")):
            t0 = time.time(); best = None; k = 0
            while best is None or time.time() - t0 < rgs:
                opt = ctg.RandomGreedyOptimizer(max_repeats=32, seed=1000 + k, parallel=False)
                t = ctg.array_contract_tree(inputs, output, size, optimize=opt); k += 1
                if best is None or t.contraction_cost() < best.contraction_cost():
                    best = t
            emit(name, "rg", inputs, output, size, best.get_ssa_path(), time.time() - t0, sdir)
        if not os.path.exists(os.path.join(sdir, f"{name}.rand.tree")):
            t0 = time.time()
            emit(name, "rand", inputs, output, size, random_tree(inputs, 1), time.time() - t0, sdir)


if __name__ == "__main__":
    main()
