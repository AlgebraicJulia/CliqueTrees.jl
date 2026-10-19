"""Seed contraction trees for the GPU annealer.
usage:
  python seeds.py <name> td <ord file> <out.tree>        elimination ordering -> tree (bucket elimination,
                                                        ftn.tree_from_order: merges binarized by 'auto-hq')
  python seeds.py <name> greedy <seed> <out.tree>       cotengra greedy (seed 0 = deterministic default;
                                                        seed>0 = one random-greedy trial, temperature 0.3)
  python seeds.py <name> rgreedy <secs> <out.tree>      best of cotengra random-greedy trials in <secs> s
  python seeds.py <name> random <seed> <out.tree>       uniformly random sequence of contractions of
                                                        tensors that share an index (connected random tree)
Prints log2C / W of the seed (cotengra).
"""
import random
import sys
import time

import cotengra as ctg
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tnio


def from_order(inputs, output, size, order):
    outs = set(output)
    ep = [tnio.sym(v) for v in order if tnio.sym(v) not in outs]
    return ctg.ContractionTree.from_path(inputs, output, size, edge_path=ep, optimize="auto-hq", autocomplete=True)


def random_tree(inputs, seed):
    rng = random.Random(seed)
    alive = {i: set(t) for i, t in enumerate(inputs)}
    nxt = len(inputs)
    ssa = []
    while len(alive) > 1:
        # random index shared by >= 2 alive tensors; else random pair
        keys = list(alive)
        a = rng.choice(keys)
        nb = [b for b in keys if b != a and alive[a] & alive[b]]
        b = rng.choice(nb) if nb else rng.choice([k for k in keys if k != a])
        ssa.append((a, b))
        alive[nxt] = alive.pop(a) | alive.pop(b)
        nxt += 1
    return ssa


def main():
    name, kind, arg, out = sys.argv[1:5]
    inputs, output, size, _ = tnio.load(name)
    t0 = time.time()
    if kind == "td":
        with open(arg) as f:
            f.readline()
            order = [int(x) for x in f.readline().split()]
        tree = from_order(inputs, output, size, order)
        ssa = tree.get_ssa_path()
    elif kind == "greedy":
        s = int(arg)
        if s == 0:
            ssa = ctg.array_contract_tree(inputs, output, size, optimize="greedy").get_ssa_path()
            # array_contract_tree may reorder? it keeps input order; verified by re-costing below
        else:
            opt = ctg.RandomGreedyOptimizer(max_repeats=1, seed=s, temperature=0.3, parallel=False)
            ssa = ctg.array_contract_tree(inputs, output, size, optimize=opt).get_ssa_path()
    elif kind == "rgreedy":
        # best (by flops) of random-greedy trials (cotengrust, batches of 32) within <arg> seconds
        best = None; k = 0
        while best is None or time.time() - t0 < float(arg):
            opt = ctg.RandomGreedyOptimizer(max_repeats=32, seed=1000 + k, parallel=False)
            t = ctg.array_contract_tree(inputs, output, size, optimize=opt)
            k += 1
            if best is None or t.contraction_cost() < best.contraction_cost():
                best = t
        ssa = best.get_ssa_path()
    elif kind == "random":
        ssa = random_tree(inputs, int(arg))
    else:
        raise ValueError(kind)
    dt = time.time() - t0
    ssa = [tuple(map(int, p)) for p in ssa]
    tree = tnio.ctg_tree(inputs, output, size, ssa)
    st = tnio.tree_stats(tree)
    tnio.write_tree(out, len(inputs), ssa, (), {"seed_kind": kind, "seed_arg": arg, "seed_secs": f"{dt:.3f}",
                                                 "ctg_log2C": f"{st['log2C']:.6f}", "ctg_W": f"{st['W']:.3f}"})
    print(name, kind, arg, f"{dt:.2f}s", {k: round(v, 3) for k, v in st.items()})


if __name__ == "__main__":
    main()
