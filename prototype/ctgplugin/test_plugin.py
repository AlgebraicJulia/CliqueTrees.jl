"""Plugin tests: (a) numerical validity on a tiny random network with size-1 dims, scalars, hyperindices
and open indices; (b) on 5 corpus networks: path valid, and plugin cost == bridge cost
(frontier/scripts/ftn.py tree_from_order + refine) for the same ordering.
usage: python test_plugin.py [threads]"""
import math
import os
import sys
import time

import numpy as np
import cotengra as ctg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/itaykadosh/Projects/GATAS/tn-work/frontier/scripts")
from ctg_cliquetrees import CliqueTreesOptimizer  # noqa: E402
import ftn  # noqa: E402
import tnc  # noqa: E402

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def valid_ssa(ssa, n):
    used = set()
    for k, (a, b) in enumerate(ssa):
        assert a != b and a < n + k and b < n + k and a not in used and b not in used, (k, a, b)
        used.update((a, b))
    # exactly one live node left
    assert len(used) == n + len(ssa) - 1


# (a) numerics
rng = np.random.default_rng(0)
eq = "ab,bcd,cde,e,fg,gh,,ah,ijk,kl,dl,ab->ijf"   # size-1 dim 'a'? set below; scalar tensor; repeated tensor
eq = "ab,bcd,cde,e,fg,gh,,ahi,ijk,kl,dl->jf"
sizes = dict(a=1, b=2, c=3, d=2, e=2, f=3, g=2, h=2, i=2, j=2, k=2, l=2)
terms = eq.split("->")[0].split(",")
arrs = [rng.standard_normal([sizes[c] for c in t]) for t in terms]
ref = np.einsum(eq, *arrs, optimize=True)
for method in ("bw", "hbtnd", "treesa"):
    opt = CliqueTreesOptimizer(method=method, time=0.3, threads=threads, portfolio=False)
    out = ctg.einsum(eq, *arrs, optimize=opt)
    assert np.allclose(out, ref), method
    ssa = opt.ssa_path([tuple(t) for t in terms], tuple("jf"), sizes)
    valid_ssa(ssa, len(terms))
    print(f"numeric ok  {method:6s} nsteps {len(ssa)}", flush=True)

# (b) corpus
NAMES = ["gm_pedigree50", "lm_batch_likelihood_brackets_3_16d", "str_nw_mera_open_26", "qc_qft_27", "mc_2021_074"]
allnames = tnc.einsum_names()
want = [n for n in NAMES if n in allnames]
if len(want) < 5:   # fall back: spread over the size range
    want = [allnames[i] for i in (3, 25, 60, 100, 130)]
for name in want:
    inputs, output, size = tnc.load(name)
    for method in ("hbtnd", "hbt"):
        opt = CliqueTreesOptimizer(method=method, time=1.0, threads=threads, portfolio=False)
        t0 = time.time()
        ssa = opt.ssa_path(inputs, output, size)
        dt = time.time() - t0
        valid_ssa(ssa, len(inputs))
        tree = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=ssa)   # rebuild from the returned path only
        c_path = tree.contraction_cost(log=10)
        # bridge: same ordering through ftn
        order = [{ftn.sym(v): v for v in range(1, len(size) + 1)}[s] for s in opt.last_info["order"]]
        bt = ftn.refine(ftn.tree_from_order(inputs, output, size, order))
        c_bridge = ftn.stats(bt)["cost"]
        ok = abs(c_path - c_bridge) < 1e-6
        # portfolio never worse than greedy
        pf = CliqueTreesOptimizer(method=method, time=0.5, threads=threads, portfolio=True)
        pt = pf.search(inputs, output, size)
        g = ctg.array_contract_tree(inputs, output, size, optimize="greedy").contraction_cost(log=10)
        okp = pt.contraction_cost(log=10) <= g + 1e-9
        print(f"{name:22s} {method:5s} n={len(inputs):5d} {dt:5.2f}s  logC plugin {c_path:.4f} bridge {c_bridge:.4f} "
              f"{'MATCH' if ok else 'MISMATCH'}  greedy {g:.3f} portfolio {pt.contraction_cost(log=10):.3f} "
              f"({pf.last_info['chose']}) {'ok' if okp else 'WORSE'}", flush=True)
        assert ok and okp
print("ALL OK")
