"""cotengra baseline with an anytime trace.
usage: python ctg_hyper.py <name> <budget s> <parallel (0 = single thread)> <out prefix> [Wt]
HyperOptimizer(methods greedy+kahypar [default], optuna, minimize='flops', reconf_opts={} = subtree
reconfiguration of every trial; with Wt: slicing_reconf_opts target_size 2**Wt as in the cotengra
notebooks). A callback-free anytime trace: the optimizer's per-trial scores and their timestamps are
recorded by wrapping the trial function; at the end the best tree is written as .tree for re-costing.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cotengra as ctg
import tnio

name, budget, par, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
Wt = float(sys.argv[5]) if len(sys.argv) > 5 else None
inputs, output, size, _ = tnio.load(name)
kw = dict(max_time=budget, max_repeats=10**9, parallel=(par if par > 0 else False), reconf_opts={},
          minimize="flops", progbar=False)
if Wt is not None:
    kw["slicing_reconf_opts"] = {"target_size": 2 ** int(Wt)}
opt = ctg.HyperOptimizer(**kw)
t0 = time.time()
tree = ctg.array_contract_tree(inputs, output, size, optimize=opt)
ts = time.time() - t0
# anytime trace from the optimizer's history (scores in log10 flops for minimize='flops'), cumulative min
tr = []
try:
    best = float("inf")
    for i, s in enumerate(opt.scores):
        best = min(best, s); tr.append(best)
except Exception:
    pass
st = tnio.tree_stats(tree)
lab = {}
for a, b in zip(tree.inputs, inputs):
    for x, y in zip(a, b):
        lab.setdefault(x, y)
inv = {tnio.sym(v + 1): v + 1 for v in range(len(size))}
sl = [inv[lab[x]] for x in tree.sliced_inds]
tnio.write_tree(out + ".tree", len(inputs), [tuple(map(int, p)) for p in tree.get_ssa_path()], sl,
                {"t": f"{ts:.3f}", "ntrials": len(opt.scores) if hasattr(opt, "scores") else -1, "ctg_log2C": f"{st['log2C']:.6f}",
                 "ctg_W": st["W"], "parallel": par, "budget": budget})
with open(out + ".trace.json", "w") as f:
    json.dump({"times": getattr(opt, "times", []), "scores": list(getattr(opt, "scores", []))}, f)
print(name, budget, par, f"{ts:.1f}s", st, flush=True)
