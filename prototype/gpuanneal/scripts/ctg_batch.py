"""cotengra hyper batch with anytime traces: python ctg_batch.py <list: name per line> <budget s> <parallel> <outdir> <results tsv> [Wt]
Per instance: one HyperOptimizer run (greedy+kahypar, optuna, minimize flops, subtree reconfiguration of
every trial; slicing_reconf to 2**Wt if given) with max_time = budget. The anytime curve comes from the
optimizer's per-trial (time, score) history (cumulative best). Final tree -> <outdir>/<name>__ctg....tree."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cotengra as ctg
import tnio

lst, budget, par, odir, resf = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
Wt = float(sys.argv[6]) if len(sys.argv) > 6 else None
os.makedirs(odir, exist_ok=True)
for name in open(lst).read().split():
    tag = f"{name}__ctg_p{par}_T{int(budget)}" + (f"_W{int(Wt)}" if Wt else "")
    if os.path.exists(os.path.join(odir, tag + ".tree")):
        continue
    inputs, output, size, _ = tnio.load(name)
    kw = dict(max_time=budget, max_repeats=10**9, parallel=(par if par > 0 else False), reconf_opts={}, minimize="flops", progbar=False)
    if Wt:
        kw["slicing_reconf_opts"] = {"target_size": 2 ** int(Wt)}
    opt = ctg.HyperOptimizer(**kw)
    t0 = time.time()
    tree = ctg.array_contract_tree(inputs, output, size, optimize=opt)
    ts = time.time() - t0
    lab = {}
    for a, b in zip(tree.inputs, inputs):
        for x, y in zip(a, b):
            lab.setdefault(x, y)
    inv = {tnio.sym(v + 1): v + 1 for v in range(len(size))}
    sl = [inv[lab[x]] for x in tree.sliced_inds]
    st = tnio.tree_stats(tree)
    tnio.write_tree(os.path.join(odir, tag + ".tree"), len(inputs), [tuple(map(int, p)) for p in tree.get_ssa_path()], sl,
                    {"t": f"{ts:.3f}", "ctg_log2C": f"{st['log2C']:.6f}"})
    with open(os.path.join(odir, tag + ".trace.json"), "w") as f:
        json.dump({"times": list(getattr(opt, "times", [])), "scores": list(getattr(opt, "scores", []))}, f)
    with open(resf, "a") as f:
        f.write("\t".join(map(str, (name, "ctg", par, budget, Wt or "Inf", f"{ts:.2f}", f"{st['log2C']:.4f}", st["W"], st["nsliced"]))) + "\n")
    print(name, f"{ts:.1f}s", round(st["log2C"], 3), st["W"], flush=True)
