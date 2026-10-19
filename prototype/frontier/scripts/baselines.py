"""cotengra baselines on one frontier instance, single-threaded (parallel=False).
usage: python baselines.py <name> <method> <budget s> <seed> <outdir> <targets comma>
methods:
  greedy        optimize='greedy' (cotengra default greedy, one deterministic run)
  rgreedy       RandomGreedyOptimizer (cotengrust), batches of 32 random greedy trials until the
                budget is spent; best by flops
  hyper         HyperOptimizer(methods=('greedy','kahypar') [the default], optlib optuna [default],
                minimize='flops', reconf_opts={} -> subtree reconfiguration of every trial)
  hslice<t>     the same plus slicing_reconf_opts={'target_size': 2**t} (dynamic slicing to 2^t
                inside every trial, the setting of the cotengra notebooks reproducing 2005.06787
                and 2103.03074)
Writes <outdir>/<name>__<method>__T<budget>__s<seed>.tsv with the post-processing stages of
ftn.post (raw, refined, sliced) and <...>.path.json (the ssa path of the tree found).
"""
import json
import os
import sys
import time

import cotengra as ctg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftn  # noqa: E402


def cut_task(name, method, budget, seed, outdir, targets):
    """method 'cut<src>' (e.g. cuthslice27): wait for <outdir>/<name>__<src>__T<budget>__s<seed>.path.json
    (written by another task of the same job), then run the fixed-slicing experiment cutexp.sh on it."""
    import subprocess
    src = method[3:]
    pj = os.path.join(outdir, f"{name}__{src}__T{int(budget)}__s{seed}.path.json")
    t0 = time.time()
    while not os.path.exists(pj):
        if time.time() - t0 > 4 * 3600:
            print("gave up waiting for", pj); return
        time.sleep(30)
    core = str(min(os.sched_getaffinity(0)))
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(os.path.dirname(os.path.abspath(outdir)), "cut")
    budgets = os.environ.get("CUT_BUDGETS", "10,100,1000")
    print("waited", round(time.time() - t0), "s for", pj, flush=True)
    import shutil
    env = dict(os.environ)
    env["PY"] = sys.executable
    hpg_env = "/blue/fairbanksj/itaykadosh/gpu/ct-mt/envs/mt-65e72a3"
    if os.path.isdir(hpg_env):
        env.setdefault("JENV", hpg_env)
        env.setdefault("JL", shutil.which("julia") or "julia")
    subprocess.run(["bash", os.path.join(here, "cutexp.sh"), name, pj, str(targets[0]), out, core, budgets], check=False, env=env)


def main(name, method, budget, seed, outdir, targets):
    if method.startswith("cut"):
        return cut_task(name, method, budget, seed, outdir, targets)
    inputs, output, size = ftn.load(name)
    os.makedirs(outdir, exist_ok=True)
    tag = f"{name}__{method}__T{int(budget)}__s{seed}"
    t0 = time.time()
    ntrial = 0
    if method == "greedy":
        tree = ctg.array_contract_tree(inputs, output, size, optimize="greedy")
        ntrial = 1
    elif method == "rgreedy":
        tree = None
        k = 0
        while tree is None or time.time() - t0 < budget:
            opt = ctg.RandomGreedyOptimizer(max_repeats=32, seed=seed * 100003 + k, parallel=False)
            t = ctg.array_contract_tree(inputs, output, size, optimize=opt)
            k += 1; ntrial += 32
            if tree is None or t.contraction_cost() < tree.contraction_cost():
                tree = t
    elif method == "hyper" or method.startswith("hslice"):
        kw = dict(max_time=budget, max_repeats=10**9, parallel=False, reconf_opts={}, minimize="flops", progbar=False)
        if method.startswith("hslice"):
            kw["slicing_reconf_opts"] = {"target_size": 2 ** int(method[6:])}
        try:
            kw["optlib_opts"] = {"seed": seed}
            opt = ctg.HyperOptimizer(**kw)
        except TypeError:
            kw.pop("optlib_opts"); opt = ctg.HyperOptimizer(**kw)
        tree = ctg.array_contract_tree(inputs, output, size, optimize=opt)
        ntrial = len(opt.scores) if hasattr(opt, "scores") else -1
    else:
        raise ValueError(method)
    tsearch = time.time() - t0
    # cotengra canonicalizes index labels inside array_contract_tree: map back to ours (same
    # tensor order, same index order within each tensor)
    lab = {}
    for a, b in zip(tree.inputs, inputs):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert lab.setdefault(x, y) == y
    with open(os.path.join(outdir, tag + ".path.json"), "w") as f:
        json.dump({"ssa_path": [list(map(int, p)) for p in tree.get_ssa_path()],
                   "sliced_inds": [lab[x] for x in tree.sliced_inds]}, f)
    if method.startswith("hslice"):
        rows = [("raw", ftn.stats(tree), 0.0)]
    else:
        rows, _ = ftn.post(tree, targets)
    with open(os.path.join(outdir, tag + ".tsv"), "w") as f:
        for stage, st, dt in rows:
            f.write("\t".join(map(str, (name, method, budget, seed, round(tsearch, 3), ntrial, "", stage,
                                         round(st["cost"], 4), round(st["width"], 3), round(st["maxop"], 3),
                                         st["nsliced"], round(dt, 3)))) + "\n")
    print(tag, f"{tsearch:.1f}s trials={ntrial}", {k: round(v, 2) for k, v in rows[0][1].items()}, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], float(a[2]), int(a[3]), a[4], [int(x) for x in a[5].split(",")])
