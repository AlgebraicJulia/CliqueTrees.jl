"""Benchmark worker: pops tasks from a queue file, runs them, appends result rows.
usage: python worker.py <queue file> <outdir> <kind: ours|ctg> --cfg c1|c8 [--threads 1] [--ctg-parallel 0] [--tag w0]
Configurations (both are run, rows tagged by cfg):
  c1 = equal single core: ours Julia threads=1, every cotengra method single-threaded (parallel=False);
  c8 = equal 8 cores:     ours threads=8 (HBT threads; TreeSA ntrials=8), cotengra hyper / hyper-sa / auto / auto-hq
                          with parallel=8 (cotengra's pool).
Result rows (TSV, <outdir>/res/<kind>_<cfg>_<tag>.tsv), columns:
  network method budget seed cfg status wall_s search_s post_s logC width maxop pf_logC pf_width greedy_s extra
wall_s = total wall time of the call = search_s + post_s (the primary x-axis).  ours: search_s = Julia call (line
graph + ordering; JIT excluded by warm-up), post_s = ordering->tree conversion + capped refinement.  cotengra:
search_s = wall_s, post_s = 0.  logC = log10 of cotengra's contraction_cost, width = contraction_width (log2 of
the largest intermediate), maxop = log2 of the largest single contraction.
PURE mode = the tree the method returned (logC, width).  PORTFOLIO mode = cheaper of that tree and cotengra's
greedy tree (pf_logC, pf_width; the portfolio wall time is wall_s + greedy_s; greedy_s is measured outside wall_s).
status: ok | timeout | error:<msg>.  extra: key=value;...  (ours: bag, julia, conv, refine; cotengra budgeted:
overrun = wall_s - budget).  Caps: ours 10*T+60 s (server killed and restarted); cotengra budgeted 2*T+30 s
(child killed -> timeout, counted as a failure; overruns below the cap are recorded as `overrun`), auto 300 s,
auto-hq 600 s, greedy 120 s.
Methods: see gen_tasks.py. cotengra methods run in a forked child (killable); settings follow
tn-work/frontier/scripts/baselines.py: hyper = HyperOptimizer(max_time=T, max_repeats=1e9, reconf_opts={},
minimize='flops', parallel=False|8); hyper-sa the same with simulated_annealing_opts={} instead of
reconf; auto / auto-hq = cotengra AutoOptimizer / AutoHQOptimizer with cache=False."""
import argparse
import fcntl
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("CTGPLUGIN", os.path.join(HERE, "..", "ctgplugin")))
import cotengra as ctg  # noqa: E402
import tnc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("queue"); ap.add_argument("outdir"); ap.add_argument("kind")
ap.add_argument("--cfg", default="c1")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--ctg-parallel", type=int, default=0)
ap.add_argument("--tag", default=str(os.getpid()))
a = ap.parse_args()


def pop(path):
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path) as f:
            lines = f.readlines()
        if not lines:
            return None
        with open(path, "w") as f:
            f.writelines(lines[1:])
        return lines[0].split()


def finish(tree, inputs, output, size):
    """pure stats + portfolio stats (cheaper of tree and greedy; greedy timed separately)."""
    c, w, mo = stats(tree)
    g0 = time.time()
    gt = ctg.array_contract_tree(inputs, output, size, optimize="greedy")
    gs = time.time() - g0
    gc, gw, _ = stats(gt)
    pc, pw = (gc, gw) if gc < c else (c, w)
    return dict(logc=c, width=w, maxop=mo, pf_logc=pc, pf_width=pw, greedy=gs)


def stats(tree):
    mo = max((tree.get_flops(nd) for nd, _, _ in tree.traverse()), default=1)
    return tree.contraction_cost(log=10), tree.contraction_width(), math.log2(max(mo, 1))


def ctg_run(method, T, seed, inputs, output, size, par):
    p = par if par > 0 else False
    t0 = time.time()
    if method == "greedy":
        tree = ctg.array_contract_tree(inputs, output, size, optimize="greedy")
    elif method in ("auto", "auto-hq"):
        cls = ctg.presets.AutoOptimizer if method == "auto" else ctg.presets.AutoHQOptimizer
        tree = ctg.array_contract_tree(inputs, output, size, optimize=cls(parallel=p, cache=False))
    else:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        kw = dict(max_time=T, max_repeats=10**9, parallel=p, minimize="flops", progbar=False, optlib="optuna",
                  optlib_opts={"sampler": optuna.samplers.TPESampler(seed=seed)})
        if method == "hyper":
            kw["reconf_opts"] = {}
        else:   # hyper-sa
            kw["simulated_annealing_opts"] = {}
        tree = ctg.array_contract_tree(inputs, output, size, optimize=ctg.HyperOptimizer(**kw))
    dt = time.time() - t0
    return dict(tree=tree, wall=dt, search=dt, post=0.0, extra=("overrun=%.3f" % (dt - T)) if method in ("hyper", "hyper-sa") else "")


def child(conn, *args):
    os.setsid()   # own process group: the killer also takes cotengra's pool workers (parallel=8)
    try:
        r = ctg_run(*args)
        t = r.pop("tree")
        conn.send(("ok", dict(r, **finish(t, args[3], args[4], args[5]))))
    except Exception as e:   # noqa: BLE001
        conn.send(("error", repr(e)[:200]))


def run_ctg(method, T, seed, inputs, output, size):
    cap = {"auto": 300, "auto-hq": 600, "greedy": 120}.get(method, 2 * T + 30)
    ctx = mp.get_context("fork")
    pa, pb = ctx.Pipe(False)
    pr = ctx.Process(target=child, args=(pb, method, T, seed, inputs, output, size, a.ctg_parallel))
    pr.start(); t0 = time.time()
    if pa.poll(cap):
        st, val = pa.recv()
    else:
        st, val = "timeout", None
    try:
        os.killpg(pr.pid, 9)    # child + any pool workers it spawned
    except (ProcessLookupError, PermissionError):
        pass
    pr.join()
    nan = math.nan
    return (st, val) if st != "timeout" else ("timeout", dict(wall=time.time() - t0, search=time.time() - t0, post=0.0, logc=nan, width=nan, maxop=nan, pf_logc=nan, pf_width=nan, greedy=0.0, extra="killed"))


if a.kind == "ours":
    from ctg_cliquetrees import CliqueTreesOptimizer, get_server
    srv = get_server(a.threads, log=os.path.join(a.outdir, "logs", f"julia_{a.tag}.log"))


def run_ours(method, T, seed, inputs, output, size):
    opt = CliqueTreesOptimizer(method=method, time=T, threads=a.threads, portfolio=False, refine=True, seed=seed,
                               timeout=10 * T + 60)
    opt.warm()
    t0 = time.time()
    nan = math.nan
    try:
        tree = opt.search(inputs, output, size)
    except TimeoutError:
        dt = time.time() - t0
        return "timeout", dict(wall=dt, search=dt, post=0.0, logc=nan, width=nan, maxop=nan, pf_logc=nan, pf_width=nan, greedy=0.0, extra="killed")
    dt = time.time() - t0
    i = opt.last_info
    ex = f"bag={i.get('bagwidth', '')};julia_inner={i.get('julia_seconds', 0):.3f};conv={i.get('convert_only', 0):.3f};refine={i.get('refine_s', 0):.3f}"
    return "ok", dict(wall=dt, search=i.get("julia", 0.0), post=dt - i.get("julia", 0.0), extra=ex, **finish(tree, inputs, output, size))


os.makedirs(os.path.join(a.outdir, "res"), exist_ok=True); os.makedirs(os.path.join(a.outdir, "logs"), exist_ok=True)
out = open(os.path.join(a.outdir, "res", f"{a.kind}_{a.cfg}_{a.tag}.tsv"), "a")
cache = {}
while True:
    t = pop(a.queue)
    if t is None:
        break
    name, method, T, seed = t[0], t[1], float(t[2]), int(t[3])
    if name not in cache:
        cache.clear(); cache[name] = tnc.load(name)
    inputs, output, size = cache[name]
    nan = math.nan
    try:
        st, val = (run_ours if a.kind == "ours" else run_ctg)(method, T, seed, inputs, output, size)
    except Exception as e:   # noqa: BLE001
        st, val = f"error:{e!r}"[:120], None
    if st == "error":
        st = "error:" + str(val)[:100]; val = None
    if val is None:
        val = dict(wall=nan, search=nan, post=nan, logc=nan, width=nan, maxop=nan, pf_logc=nan, pf_width=nan, greedy=nan, extra="")
    v = val
    out.write("\t".join(map(str, (name, method, T, seed, a.cfg, st, round(v["wall"], 4), round(v["search"], 4), round(v["post"], 4),
              round(v["logc"], 4), round(v["width"], 3), round(v["maxop"], 3), round(v["pf_logc"], 4), round(v["pf_width"], 3),
              round(v["greedy"], 4), v["extra"]))) + "\n")
    out.flush()
    print(f"{time.strftime('%T')} {name} {method} T={T:g} s={seed} {a.cfg}: {st} wall={v['wall']:.2f}s (search {v['search']:.2f} post {v['post']:.2f}) logC={v['logc']:.3f} W={v['width']:.2f}", flush=True)
if a.kind == "ours":
    srv.kill()
