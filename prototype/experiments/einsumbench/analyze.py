"""Performance profiles and cost-vs-time curves from the result TSVs.
usage: python analyze.py <outdir> [<plotdir>]   (reads <outdir>/res/*.tsv; writes plots + summary.tsv)
Runs once per configuration cfg (c1 = equal single core, c8 = equal 8 cores) and mode (pure = the method's own
tree; portfolio = cheaper of it and cotengra greedy, wall time + greedy_s), into <plotdir>/<cfg>_<mode>/.
Total wall time (search + post) is the x-axis everywhere; summary.tsv also gives median search_s and post_s.
best(network) is over every run of every method, cfg and mode (common reference).
`python analyze.py <outdir> <plotdir> <cfg> <mode>` does a single combination (used internally).
Per (network, method, budget): the median over seeds of log10 C (a failed seed = +inf; the median is
inf if more than half failed). best(network) = least log10 C over every run of every method.
gap = log10 C - best, so 10**gap is the factor over the best known tree.
One-shot methods (budget 0: greedy, bw, nd, auto, auto-hq) enter the panel of budget B when their
median wall time <= 10 B (the 10 T cap of the draft protocol; otherwise they count as failures there).
Figures:
  profile_<B>.png/ profiles.png : fraction of networks with 10**gap <= tau (tau on a log axis), one panel per budget
  curves.png : left, median gap (log10 C) vs median wall time; right, fraction within a factor 2 (gap <= log10 2)
               vs median wall time. Budgeted methods give one point per budget (line); one-shot methods one marker."""
import csv
import glob
import math
import os
import statistics
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import subprocess
out = sys.argv[1]
pdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(out, "plots")
os.makedirs(pdir, exist_ok=True)
CFG = sys.argv[3] if len(sys.argv) > 3 else None
MODE = sys.argv[4] if len(sys.argv) > 4 else None

allrows = []
for f in sorted(glob.glob(os.path.join(out, "res", "*.tsv"))):
    for r in csv.reader(open(f), delimiter="\t"):
        if len(r) >= 15:
            ok = r[5] == "ok"
            allrows.append(dict(net=r[0], method=r[1], budget=float(r[2]), seed=int(r[3]), cfg=r[4], status=r[5], wall=float(r[6]),
                                search=float(r[7]), post=float(r[8]), logc=float(r[9]) if ok else math.inf, width=float(r[10]),
                                pf=float(r[12]) if ok else math.inf, pfw=float(r[13]), greedy=float(r[14]), extra=r[15] if len(r) > 15 else ""))
if CFG is None:
    cfgs = sorted({r["cfg"] for r in allrows})
    for c in cfgs:
        for m in ("pure", "portfolio"):
            subprocess.run([sys.executable, "-I", os.path.abspath(__file__), out, os.path.join(pdir, f"{c}_{m}"), c, m], check=True)
    sys.exit(0)
best_all = {}
for r in allrows:
    best_all[r["net"]] = min(best_all.get(r["net"], math.inf), r["logc"], r["pf"])
rows = []
for r in allrows:
    if r["cfg"] != CFG:
        continue
    r = dict(r)
    if MODE == "portfolio":
        r["logc"] = r["pf"]; r["width"] = r["pfw"]; r["wall"] = r["wall"] + (r["greedy"] if math.isfinite(r["greedy"]) else 0)
    rows.append(r)
nets = sorted({r["net"] for r in rows})
cell = defaultdict(list)
for r in rows:
    cell[(r["net"], r["method"], r["budget"])].append(r)


def med(vals):
    vals = sorted(vals)
    n = len(vals)
    return vals[n // 2] if n % 2 else (math.inf if math.inf in vals[n // 2 - 1:n // 2 + 1] else sum(vals[n // 2 - 1:n // 2 + 1]) / 2)


best = {n: best_all[n] for n in nets}
methods = sorted({r["method"] for r in rows})
budgets = sorted({r["budget"] for r in rows if r["budget"] > 0})
oneshot = sorted({r["method"] for r in rows if r["budget"] == 0})
bmeth = [m for m in methods if m not in oneshot]
ORDER = ["hbtnd", "hbt", "nd", "bw", "treesa", "hyper-sa", "hyper", "auto-hq", "auto", "greedy"]
COL = {"hbtnd": "#c0392b", "hbt": "#e67e22", "nd": "#f1c40f", "bw": "#95a5a6", "treesa": "#16a085", "hyper-sa": "#2c3e50",
       "hyper": "#3498db", "auto-hq": "#8e44ad", "auto": "#9b59b6", "greedy": "#7f8c8d"}
methods.sort(key=lambda m: ORDER.index(m) if m in ORDER else 99)


def value(net, m, B):
    """(median log10 C, median wall) of method m on net at budget B, or None if absent/over budget."""
    if m in oneshot:
        c = cell.get((net, m, 0.0))
        if not c:
            return None
        w = statistics.median(x["wall"] for x in c)
        return (med([x["logc"] for x in c]), w) if (B is None or w <= 10 * B) else (math.inf, w)
    c = cell.get((net, m, B))
    return (med([x["logc"] for x in c]), statistics.median(x["wall"] for x in c)) if c else None


def gaps(m, B):
    g = []
    for n in nets:
        v = value(n, m, B)
        if v is not None:
            g.append(v[0] - best[n] if math.isfinite(v[0]) else math.inf)
    return g


# ---- summary
with open(os.path.join(pdir, "summary.tsv"), "w") as f:
    f.write("budget\tmethod\tn_nets\tmedian_wall_s\tmedian_search_s\tmedian_post_s\tmedian_gap_log10C\tfrac_within_1.1x\tfrac_within_2x\tfrac_within_10x\tfails\n")
    for B in budgets:
        for m in methods:
            ms = [m] if True else []
            g = gaps(m, B)
            if not g:
                continue
            w = [v[1] for n in nets if (v := value(n, m, B)) is not None]
            fr = lambda t: sum(x <= math.log10(t) + 1e-9 for x in g) / len(g)
            cc = [x for n in nets for x in (cell.get((n, m, 0.0 if m in oneshot else B)) or [])]
            ms_, mp_ = statistics.median([x["search"] for x in cc]), statistics.median([x["post"] for x in cc])
            f.write(f"{B:g}\t{m}\t{len(g)}\t{statistics.median(w):.3f}\t{ms_:.3f}\t{mp_:.3f}\t{statistics.median(g):.4f}\t{fr(1.1):.3f}\t{fr(2):.3f}\t{fr(10):.3f}\t{sum(math.isinf(x) for x in g)}\n")

# ---- performance profiles
fig, axs = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.6), sharey=True, squeeze=False)
taus = [10 ** (k / 50) for k in range(0, 201)]   # 1 .. 1e4
for ax, B in zip(axs[0], budgets):
    for m in methods:
        g = gaps(m, B)
        if not g:
            continue
        ys = [sum(x <= math.log10(t) + 1e-9 for x in g) / len(g) for t in taus]
        ax.plot(taus, ys, color=COL.get(m), lw=2 if m in ("hbtnd", "hbt") else 1.3, ls="--" if m in oneshot else "-", label=m)
    ax.set_xscale("log"); ax.set_title(f"budget {B:g} s"); ax.set_xlabel(r"$\tau$ (cost / best known)"); ax.grid(alpha=.3)
    ax.set_ylim(0, 1.02)
axs[0][0].set_ylabel("fraction of networks within factor $\\tau$")
axs[0][-1].legend(fontsize=7, loc="lower right")
fig.suptitle(f"[{CFG} {MODE}] Performance profiles on {len(nets)} einsum-benchmark networks (median of seeds; dashed = one-shot, counted if within budget)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(pdir, "profiles.png"), dpi=150); plt.close(fig)

# ---- cost vs time
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for m in methods:
    xs, y1, y2 = [], [], []
    pts = [(None if m in oneshot else B) for B in ([None] if m in oneshot else budgets)]
    for B in pts:
        vs = [value(n, m, B if B is not None else None) for n in nets]
        vs = [v for v in vs if v is not None]
        if not vs:
            continue
        g = [(v[0] - best[n]) if math.isfinite(v[0]) else math.inf for n, v in zip([n for n in nets if value(n, m, B) is not None], vs)]
        xs.append(statistics.median(v[1] for v in vs))
        y1.append(statistics.median(g)); y2.append(sum(x <= math.log10(2) + 1e-9 for x in g) / len(g))
    if not xs:
        continue
    kw = dict(color=COL.get(m), label=m, marker="o" if m not in oneshot else "s", lw=2 if m in ("hbtnd", "hbt") else 1.3,
              ls="none" if m in oneshot else "-", ms=6)
    axs[0].plot(xs, [max(v, 1e-3) for v in y1], **kw); axs[1].plot(xs, y2, **kw)
for ax in axs:
    ax.set_xscale("log"); ax.set_xlabel("median total wall time (s) = search + post-processing"); ax.grid(alpha=.3)
axs[0].set_yscale("log"); axs[0].set_ylabel("median gap to best (log10 C; floor 1e-3)")
axs[1].set_ylabel("fraction within 2x of best"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(pdir, "curves.png"), dpi=150); plt.close(fig)
print("wrote", pdir, "networks", len(nets), "rows", len(rows))
print(open(os.path.join(pdir, "summary.tsv")).read())
