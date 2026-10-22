"""Harvest + summarise the baseline runs.
usage: python harvest.py [--fetch] [--local DIR]       (needs pandas; any env, e.g. tncorpus venv)
--fetch rsyncs /blue/.../baselines/tn/{full_cutn,full_td}/res (and einsumbench/run1/res if present) into <DIR>/{cutn,td,ours}.
Outputs <DIR>/summary_*.tsv:
  cutn_budget.tsv : per (source, network, cap, memf, budget T in {1,10,100,1000,default}) best row (min logC among rows with search_s <= T;
                    default = OptimizerOptions() row), plus search_s, width, nslices.
  td_budget.tsv   : per (network, method, T) median over seeds of logC/width/wall.
  wmc_compare.tsv : mc_*/wmc_* subset: cutn, TensorOrder, QTensor, and (if run1 harvested) ours/cotengra at the same T.
Cap handling for cutn: for cap W rows only 'cap_met' (width<=W) rows count, taking the best over memf in {1,2,4}.
Units: logC = log10 cotengra contraction_cost (multiply-adds, summed over slices); width = log2 largest intermediate per slice.
"""
import argparse, glob, os, subprocess, sys
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--fetch", action="store_true"); ap.add_argument("--local", default="harvest")
a = ap.parse_args()
R = "hpg:/blue/fairbanksj/itaykadosh/gpu/baselines/tn"
D = a.local
if a.fetch:
    for src, dst in ((f"{R}/full_cutn/res/", "cutn"), (f"{R}/full_td/res/", "td"), ("hpg:/blue/fairbanksj/itaykadosh/gpu/einsumbench/run1/res/", "ours")):
        os.makedirs(f"{D}/{dst}", exist_ok=True)
        subprocess.run(["rsync", "-a", src, f"{D}/{dst}/"], check=False)


def rd(pat, **k):
    fs = sorted(glob.glob(pat))
    return pd.concat([pd.read_csv(f, sep="\t", **k) for f in fs], ignore_index=True) if fs else pd.DataFrame()


c = rd(f"{D}/cutn/cutn_*.tsv")
if len(c):
    c = c[c.status == "ok"].copy()
    c["memf"] = c.extra.str.extract(r"memf=([0-9.]+)").astype(float)
    c["fallback"] = c.extra.str.contains("fallback_sliced")
    rows = []
    for (src, net, cap), g in c.groupby(["source", "network", "cap"]):
        if cap:
            g = g[g.cap_met == 1]
        for T in (1, 10, 100, 1000, "default"):
            h = g[g.samples == "default"] if T == "default" else g[(g.samples != "default") & (g.search_s <= T)]
            if len(h):
                b = h.loc[h.logC.idxmin()]
                rows.append(dict(source=src, network=net, cap=cap, T=T, logC=b.logC, width=b.width, nslices=b.nslices, search_s=b.search_s,
                                 samples=b.samples, memf=b.memf, fallback=b.fallback, cutn_log10flops=b.cutn_log10flops))
    pd.DataFrame(rows).to_csv(f"{D}/summary_cutn_budget.tsv", sep="\t", index=False)
    print("cutn rows", len(c), "->", len(rows))
t = rd(f"{D}/td/td_*.tsv", names="network method budget seed cfg status wall search post logC width maxop pf_logC pf_width greedy_s extra".split(), header=None)
if len(t):
    t["ok"] = t.status == "ok"
    g = t.groupby(["network", "method", "budget"]).agg(logC=("logC", "median"), width=("width", "median"), wall=("wall", "median"), ok=("ok", "mean")).reset_index()
    g.to_csv(f"{D}/summary_td_budget.tsv", sep="\t", index=False)
    print("td rows", len(t), "->", len(g))
    w = g[g.network.str.startswith(("mc_", "wmc_", "mc3sat"))]
    w.to_csv(f"{D}/wmc_td.tsv", sep="\t", index=False)
    o = rd(f"{D}/ours/*.tsv", names="network method budget seed cfg status wall search post logC width maxop pf_logC pf_width greedy_s extra".split(), header=None)
    if len(o):
        o = o[o.network.str.startswith(("mc_", "wmc_"))]
        o = o.groupby(["network", "method", "budget"]).logC.median().reset_index().rename(columns={"method": "m_other", "logC": "logC_other"})
        j = w.merge(o, on=["network", "budget"])
        j.to_csv(f"{D}/wmc_compare.tsv", sep="\t", index=False); print("wmc compare rows", len(j))
