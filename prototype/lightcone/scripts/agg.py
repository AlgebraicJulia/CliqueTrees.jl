import json, glob, os, numpy as np, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = [json.load(open(f)) for f in glob.glob(HERE + "/results/*.json")]
R.sort(key=lambda r: (r["kind"], r["N"], r["p"]))
OPTS = ["greedy", "auto", "auto-hq", "hyper"]
# contraction time model t = a*nops + b*cost (nonneg lstsq) from every measured contraction
def fit(key):
    X, y = [], []
    for r in R:
        for c in r["ct"]:
            for o in ("auto", "auto-hq", "hyper"):
                s = c.get(o)
                if s and key in s and s[key] > 0: X.append([s["nops"], s["cost"]]); y.append(s[key])
    X = np.array(X, float); y = np.array(y)
    # weighted to relative error
    A = X / y[:, None]; b = np.ones(len(y))
    from itertools import product
    best = None
    for use in ([0, 1], [0], [1]):
        co, *_ = np.linalg.lstsq(A[:, use], b, rcond=None)
        if (co < 0).any(): continue
        full = np.zeros(2); full[use] = co
        err = np.abs(A @ full - 1).mean()
        if best is None or err < best[0]: best = (err, full)
    return best[1], best[0], len(y)
MODEL = {k: fit(k) for k in ("t_cpu", "t_gpu")}
def model(key, s): a, b = MODEL[key][0]; return a * s["nops"] + b * s["cost"]
def wmean(rows, f, wk):
    ws = np.array([r[wk] for r in rows], float); v = np.array([f(r) for r in rows], float)
    return float((ws * v).sum() / ws.sum())
out = []
P = out.append
P("### Contraction-time model (fit on all measured contractions, relative error)\n")
for k, (co, err, n) in MODEL.items():
    P(f"- {k}: t = {co[0]*1e6:.2f} us * (#pairwise contractions) + {co[1]*1e9:.3f} ns * (scalar mult-adds); mean rel. error {err:.2f} over {n} measurements")
P("")
# T1
P("### T1. Terms, distinct light-cone networks, line-graph sizes (distinct nets)\n")
P("| graph | N | p | terms | terms sampled | distinct (sampled) | distinct (est. all terms) | quimb extract s/term | lg n min/med/max | lg m min/med/max | <=128 v | <=512 v |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in R:
    P(f"| {r['kind']} | {r['N']} | {r['p']} | {r['nterms']} | {r['nsampled']} | {r['ndistinct']} | {r['ndistinct_est']} | {r['extract_per_term']:.3f} | {r['lg_n']['min']}/{r['lg_n']['med']:.0f}/{r['lg_n']['max']} | {r['lg_m']['min']}/{r['lg_m']['med']:.0f}/{r['lg_m']['max']} | {r['lg_n']['le128']} | {r['lg_n']['le512']} |")
P("\n(lg = line graph: one vertex per tensor index; edge = two indices sharing a tensor. The <=128/<=512 columns count among the sampled-term distinct nets.)\n")
# T2 cotengra
P("### T2. cotengra per distinct network (quantile sample of <=10 nets per setting; mean over the sample, dedup-weighted)\n")
P("search s = wall seconds of the optimizer (auto / auto-hq are capped at 60 s, 90 s in the first runs; a capped net contributes its cap as search time and the stats of the greedy tree); logC = log10 scalar mult-adds; W = log2 max intermediate. Hyper = HyperOptimizer(max_time=0.5).\n")
P("| graph N p | #sampled | greedy s / logC / W | auto s / logC / W | auto-hq s / logC / W | hyper0.5 s / logC / W |")
P("|---|---|---|---|---|---|")
for r in R:
    rows = [c for c in r["ct"] if "skipped" not in c]
    if not rows: P(f"| {r['kind']} {r['N']} {r['p']} | 0 (too large) | | | | |"); continue
    cells = []
    for o in OPTS:
        nto = sum(1 for c in rows if c[o].get("timeout"))
        cells.append(f"{wmean(rows, lambda c: c[o]['t'], 'w_dedup'):.3f} / {wmean(rows, lambda c: c[o]['logC'], 'w_dedup'):.2f} / {wmean(rows, lambda c: c[o]['W'], 'w_dedup'):.1f}" + (f" ({nto} capped)" if nto else ""))
    P(f"| {r['kind']} {r['N']} {r['p']} | {len(rows)}/{len(r['ct'])} | " + " | ".join(cells) + " |")
# T3 contraction measured
P("\n### T3. Actual contraction time (numpy complex128, 4 BLAS threads; cupy complex128 on RTX 5060 Laptop), nets with logC<=9.3 and W<=24, auto-hq tree\n")
P("| graph N p | #measured | cpu ms (mean) | gpu ms (mean) | mean logC measured | note |")
P("|---|---|---|---|---|---|")
for r in R:
    m = [c["auto-hq"] for c in r["ct"] if "auto-hq" in c and "t_cpu" in c["auto-hq"]]
    allc = [c["auto-hq"] for c in r["ct"] if "auto-hq" in c]
    if m:
        P(f"| {r['kind']} {r['N']} {r['p']} | {len(m)}/{len(allc)} | {1e3*np.mean([x['t_cpu'] for x in m]):.2f} | {1e3*np.mean([x.get('t_gpu', np.nan) for x in m]):.2f} | {np.mean([x['logC'] for x in m]):.2f} | |")
    else:
        P(f"| {r['kind']} {r['N']} {r['p']} | 0/{len(allc)} | - | - | - | too expensive for laptop numpy (median logC {np.median([x['logC'] for x in allc]) if allc else float('nan'):.1f}) |")
# helpers
def okc(c, o): return c[o] if (o in c and not c[o].get("timeout")) else None
def best_cot(c):
    v = [c[o] for o in OPTS if okc(c, o)]
    return min(x["logC"] for x in v), min(x["W"] for x in v)
def bestcost_cost(c, ours_row):
    """scalar cost (mult-adds) and nops of the cheapest tree known for this net (cotengra uncensored + ours)"""
    cands = [c[o] for o in OPTS if okc(c, o)]
    if ours_row: cands += [ours_row[m] for m in ("hbt1", "hbtnd1", "exact", "hbt0.2", "bw") if m in ours_row and "cost" in ours_row[m]]
    return min(cands, key=lambda x: x["cost"])
HOUR = 3600.0
P("\n### T4. Total path-search time vs total contraction time\n")
P("Per setting: totals over all terms (no dedup) and over all distinct nets (dedup; distinct count extrapolated for partially-sampled settings, see T1). "
  "search = optimizer wall time per net (single thread); contract = modelled numpy-CPU / cupy-GPU time of the optimizer's own tree (a net whose search was capped uses the cheapest tree known for it, so its contraction term is optimistic and its search term a lower bound). "
  "'infeas' = fraction of distinct nets whose cheapest known tree needs > 1 h of modelled CPU contraction (exact light-cone contraction is out of reach on the laptop regardless of search). "
  "ratio = search / contract; shares are search/(search+contract) over the FEASIBLE nets only, dedup. Hyper-0.5s is omitted (its trees are erratic on big nets, see T2).\n")
P("| graph N p | opt | infeas | search no-dedup s | search dedup s | contract-CPU dedup s (feasible) | contract-GPU dedup s (feasible) | ratio CPU, no dedup | ratio CPU, dedup | ratio GPU, dedup | search share CPU dedup | search share GPU dedup |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
f3 = lambda x: f"{x:.3g}"
OURM = {}
for r in R:
    rows = [c for c in r["ct"] if "skipped" not in c]
    if not rows: continue
    om = {o["h"]: o for o in r["ours"]}
    for o in ("auto", "auto-hq"):
        def own(c): return c[o] if not c[o].get("timeout") else bestcost_cost(c, om.get(c["h"]))
        sn = wmean(rows, lambda c: c[o]["t"], "w_all") * r["nterms"]; sd = wmean(rows, lambda c: c[o]["t"], "w_dedup") * r["ndistinct_est"]
        feas = [c for c in rows if model("t_cpu", bestcost_cost(c, om.get(c["h"]))) <= HOUR]
        infeas = 1 - sum(c["w_dedup"] for c in feas) / sum(c["w_dedup"] for c in rows)
        # contraction totals (all nets; infeasible ones included -> astronomically large, shown in ratio columns for no-dedup/dedup)
        cn = wmean(rows, lambda c: model("t_cpu", own(c)), "w_all") * r["nterms"]; cd_all = wmean(rows, lambda c: model("t_cpu", own(c)), "w_dedup") * r["ndistinct_est"]
        if feas:
            cdf = wmean(feas, lambda c: model("t_cpu", own(c)), "w_dedup") * r["ndistinct_est"] * (1 - infeas)
            gdf = wmean(feas, lambda c: model("t_gpu", own(c)), "w_dedup") * r["ndistinct_est"] * (1 - infeas)
            sdf = wmean(feas, lambda c: c[o]["t"], "w_dedup") * r["ndistinct_est"] * (1 - infeas)
            shc, shg = sdf / (sdf + cdf), sdf / (sdf + gdf)
        else: cdf = gdf = float("nan"); shc = shg = float("nan")
        gd_all = wmean(rows, lambda c: model("t_gpu", own(c)), "w_dedup") * r["ndistinct_est"]
        P(f"| {r['kind']} {r['N']} {r['p']} | {o} | {infeas:.2f} | {f3(sn)} | {f3(sd)} | {f3(cdf)} | {f3(gdf)} | {f3(sn/cn)} | {f3(sd/cd_all)} | {f3(sd/gd_all)} | {shc:.3f} | {shg:.3f} |")
# ours as a path source: hbt1 search + bucket tree; contraction from its tree (mean over our subset)
P("\n#### T4b. Our HBT (T=1 s, 1 thread) as the path source, same nets (subset of <=5 per setting; mean per net; contraction modelled)\n")
P("| graph N p | #nets | HBT1 search s/net (+bucket tree s) | contract CPU s/net | contract GPU s/net | search/contract CPU | search/contract GPU | cotengra auto-hq on same nets: search s/net, contract CPU s/net (uncapped nets only) |")
P("|---|---|---|---|---|---|---|---|")
for r in R:
    ours = [o for o in r["ours"] if "hbt1" in o]
    if not ours: continue
    ctm = {c["h"]: c for c in r["ct"]}
    ts = np.mean([o["hbt1"]["t"] + o["hbt1"]["t_tree"] for o in ours]); tb = np.mean([o["hbt1"]["t_tree"] for o in ours])
    cc = np.mean([model("t_cpu", o["hbt1"]) for o in ours]); cg = np.mean([model("t_gpu", o["hbt1"]) for o in ours])
    hq = [ctm[o["h"]]["auto-hq"] for o in ours if okc(ctm[o["h"]], "auto-hq")]
    hs = f"{np.mean([x['t'] for x in hq]):.3g}, {np.mean([model('t_cpu', x) for x in hq]):.3g} ({len(hq)}/{len(ours)})" if hq else "all capped"
    P(f"| {r['kind']} {r['N']} {r['p']} | {len(ours)} | {ts:.3g} ({tb:.2g}) | {f3(cc)} | {f3(cg)} | {f3(ts/cc)} | {f3(ts/cg)} | {hs} |")

P("\n### T6. Where the wall time goes, per setting (dedup, totals over all distinct nets; HBT row from the <=5-net subset mean x #distinct)\n")
P("extract = quimb light-cone extraction + simplification of ALL terms (needed even with dedup, to find the duplicates); search/contract columns as in T4/T4b; contract = modelled on the HBT1 tree, feasible nets only if any (else n/a: >1 h per net).\n")
P("| graph N p | #terms | #distinct (est) | extract s | auto-hq search s | HBT1 search s | contract CPU s (HBT1 trees) | contract GPU s (HBT1 trees) | nets feasible (<=1 h CPU) |")
P("|---|---|---|---|---|---|---|---|---|")
for r in R:
    rows = [c for c in r["ct"] if "skipped" not in c]
    ours = [o for o in r["ours"] if "hbt1" in o]
    if not rows or not ours: continue
    nd = r["ndistinct_est"]
    hq = wmean(rows, lambda c: c["auto-hq"]["t"], "w_dedup") * nd
    hb = np.mean([o["hbt1"]["t"] + o["hbt1"]["t_tree"] for o in ours]) * nd
    fe = [o for o in ours if model("t_cpu", o["hbt1"]) <= HOUR]
    fr = len(fe) / len(ours)
    cc = np.mean([model("t_cpu", o["hbt1"]) for o in fe]) * nd * fr if fe else float("nan")
    cg = np.mean([model("t_gpu", o["hbt1"]) for o in fe]) * nd * fr if fe else float("nan")
    P(f"| {r['kind']} {r['N']} {r['p']} | {r['nterms']} | {nd} | {f3(r['extract_per_term']*r['nterms'])} | {f3(hq)} | {f3(hb)} | {f3(cc)} | {f3(cg)} | {len(fe)}/{len(ours)} |")
# T5 ours vs cotengra
P("\n### T5. Our methods vs cotengra on the same nets (sample of <=5 nets per setting spanning the size range)\n")
P("exact = SafeRules+SafeSeparators+PIDBT (12 s cap, stop after first timeout); 'proven' = exact finished OR HBT bag width == MMW lower bound. Costs are of the contraction trees obtained by bucket elimination from our order (ftn bridge, no subtree refinement). "
  "'ours best' = cheaper of hbt1 / hbtnd1 / exact (by logC). beat/tie/lose counts use tol 0.01 in logC and 0.5 in W (W = log2 largest intermediate). "
  "vs hq: only nets where auto-hq finished within the cap. vs best-cot: best of greedy/auto/auto-hq/hyper0.5 among the uncapped ones.\n")
P("| graph N p | #nets | lg n range | exact finished | width proven | exact time s (max) | HBT1 = exact width (of finished) | exact tree cheaper than HBT1 tree (of finished) | ours vs hq logC b/t/l (n) | ours vs best-cot logC b/t/l | ours vs best-cot W b/t/l | mean dlogC vs hq | mean dlogC vs best-cot | max dlogC vs best-cot (best/worst) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
def cmp(a, b, tol=0.01):
    d = np.array(a) - np.array(b); return f"{(d<-tol).sum()}/{(abs(d)<=tol).sum()}/{(d>tol).sum()}"
for r in R:
    ours = r["ours"]
    if not ours: continue
    ctm = {c["h"]: c for c in r["ct"]}
    fin = [o for o in ours if "exact" in o and "timeout" not in o["exact"]]
    proven = [o for o in ours if ("exact" in o and "timeout" not in o["exact"]) or any(o.get(m, {}).get("bag") == o["lb"] for m in ("hbt1", "hbtnd1", "bw"))]
    agree = sum(1 for o in fin if min(o["hbt1"]["bag"], o["hbtnd1"]["bag"]) == o["exact"]["bag"])
    cheaper = sum(1 for o in fin if o["exact"]["logC"] < min(o["hbt1"]["logC"], o["hbtnd1"]["logC"]) - 0.01)
    ob, obW, bc, bcW, hq_o, hq_c = [], [], [], [], [], []
    for o in ours:
        c = ctm[o["h"]]
        cands = [o[m] for m in ("hbt1", "hbtnd1", "exact") if m in o and "logC" in o[m]]
        ob.append(min(x["logC"] for x in cands)); obW.append(min(x["W"] for x in cands))
        l, w = best_cot(c); bc.append(l); bcW.append(w)
        if okc(c, "auto-hq"): hq_o.append(ob[-1]); hq_c.append(c["auto-hq"]["logC"])
    d = np.array(ob) - np.array(bc)
    et = max([o["exact"]["t"] for o in fin], default=float("nan"))
    ns = [o["n"] for o in ours]
    P(f"| {r['kind']} {r['N']} {r['p']} | {len(ours)} | {min(ns)}-{max(ns)} | {len(fin)}/{len(ours)} | {len(proven)}/{len(ours)} | {et:.3g} | {agree}/{len(fin)} | {cheaper}/{len(fin)} | {cmp(hq_o,hq_c) if hq_o else '-'} ({len(hq_o)}) | {cmp(ob,bc)} | {cmp(obW,bcW,0.5)} | {np.mean(np.array(hq_o)-np.array(hq_c)):+.2f} | {d.mean():+.2f} | {d.min():+.2f} / {d.max():+.2f} |" if hq_o else
      f"| {r['kind']} {r['N']} {r['p']} | {len(ours)} | {min(ns)}-{max(ns)} | {len(fin)}/{len(ours)} | {len(proven)}/{len(ours)} | {et:.3g} | {agree}/{len(fin)} | {cheaper}/{len(fin)} | - (0) | {cmp(ob,bc)} | {cmp(obW,bcW,0.5)} | - | {d.mean():+.2f} | {d.min():+.2f} / {d.max():+.2f} |")
open(HERE + "/tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
