"""python analyzeStrong.py <strong1.tsv> <strong2.tsv> > table   (pandas).  ours60 vs: ctg_hyper60 (old), strong cotengra (best predicted C /
best measured time over the executed configs sf60 sc60 sc256_60 sr60 sc600 sc256_600 sf600), cutn60 (cuTensorNet own path, 2^30 cap)."""
import sys, numpy as np, pandas as pd
cols = "net tag C W nsl search t_med t_min t_all relerr ref status".split()
def rd(p):
    try: return pd.read_csv(p, sep="\t", names=cols, header=0, dtype={"t_all": str})
    except Exception: return pd.DataFrame(columns=cols)
s1, s2 = rd(sys.argv[1]), rd(sys.argv[2])
d = pd.concat([s1.assign(job=1), s2.assign(job=2)])
d = d[d.status == "ok"].copy(); d["t_med"] = d.t_med.astype(float)
STRONG = ["sf60", "sc60", "sc256_60", "sr60", "sc600", "sc256_600", "sf600"]
def best(g, tags, by):
    h = g[g.tag.isin(tags)]
    return h.loc[h[by].idxmin()] if len(h) else None
rows = []
for net, g in d.groupby("net"):
    for ours_job in (1, 2):
        pass
    o1 = g[(g.tag == "ours60") & (g.job == 1)]; o2 = g[(g.tag == "ours60") & (g.job == 2)]
    for name, tags, by, job in (("ctg_old", ["ctg_hyper60"], "t_med", 1), ("cutn60", ["cutn60"], "t_med", 1),
                                ("ctg_strong_bestC", STRONG, "C", 2), ("ctg_strong_bestT", STRONG, "t_med", 2),
                                ("ctg_strong60_bestT", [t for t in STRONG if t.endswith("60")], "t_med", 2)):
        o = o1 if job == 1 else o2
        if not len(o): continue
        c = best(g[g.job == job], tags, by)
        if c is None: continue
        o = o.iloc[0]
        rows.append(dict(cmp=name, net=net, tag=c.tag, ours_t=o.t_med, cmp_t=c.t_med, ours_C=o.C, cmp_C=c.C, ours_nsl=o.nsl, cmp_nsl=c.nsl,
                         cmp_W=c.W, sp=c.t_med / o.t_med, cmp_search=c.search, ours_search=o.search))
r = pd.DataFrame(rows)
r["cls"] = np.where((r.ours_nsl == 1) & (r.cmp_nsl == 1), "a unsliced both", np.where(r.cmp_nsl > 1, "b baseline sliced", "c other"))
def summ(x):
    sp = x.sp
    return dict(n=len(x), W=int((sp > 1.05).sum()), T=int(((sp >= .95) & (sp <= 1.05)).sum()), L=int((sp < .95).sum()),
                median=round(sp.median(), 2), geomean=round(float(np.exp(np.log(sp).mean())), 2))
out = []
for k, x in r.groupby("cmp"):
    out.append(dict(cmp=k, cls="all", **summ(x)))
    for c, y in x.groupby("cls"): out.append(dict(cmp=k, cls=c, **summ(y)))
print(pd.DataFrame(out).to_string(index=False))
print(); print(r.round(3).to_string(index=False))
