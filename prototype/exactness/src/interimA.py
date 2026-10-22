import glob, sys, numpy as np
sys.path.insert(0, "/blue/fairbanksj/itaykadosh/gpu/exactness/src")
import tdfile
from scipy.stats import spearmanr
L2 = np.log10(2)
rows = []
for f in sorted(glob.glob("/blue/fairbanksj/itaykadosh/gpu/exactness/resA/td/*.td")):
    td = tdfile.parse(f)
    if not td["exact"] or not td["exact"]["certified"]: continue
    opt = td["exact"]["width"]; O = td["orders"]
    w = np.array([d["width"] for d in O]); c = np.array([d["br_cost"] for d in O]) * L2; cb = np.array([d["bt_cost"] for d in O]) * L2
    tags = [d["tag"] for d in O]
    ex = np.array([t.startswith("exact") for t in tags]); heur = ~ex
    isopt = w <= opt + 1e-6
    rho = spearmanr(w, c).correlation if np.ptp(w) > 0 else np.nan
    i = np.argmin(c)
    rows.append((td["net"]["name"], len(O), isopt.sum(), ex.sum(), rho, w[i] - opt, c[ex].min() - c[heur].min(), c[tags.index("exact")] - c[heur].min(), c[isopt].max() - c[isopt].min(), tags[i]))
print("net nTD nopt nexact rho gapw_of_cheapest d(exactbest-heur) d(exact0-heur) spread_opt cheapest_tag")
for r in rows: print(r[0][:40], r[1], r[2], r[3], f"{r[4]:.2f}", f"{r[5]:.2f}", f"{r[6]:+.3f}", f"{r[7]:+.3f}", f"{r[8]:.3f}", r[9])
a = np.array([r[4:9] for r in rows], dtype=float)
print("N", len(rows), "median rho", np.nanmedian(a[:,0]), "cheapest is width-opt", (a[:,1] < 1e-6).sum(), "exact-best vs heur W/T/L", (a[:,2] < -0.01).sum(), (abs(a[:,2]) <= 0.01).sum(), (a[:,2] > 0.01).sum(), "median d", np.median(a[:,2]), "median spread", np.median(a[:,4]))
