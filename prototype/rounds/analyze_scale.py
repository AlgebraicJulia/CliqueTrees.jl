# fit time exponents and crossovers from scale TSVs
# usage: analyze_scale.py nfit file1.tsv [file2.tsv ...]
import sys, math, re
from collections import defaultdict
nfit = float(sys.argv[1])
cols = ["host","gpu","threads","fam","name","n","m","method","width","lcost","time","extra"]
data = defaultdict(dict)   # (fam) -> name -> method -> (n, width, time, extra)
for f in sys.argv[2:]:
    for line in open(f):
        p = line.rstrip("\n").split("\t")
        if len(p) < 11 or p[0] == "host": continue
        r = dict(zip(cols, p))
        meth = r["method"]
        if meth.startswith("cpu-") or meth.startswith("fcpu-"): meth = f"{meth}@t{r['threads']}"
        data[r["fam"]].setdefault(r["name"], {})[meth] = (int(r["n"]), float(r["width"]), float(r["time"]), r.get("extra",""))
def fit(xs, ys):
    if len(xs) < 2: return float("nan"), float("nan")
    lx = [math.log10(x) for x in xs]; ly = [math.log10(y) for y in ys]
    k = len(lx); mx = sum(lx)/k; my = sum(ly)/k
    sxx = sum((a-mx)**2 for a in lx)
    if k < 2 or sxx == 0: return float('nan'), float('nan')
    b = sum((a-mx)*(c-my) for a, c in zip(lx, ly))/sxx
    return b, my - b*mx
pair = {"gpu-md": "MMD", "gpu-mf": "MF", "gpu-mf1": "MF1"}
for fam, insts in data.items():
    meths = sorted({m for d in insts.values() for m in d})
    print(f"\n### {fam}  ({len(insts)} instances)")
    for m in meths:
        pts = [(d[m][0], d[m][2]) for d in insts.values() if m in d and d[m][0] >= nfit and d[m][2] > 0 and not math.isnan(d[m][1])]
        pts.sort()
        b, a = fit([p[0] for p in pts], [p[1] for p in pts])
        allp = sorted((d[m][0], d[m][2]) for d in insts.values() if m in d)
        rng = f"n {allp[0][0]}..{allp[-1][0]}" if allp else ""
        print(f"  {m:14s} exponent {b:5.2f} (n>={nfit:g}, {len(pts)} pts)  {rng}  t(nmax)={allp[-1][1] if allp else float('nan'):.4g}s")
    for g, s in pair.items():
        rows = sorted((d[g][0], d[g][2], d[s][2], d[g][1], d[s][1], d[g][3]) for d in insts.values() if g in d and s in d)
        if not rows: continue
        cross = None
        for i in range(len(rows)):
            if all(r[1] < r[2] for r in rows[i:]): cross = rows[i][0]; break
        print(f"  {g} vs {s}: crossover n={cross}; " + " ".join(f"[{r[0]}: {r[2]/r[1]:.1f}x, dW={r[3]-r[4]:+.1f}]" for r in rows[-4:]))
