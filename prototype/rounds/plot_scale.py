# log-log time vs n, one panel per family; fitted exponents in the legend
# usage: plot_scale.py out.png fam1,fam2 file.tsv[:label] ...
import sys, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
out = sys.argv[1]; fams = sys.argv[2].split(",")
cols = ["host","gpu","threads","fam","name","n","m","method","width","lcost","time","extra"]
pts = defaultdict(list)   # (fam, series) -> [(n, t)]
for spec in sys.argv[3:]:
    f, _, lab = spec.partition(":")
    for line in open(f):
        p = line.rstrip("\n").split("\t")
        if len(p) < 11 or p[0] == "host": continue
        r = dict(zip(cols, p))
        m = r["method"]
        if m in ("MF", "MF1", "MMD", "AMF"): s = f"serial {m}"
        elif m.startswith("gpu-"): s = f"rounds {m[4:]}, {lab or r['gpu']}"
        elif m.startswith("cpu-"): s = f"rounds {m[4:]}, CPU {r['threads']} thr"
        else: s = m
        if not math.isnan(float(r["width"])): pts[(r["fam"], s)].append((int(r["n"]), float(r["time"])))
order = ["serial MF", "serial MMD", "serial AMF"]
colors = {"serial MF": "#2a78d6", "serial MMD": "#eb6834", "serial AMF": "#1baf7a"}
gpucol = ["#6250d6", "#e87ba4", "#eda100", "#008300"]
def fit(xy, nmin):
    xy = [(x, y) for x, y in xy if x >= nmin]
    if len(xy) < 2: return float("nan")
    lx = [math.log10(x) for x, _ in xy]; ly = [math.log10(y) for _, y in xy]
    mx = sum(lx)/len(lx); my = sum(ly)/len(ly)
    return sum((a-mx)*(b-my) for a, b in zip(lx, ly)) / sum((a-mx)**2 for a in lx)
fig, axes = plt.subplots(1, len(fams), figsize=(5.2*len(fams), 4.2), squeeze=False)
for ax, fam in zip(axes[0], fams):
    series = sorted({s for (f, s) in pts if f == fam}, key=lambda s: (s not in order, order.index(s) if s in order else 0, s))
    series = [s for s in series if s in order or s.startswith("rounds mf,") or s.startswith("rounds md,")]
    gi = 0
    for s in series:
        xy = sorted(pts[(fam, s)])
        if s in colors: c, ls = colors[s], "--"
        else: c, ls = gpucol[gi % len(gpucol)], "-"; gi += 1
        b = fit(xy, 1e4)
        ax.plot([x for x, _ in xy], [y for _, y in xy], ls, color=c, lw=2, marker="o", ms=4, label=f"{s}  (slope {b:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("line-graph vertices n"); ax.set_ylabel("ordering time (s)")
    ax.set_title(fam); ax.grid(True, which="major", color="#ddd", lw=0.6)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(out, dpi=150)
print("wrote", out)
