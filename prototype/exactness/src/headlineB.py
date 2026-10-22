"""Headline of experiment B under numerical-validity filters.  python headlineB.py <gpu_torch.tsv> [extra_rerun.tsv]
Rerun rows (2nd file) replace the rows of the same (net, tag) in the first.
modes: all | row (drop rows relerr>1e-3) | row4 (relerr>1e-4) | net (drop a network if ANY of its trees has relerr>1e-3)"""
import sys, math
from collections import defaultdict
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from analyzeB import read, fl
rows = {(r["net"], r["tag"]): r for r in read(sys.argv[1])}
if len(sys.argv) > 2:
    for r in read(sys.argv[2]):
        if r["status"] == "ok":
            rows[(r["net"], r["tag"])] = r
rows = [r for r in rows.values() if r["status"] == "ok"]
badnet = {r["net"] for r in rows if fl(r["relerr"]) > 1e-3}
for mode in ("all", "row", "row4", "net"):
    keep = [r for r in rows if mode == "all" or (mode == "row" and not fl(r["relerr"]) > 1e-3)
            or (mode == "row4" and not fl(r["relerr"]) > 1e-4) or (mode == "net" and r["net"] not in badnet)]
    by = defaultdict(dict)
    for r in keep:
        by[r["net"]][r["tag"].replace("td_hyb300", "td_exact")] = r
    print(f"## mode={mode}: {len(keep)} rows, {len(by)} networks")
    for a, b in (("ours60", "ctg_hyper60"), ("ours10", "ctg_hyper10"), ("td_exact", "ctg_hyper60")):
        W = [0, 0, 0]; tts = []; tc = []; tt = []
        for n, d in by.items():
            if a in d and b in d:
                rt = fl(d[a]["t_med"]) / fl(d[b]["t_med"])
                W[0 if rt < 0.95 else (1 if rt <= 1.05 else 2)] += 1
                tts.append((fl(d[a]["search_s"]) + fl(d[a]["t_med"])) / (fl(d[b]["search_s"]) + fl(d[b]["t_med"])))
                tc.append(fl(d[a]["log10C"]) - fl(d[b]["log10C"])); tt.append(math.log10(rt))
        print(f"  {a} vs {b}: faster/tie/slower {W[0]}/{W[1]}/{W[2]} (n={sum(W)}), median log10C diff {np.median(tc):+.3f}, "
              f"median log10 t ratio {np.median(tt):+.3f} (x{10**-np.median(tt):.1f} faster), median time-to-solution ratio {np.median(tts):.2f}")
