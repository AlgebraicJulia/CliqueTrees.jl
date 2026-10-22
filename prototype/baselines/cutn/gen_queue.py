"""usage: python gen_queue.py <outdir>  -> <outdir>/tasks_einsum.txt, tasks_frontier.txt  (line: source net maxT cap)
einsum: the 163 paper=1 einsum-benchmark networks (tnc.einsum_names()), unsliced, maxT=100 (ladder covers 1/10/100 s).
frontier: syc53 amp/marg6/raw m12..m20, marg21, open6huang, syc70_m24, syc67_m32: unsliced + the width caps of our
runs (make_tasks.py: amp 27,30; marg6/open6huang 29; marg21 30; raw none); maxT=1000 for the long set else 100."""
import os, sys
sys.path.insert(0, os.environ.get("CTGPLUGIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ctgplugin")))
import tnc
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
ein = tnc.einsum_names()
with open(f"{out}/tasks_einsum.txt", "w") as f:
    for n in ein:
        f.write(f"einsum {n} 100 0\n")
AMP = [f"syc53_m{m}_amp" for m in (12, 14, 16, 18, 20)]
MARG = [f"syc53_m{m}_marg6" for m in (12, 14, 16, 18, 20)] + ["syc53_m20_open6huang"]
RAW = [f"syc53_m{m}_raw" for m in (12, 14, 16, 18, 20)]
LONG = {"syc53_m20_amp", "syc53_m18_amp", "syc53_m20_marg6", "syc53_m20_marg21", "syc70_m24_amp", "syc67_m32_amp"}
CAPS = lambda c: [(c, f) for f in (1, 2, 4)]
rows = []
for n in AMP: rows += [(n, 0, 1)] + [(n,) + x for x in CAPS(27) + CAPS(30)]
for n in MARG: rows += [(n, 0, 1)] + [(n,) + x for x in CAPS(29)]
rows += [("syc53_m20_marg21", 0, 1)] + [("syc53_m20_marg21",) + x for x in CAPS(30)]
for n in RAW: rows += [(n, 0, 1)]
for n in ("syc70_m24_amp", "syc67_m32_amp"): rows += [(n, 0, 1)] + [(n,) + x for x in CAPS(30)]
rows.sort(key=lambda r: -(1000 if (r[0] in LONG and r[1] == 0) else 100))
with open(f"{out}/tasks_frontier.txt", "w") as f:
    for n, c, mf in rows:
        f.write(f"frontier {n} {1000 if (n in LONG and c == 0) else 100} {c} {mf}\n")
print(len(ein), "einsum;", len(rows), "frontier")
