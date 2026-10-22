"""usage: python gen_queue.py <outdir> [--seeds 1]  -> <outdir>/tasks_td.txt   (line: source net method budget seed)
einsum: 163 paper=1 networks x {to-flow,to-tamaki,to-htd,qt-tamaki} x budgets {1,10,100} + qt-greedy (budget 0).
frontier: unsliced syc53 amp/marg6/raw m12..m20, marg21, open6huang, syc70_m24, syc67_m32 x same + budget 1000.
Longest budgets first."""
import os, sys
sys.path.insert(0, os.environ.get("CTGPLUGIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ctgplugin")))
import tnc
out = sys.argv[1]; seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 1
os.makedirs(out, exist_ok=True)
M = ("to-flow", "to-tamaki", "to-htd", "qt-tamaki")
rows = []
for n in tnc.einsum_names():
    rows += [("einsum", n, "qt-greedy", 0, 1)] + [("einsum", n, m, T, s) for m in M for T in (1, 10, 100) for s in range(1, seeds + 1)]
F = [f"syc53_m{m}_{k}" for m in (12, 14, 16, 18, 20) for k in ("amp", "marg6", "raw")] + ["syc53_m20_marg21", "syc53_m20_open6huang", "syc70_m24_amp", "syc67_m32_amp"]
for n in F:
    rows += [("frontier", n, "qt-greedy", 0, 1)] + [("frontier", n, m, T, 1) for m in M for T in (1, 10, 100, 1000)]
rows.sort(key=lambda r: -r[3])
with open(f"{out}/tasks_td.txt", "w") as f:
    for r in rows:
        f.write(" ".join(map(str, r)) + "\n")
print(len(rows), "tasks, nominal budget sum %.0f s" % sum(r[3] for r in rows))
