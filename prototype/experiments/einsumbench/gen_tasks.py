"""Write the task queues of the einsum benchmark run.
usage: python gen_tasks.py <outdir> [--nets N1,N2,...|all|first:K|pick:K] [--budgets 0.1,1,10,100] [--seeds 3]
                           [--methods ours,ctg] [--include-oneshot 1]
Queues: <outdir>/tasks_ours.txt (Julia-backed: needs a t-thread slot) and <outdir>/tasks_ctg.txt (single-core).
Line: <network> <method> <budget> <seed>. Longest budgets first, so the tail of the run is short.
Budgeted methods (budget x seed grid): hbtnd hbt treesa (ours/OMEC) and hyper hyper-sa (cotengra).
One-shot methods (budget 0, run once, seed 1; counted at budget B only if their time <= B):
bw nd (ours, ms) and greedy auto auto-hq (cotengra)."""
import argparse
import os
import sys

sys.path.insert(0, os.environ.get("CTGPLUGIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ctgplugin")))
import tnc  # noqa: E402

OURS_B, CTG_B = ("hbtnd", "hbt", "treesa"), ("hyper", "hyper-sa")
OURS_1, CTG_1 = ("bw", "nd"), ("greedy", "auto", "auto-hq")

ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--nets", default="all")
ap.add_argument("--budgets", default="0.1,1,10,100")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--methods", default="ours,ctg")
ap.add_argument("--include-oneshot", type=int, default=1)
a = ap.parse_args()

allnets = tnc.einsum_names()
if a.nets == "all":
    nets = allnets
elif a.nets.startswith("first:"):
    nets = allnets[:int(a.nets[6:])]
elif a.nets.startswith("pick:"):      # K networks spread over the size range
    k = int(a.nets[5:]); nets = [allnets[round(i * (len(allnets) - 1) / max(1, k - 1))] for i in range(k)]
else:
    nets = a.nets.split(",")
budgets = [float(b) for b in a.budgets.split(",")]
os.makedirs(a.outdir, exist_ok=True)
groups = {"ours": (OURS_B, OURS_1), "ctg": (CTG_B, CTG_1)}
for g in a.methods.split(","):
    bm, om = groups[g]
    rows = [(n, m, b, s) for b in sorted(budgets, reverse=True) for m in bm for s in range(1, a.seeds + 1) for n in nets]
    if a.include_oneshot:
        rows = [(n, m, 0, 1) for m in om for n in nets] + rows
    # one-shot first would delay long ones; put them last instead (they are cheap fillers) except auto-hq (can be long)
    rows.sort(key=lambda r: -(r[2] if r[2] else (60 if r[1] == "auto-hq" else 0)))
    with open(os.path.join(a.outdir, f"tasks_{g}.txt"), "w") as f:
        for r in rows:
            f.write(" ".join(map(str, r)) + "\n")
    print(g, len(rows), "tasks;", "nominal budget sum %.0f s" % sum(r[2] for r in rows))
