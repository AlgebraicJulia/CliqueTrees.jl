"""Reconfiguration-only ablation: same seed ordering (hbt 1 s); cotengra ftn.refine (subtree size 8,
<= 8 passes) vs GPU batched DP (K = 8, 10, 12, to convergence). Costs from verify.py, times from metadata.
usage: python -I ablate_table.py <verified tsv> <list>"""
import statistics, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tnio
V = {}
for l in open(sys.argv[1]):
    r = l.split("\t"); V[r[0]] = float(r[3])
names = [l.strip() for l in open(sys.argv[2]) if l.strip()]
G = "/home/itaykadosh/Projects/GATAS/tn-work/gpucost/"
rows = []
for n in names:
    tord = float(open(G + f"data/ord/{n}__hbtT1.0s1.ord").readline().split("\t")[2])
    _, _, _, m = tnio.read_tree(G + f"results/base/{n}__ctgref1.tree")
    tc = float(m["t"]) - tord
    r = dict(name=n, ctg=V[f"{n}__ctgref1.tree"], tctg=tc)
    for K in (8, 10, 12):
        _, _, _, m = tnio.read_tree(G + f"results/ablate/{n}__gpurK{K}.tree")
        r[f"g{K}"] = V[f"{n}__gpurK{K}.tree"]; r[f"t{K}"] = float(m["t"])
    rows.append(r)
print(f"{'network':40s} {'ctg C':>8s} {'s':>6s} | {'K8':>8s} {'s':>5s} | {'K10':>8s} {'s':>5s} | {'K12':>8s} {'s':>5s}")
for r in rows:
    print(f"{r['name'][:40]:40s} {r['ctg']:8.3f} {r['tctg']:6.2f} | {r['g8']:8.3f} {r['t8']:5.2f} | {r['g10']:8.3f} {r['t10']:5.2f} | {r['g12']:8.3f} {r['t12']:5.2f}")
for K in (8, 10, 12):
    d = [r[f"g{K}"] - r["ctg"] for r in rows]
    sp = [r["tctg"] / r[f"t{K}"] for r in rows]
    print(f"K={K}: GPU minus cotengra log10C median {statistics.median(d):+.3f} (better {sum(x < -0.005 for x in d)}, tie {sum(abs(x) <= 0.005 for x in d)}, worse {sum(x > 0.005 for x in d)}); "
          f"speedup median {statistics.median(sp):.1f}x (min {min(sp):.1f}, max {max(sp):.1f})")
