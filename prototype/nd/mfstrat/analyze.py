import sys, collections, math, statistics as st
rows = collections.defaultdict(dict)  # graph -> alg -> (nnz, flops, tw, t, n)
host = set()
for l in open(sys.argv[1]):
    p = l.rstrip("\n").split("\t")
    if len(p) < 10 or not p[6]: continue
    host.add(p[0]); rows[p[2]][p[5]] = (float(p[6]), float(p[7]), int(p[8]), float(p[9]), int(p[3]))
nd = {}
for l in open("nd/r2_t32_b200.tsv"):
    p = l.split("\t")
    if p[5] == "gndd:hyb=1;rmax=4" and p[6] == "256": nd[p[2]] = (float(p[7]), float(p[8]), int(p[9]), float(p[10]))
def fam(g):
    if g.startswith("lat"): return "lattice"
    if g.startswith("corpus:"): return "TN"
    if g in ("G2_circuit", "G3_circuit"): return "circuit"
    if g in ("thermal1","thermal2","parabolic_fem","jnlbrng1","ecology2","Dubcova3","thermomech_TC","thermomech_dM"): return "SS-2D"
    return "SS-3D/struct"
def bucket(n): return "n<50k" if n < 5e4 else "50k-500k" if n < 5e5 else "500k-2M" if n < 2e6 else "n>=2M"
def gm(x): return math.exp(sum(math.log(v) for v in x)/len(x)) if x else float("nan")
graphs = sorted(rows, key=lambda g: rows[g].get("MF1", (0,0,0,0,0))[4])
ALGS = ["MF0","MF1","MF2","MF3","MF4","AMD","AMF","METIS"]
out = []
P = out.append
P(f"Hosts: {sorted(host)}\n")
P("## Raw per graph (nnz(L), flops, width, time_s)\n")
P("| graph | n | fam | " + " | ".join(ALGS) + " | GPU-ND |")
P("|---|---|---|" + "---|"*(len(ALGS)+1))
for g in graphs:
    r = rows[g]; n = r.get("MF1", (0,)*5)[4]
    cells = [f"{r[a][0]:.4g} / {r[a][1]:.3g} / {r[a][2]} / {r[a][3]:.3g}" if a in r else "-" for a in ALGS]
    d = nd.get(g); cells.append(f"{d[0]:.4g} / {d[1]:.3g} / {d[2]}" if d else "-")
    P(f"| {g} | {n} | {fam(g)} | " + " | ".join(cells) + " |")
def ratio(g, a, b, k):
    r = rows[g]
    if a in r and b in r and r[b][k] > 0: return r[a][k]/r[b][k]
def wtl(xs, tol=0.005):
    return (sum(x < 1-tol for x in xs), sum(abs(x-1) <= tol for x in xs), sum(x > 1+tol for x in xs))
def table(title, key, groups, ref, algs, k, label):
    P(f"\n### {title}\n")
    P("Geomean of ratio (strategy / " + ref + ") for " + label + "; W/T/L = strategy better/equal(+-0.5%)/worse on that metric.\n")
    P("| group | #graphs | " + " | ".join(algs) + " |")
    P("|---|---|" + "---|"*len(algs))
    gs = collections.OrderedDict()
    for g in graphs: gs.setdefault("ALL", []).append(g); gs.setdefault(key(g), []).append(g)
    for name, L in sorted(gs.items(), key=lambda kv: (kv[0] != "ALL", kv[0])):
        cells = []
        for a in algs:
            xs = [x for x in (ratio(g, a, ref, k) for g in L) if x]
            w = wtl(xs) if xs else (0,0,0)
            cells.append(f"{gm(xs):.3f} ({w[0]}/{w[1]}/{w[2]})" if xs else "-")
        P(f"| {name} | {len(L)} | " + " | ".join(cells) + " |")
for ref in ["MF1","AMD","METIS"]:
    algs = [a for a in ["MF0","MF2","MF3","MF4"] if a != ref] + (["MF1"] if ref != "MF1" else [])
    for k, lab in ((0,"nnz(L)"),(1,"flops"),(2,"width")):
        if k == 2:
            # width ratio
            pass
        table(f"{lab} vs {ref}, by family", fam, None, ref, algs, k, lab)
        table(f"{lab} vs {ref}, by size bucket", lambda g: bucket(rows[g]["MF1"][4]), None, ref, algs, k, lab)
table("time vs MF1, by family", fam, None, "MF1", ["MF0","MF2","MF3","MF4","AMD","AMF","METIS"], 3, "time")
table("time vs MF1, by size bucket", lambda g: bucket(rows[g]["MF1"][4]), None, "MF1", ["MF0","MF2","MF3","MF4","AMD","AMF","METIS"], 3, "time")
# GPU ND comparison
P("\n### MF strategies vs GPU-ND hybrid (gndd:256:hyb=1;rmax=4, B200 r2_t32) on overlapping graphs\n")
P("Geomean ratio strategy/GPU-ND (<1 means CPU MF has less fill/flops).\n")
P("| metric | #graphs | " + " | ".join(ALGS) + " |"); P("|---|---|" + "---|"*len(ALGS))
ov = [g for g in graphs if g in nd]
for k, lab in ((0,"nnz(L)"),(1,"flops"),(2,"width")):
    cells = []
    for a in ALGS:
        xs = [rows[g][a][k]/nd[g][k] for g in ov if a in rows[g]]
        cells.append(f"{gm(xs):.3f}" if xs else "-")
    P(f"| {lab} | {len(ov)} | " + " | ".join(cells) + " |")
P("\nPer graph nnz ratio MF2 / GPU-ND, MF1 / GPU-ND:\n")
P("| graph | MF2/ND nnz | MF1/ND nnz | MF2/ND flops | MF1/ND flops |"); P("|---|---|---|---|---|")
for g in ov:
    r = rows[g]
    if "MF2" in r and "MF1" in r:
        P(f"| {g} | {r['MF2'][0]/nd[g][0]:.3f} | {r['MF1'][0]/nd[g][0]:.3f} | {r['MF2'][1]/nd[g][1]:.3f} | {r['MF1'][1]/nd[g][1]:.3f} |")
P("\n## Per graph ratios vs MF1 (nnz, flops, width, time) for MF0/MF2/MF3/MF4\n")
P("| graph | " + " | ".join(f"{a} nnz|flops|tw|time" for a in ["MF0","MF2","MF3","MF4"]) + " |"); P("|---|---|---|---|---|")
for g in graphs:
    c = []
    for a in ["MF0","MF2","MF3","MF4"]:
        c.append("|".join(f"{ratio(g,a,'MF1',k):.3f}" if ratio(g,a,'MF1',k) else "-" for k in range(4)))
    P(f"| {g} | " + " | ".join(c) + " |")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
