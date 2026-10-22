"""Step 1 analysis: is lcost a faithful proxy for cotengra's C of the induced tree?
usage: python -I corr_analyze.py <corr dir> <list>  -> prints a table and writes <corr dir>/summary.tsv
Per network (all orderings / variants only): Spearman rho(lcost, ctgC), rho(gpuC, ctgC), and the top-1
regret in log10 C: ctgC of the ordering ranked best by the proxy minus the best ctgC in the pool."""
import csv
import math
import os
import sys


def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    c = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra)); vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return c / (va * vb) if va > 0 and vb > 0 else float("nan")


def main(d, lst):
    L = math.log10(2)
    rows = []
    for name in [l.strip() for l in open(lst) if l.strip()]:
        a = list(csv.DictReader(open(os.path.join(d, name + ".tsv")), delimiter="\t"))
        b = list(csv.DictReader(open(os.path.join(d, name + ".ctg.tsv")), delimiter="\t"))
        assert len(a) == len(b)
        lc = [float(r["lcost"]) for r in a]; gc = [float(r["gpu_log2C"]) for r in a]; cc = [float(r["ctg_log2C"]) for r in b]
        var = [i for i, r in enumerate(a) if r["tag"].startswith("var_")]
        def regret(proxy, idx):
            k = min(idx, key=lambda i: proxy[i])
            return (cc[k] - min(cc[i] for i in idx)) * L
        allidx = list(range(len(a)))
        same = sum(abs(g - c) < 1e-3 for g, c in zip(gc, cc))
        top = sorted(allidx, key=lambda i: cc[i])[: len(a) // 4]
        row = dict(name=name, n=len(a),
                   rho_l=spearman(lc, cc), rho_g=spearman(gc, cc),
                   rho_l_var=spearman([lc[i] for i in var], [cc[i] for i in var]),
                   rho_g_var=spearman([gc[i] for i in var], [cc[i] for i in var]),
                   rho_l_top=spearman([lc[i] for i in top], [cc[i] for i in top]),
                   rho_g_top=spearman([gc[i] for i in top], [cc[i] for i in top]),
                   reg_l=regret(lc, allidx), reg_g=regret(gc, allidx), same=same,
                   spread=(max(cc) - min(cc)) * L)
        rows.append(row)
    keys = list(rows[0].keys())
    with open(os.path.join(d, "summary.tsv"), "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(f"{'network':48s} {'n':>3s} {'rho(lcost)':>10s} {'rho(gpuC)':>9s} {'var:l':>6s} {'var:g':>6s} {'top:l':>6s} {'top:g':>6s} {'regret l':>8s} {'regret g':>8s} {'gpu=ctg':>7s} {'spread':>6s}")
    for r in rows:
        print(f"{r['name']:48s} {r['n']:3d} {r['rho_l']:10.3f} {r['rho_g']:9.3f} {r['rho_l_var']:6.3f} {r['rho_g_var']:6.3f} {r['rho_l_top']:6.3f} {r['rho_g_top']:6.3f} {r['reg_l']:8.3f} {r['reg_g']:8.3f} {r['same']:7d} {r['spread']:6.2f}")
    med = lambda k: sorted(r[k] for r in rows if r[k] == r[k])[len([r for r in rows if r[k] == r[k]]) // 2]
    print(f"{'MEDIAN':48s} {'':3s} {med('rho_l'):10.3f} {med('rho_g'):9.3f} {med('rho_l_var'):6.3f} {med('rho_g_var'):6.3f} {med('rho_l_top'):6.3f} {med('rho_g_top'):6.3f} {med('reg_l'):8.3f} {med('reg_g'):8.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
