"""Step 4 table: every reported tree re-costed by cotengra (verify.py output), per network and method,
gap to the best tree known (min over all methods here), medians per family.
usage: python -I analyze.py <verified tsv> <list> [budget]
Method of a tree = file name part after '__' (e.g. ctgref1, gpuB1, hypersa10, treesa_t1_ni25).
TreeSA at budget B = the largest niters whose recorded time is <= B (meta t), else the smallest."""
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tnio  # noqa: E402


def main(vf, lst, outmd=None):
    names = [l.strip() for l in open(lst) if l.strip()]
    fam = {}
    for line in open("/home/itaykadosh/Projects/GATAS/tn-work/tncorpus/manifest.tsv"):
        r = line.split("\t"); fam[r[0]] = r[1]
    C = collections.defaultdict(dict); W = collections.defaultdict(dict); TM = collections.defaultdict(dict)
    for line in open(vf):
        r = line.rstrip("\n").split("\t")
        f, name = r[0], r[1]
        m = f[len(name) + 2:-5]
        C[name][m] = float(r[3]); W[name][m] = float(r[4])
    # TreeSA budget picks need times: read from tree meta
    tdir = "/home/itaykadosh/Projects/GATAS/tn-work/gpucost/results/treesa"
    for name in names:
        for m in list(C[name]):
            if m.startswith("treesa"):
                _, _, _, meta = tnio.read_tree(os.path.join(tdir, f"{name}__{m}.tree"))
                TM[name][m] = float(meta["t"])
    def treesa_at(name, B):
        ms = sorted((m for m in C[name] if m.startswith("treesa")), key=lambda m: TM[name][m])
        if not ms:
            return None
        ok = [m for m in ms if TM[name][m] <= B]
        return ok[-1] if ok else ms[0]
    cols = ["ctgref1", "gpuB1", "hypersa1", "treesa@1", "ctgref10", "gpuB10", "hypersa10", "hyper10", "treesa@10", "treesa200"]
    rows = []
    for name in names:
        c = dict(C[name]); w = dict(W[name])
        for B in (1, 10):
            m = treesa_at(name, B)
            if m:
                c[f"treesa@{B}"] = c[m]; w[f"treesa@{B}"] = w[m]
        if "treesa_t1_ni200" in c:
            c["treesa200"] = c["treesa_t1_ni200"]; w["treesa200"] = w["treesa_t1_ni200"]
        best = min(v for k, v in c.items())
        rows.append((name, fam[name], c, w, best))
    out = []
    out.append("| network | fam | best | " + " | ".join(cols) + " |")
    out.append("|" + "---|" * (3 + len(cols)))
    for name, f, c, w, best in rows:
        out.append(f"| {name[:34]} | {f} | {best:.3f} | " + " | ".join(f"{c[k] - best:.3f}" if k in c else "-" for k in cols) + " |")
    out.append("")
    out.append("Median gap to best known (log10 C), per family:")
    out.append("| family (n) | " + " | ".join(cols) + " |")
    out.append("|" + "---|" * (1 + len(cols)))
    fams = sorted(set(r[1] for r in rows))
    for f in fams + ["all"]:
        rr = [r for r in rows if f == "all" or r[1] == f]
        out.append(f"| {f} ({len(rr)}) | " + " | ".join(
            (f"{statistics.median([r[2][k] - r[4] for r in rr if k in r[2]]):.3f}" if any(k in r[2] for r in rr) else "-") for k in cols) + " |")
    out.append("")
    out.append("Wins/ties/losses of gpuB vs others (|d| < 0.005 = tie):")
    for B, mine, others in ((1, "gpuB1", ["ctgref1", "hypersa1", "treesa@1"]), (10, "gpuB10", ["ctgref10", "hypersa10", "hyper10", "treesa@10", "treesa200"])):
        for o in others:
            w_ = t_ = l_ = 0
            for r in rows:
                if mine in r[2] and o in r[2]:
                    d = r[2][mine] - r[2][o]
                    if d < -0.005: w_ += 1
                    elif d > 0.005: l_ += 1
                    else: t_ += 1
            out.append(f"  {mine} vs {o}: {w_}/{t_}/{l_}")
    out.append("")
    out.append("Per family, gpuB vs the better of hyper-sa and TreeSA at the same budget (W/T/L on C; median dC; median dW of those two trees):")
    for B in (1, 10):
        out.append(f"  B = {B} s:")
        for f in fams + ["all"]:
            rr = [r for r in rows if (f == "all" or r[1] == f) and f"gpuB{B}" in r[2]]
            dc = []; dw = []; w_ = t_ = l_ = 0
            for r in rows:
                if r not in rr:
                    continue
                c, w = r[2], r[3]
                opp = min((k for k in (f"hypersa{B}", f"treesa@{B}") if k in c), key=lambda k: c[k])
                d = c[f"gpuB{B}"] - c[opp]; dc.append(d); dw.append(w[f"gpuB{B}"] - w[opp])
                if d < -0.005: w_ += 1
                elif d > 0.005: l_ += 1
                else: t_ += 1
            if rr:
                out.append(f"    {f:4s} ({len(rr):2d}): {w_}/{t_}/{l_}  median dC {statistics.median(dc):+.3f}  median dW {statistics.median(dw):+.2f}")
    txt = "\n".join(out)
    print(txt)
    if outmd:
        open(outmd, "w").write(txt + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
