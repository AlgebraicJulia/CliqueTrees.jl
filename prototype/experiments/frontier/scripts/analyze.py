"""Tables from the frontier result TSVs.
usage: python analyze.py <res dir> [<res dir> ...] > tables.md
All costs log10 C (C = cotengra contraction_cost = scalar multiply-adds, complex for circuits;
real FLOPs ~ 8C, i.e. +0.90 in log10). W = log2 largest intermediate."""
import csv
import glob
import os
import sys
from collections import defaultdict

COLS = ["name", "method", "budget", "seed", "tsearch", "ntrial", "bag", "stage", "cost", "width", "maxop", "nsliced", "post_s"]

# published bars, log10 C (complex multiply-adds); see litreview/frontier_costs.md
LIT = {
    "syc53_m12_amp": [("GK21 unsliced [derived]", 13.38), ("GK21 W=27", 14.26), ("Kalachev21 16GB", 13.26)],
    "syc53_m14_amp": [("GK21 unsliced [derived]", 14.00), ("GK21 W=27", 15.14), ("Kalachev21 16GB", 14.00)],
    "syc53_m16_amp": [("GK21 unsliced [derived]", 17.27), ("GK21 W=27", 18.38), ("Kalachev21 16GB", 16.95)],
    "syc53_m18_amp": [("GK21 unsliced [derived]", 17.79), ("GK21 W=27", 19.42)],
    "syc53_m20_amp": [("GK21 unsliced [derived]", 18.68), ("GK21 W=27", 22.49), ("Morvan24 unconstrained (5e18 real/8)", 17.80)],
    "syc53_m12_marg6": [("Huang20 batch-64 W<=29", 13.04)],
    "syc53_m14_marg6": [("Huang20 batch-64 W<=29", 13.85)],
    "syc53_m16_marg6": [("Huang20 batch-64 W<=29", 17.06)],
    "syc53_m18_marg6": [("Huang20 batch-64 W<=29", 17.41)],
    "syc53_m20_marg6": [("Huang20 batch-64 W<=29", 18.82), ("ctg notebook repro W=29", 18.73)],
    "syc53_m20_open6huang": [("Huang20 batch-64 W<=29", 18.82)],
    "syc53_m20_marg21": [("Pan-Zhang22 2^21 W=30", 18.65), ("ctg notebook repro W=30", 18.66)],
    "syc70_m24_amp": [("Morvan24 unconstrained (4e24 real/8)", 23.70)],
    "syc67_m32_amp": [("Morvan24 unconstrained (1e24 real/8) [different TN, see caveat]", 23.10)],
}
OURS = ("bw", "nd", "hbt", "hbtnd")


def read(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.tsv")):
            for r in csv.reader(open(f), delimiter="\t"):
                if len(r) == len(COLS):
                    rows.append(dict(zip(COLS, r)))
    return rows


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main(dirs):
    rows = read(dirs)
    by = defaultdict(dict)   # (name, method, budget) -> stage -> row
    for r in rows:
        by[(r["name"], r["method"], float(r["budget"]))][r["stage"]] = r
    names = sorted({k[0] for k in by})
    for name in names:
        keys = sorted([k for k in by if k[0] == name], key=lambda k: (k[1].startswith("hslice"), k[1] not in OURS, k[1], k[2]))
        targets = sorted({int(s[10:]) for k in keys for s in by[k] if s.startswith("best_slice")})
        print(f"\n### {name}\n")
        for lab, v in LIT.get(name, []):
            print(f"- published: {lab}: **{v:.2f}**")
        print()
        hdr = ["method", "budget s", "search s", "bag / maxop", "raw C", "raw W", "refined C", "refined W"] + [f"sliced W={t}" for t in targets]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for k in keys:
            st = by[k]
            raw = st.get("raw"); ref = st.get("refined")
            if raw is None:
                continue
            bag = raw["bag"] if raw["bag"] else raw["maxop"]
            if k[1].startswith("hslice"):
                t = int(k[1][6:])
                cells = [fmt(float(raw["cost"])) + f" (W={float(raw['width']):.0f})" if t == tt else "" for tt in targets]
                line = [k[1], f"{k[2]:g}", fmt(float(raw["tsearch"]), 1), f"{float(raw['maxop']):.0f}", "", "", "", ""] + cells
            else:
                cells = [fmt(float(st[f"best_slice{t}"]["cost"])) if f"best_slice{t}" in st else "" for t in targets]
                line = [k[1], f"{k[2]:g}", fmt(float(raw["tsearch"]), 1), f"{float(bag):.0f}",
                        fmt(float(raw["cost"])), f"{float(raw['width']):.0f}",
                        fmt(float(ref["cost"])) if ref else "", f"{float(ref['width']):.0f}" if ref else ""] + cells
            print("| " + " | ".join(line) + " |")


if __name__ == "__main__" and not os.environ.get("CUT") and not os.environ.get("ANYTIME"):
    main([d for d in sys.argv[1:]])


def cut_tables(dirs):
    """Fixed-slicing experiment: totals = per-slice cost + |S| log10 2."""
    import math
    import re
    rows = read(dirs)
    by = defaultdict(dict)
    for r in rows:
        if "_cut" in r["name"]:
            by[(r["name"], r["method"], float(r["budget"]))][r["stage"]] = r
    for name in sorted({k[0] for k in by}):
        nS = int(re.search(r"_cut(\d+)_", name).group(1))
        add = nS * math.log10(2)
        W = re.search(r"hslice(\d+)", name).group(1)
        print(f"\n### {name}  (|S| = {nS} sliced indices, target W = {W}; totals include 2^{nS} slices)\n")
        print("| ordering of the cut network | budget s | search s | bag width | per-slice W (raw) | total C raw | total C refined (W) | total C at W<=%s |" % W)
        print("|---|---|---|---|---|---|---|---|")
        keys = sorted([k for k in by if k[0] == name], key=lambda k: (k[1] != "ctgslice", k[1], k[2]))
        for k in keys:
            st = by[k]; raw = st["raw"]; ref = st.get("refined"); bs = st.get(f"best_slice{W}")
            lab = "cotengra's own per-slice tree (via our bridge)" if k[1] == "ctgslice" else k[1]
            print(f"| {lab} | {k[2]:g} | {float(raw['tsearch']):.1f} | {float(raw['bag']):.0f} | {float(raw['width']):.0f} | "
                  f"{float(raw['cost']) + add:.2f} | {float(ref['cost']) + add:.2f} ({float(ref['width']):.0f}) | "
                  f"{(float(bs['cost']) + add) if bs else float('nan'):.2f} |")


if __name__ == "__main__" and os.environ.get("CUT"):
    cut_tables(sys.argv[1:])


def anytime(dirs):
    """Does HBT keep improving with budget? bag width and refined C per budget."""
    rows = read(dirs)
    by = defaultdict(dict)
    for r in rows:
        if r["method"] in ("hbt", "hbtnd", "hyper", "rgreedy") and "_cut" not in r["name"]:
            by[(r["name"], r["method"])][float(r["budget"])] = r if r["stage"] == "raw" else by[(r["name"], r["method"])].get(float(r["budget"]))
    ref = {}
    for r in rows:
        if r["stage"] == "refined":
            ref[(r["name"], r["method"], float(r["budget"]))] = float(r["cost"])
    budgets = sorted({b for v in by.values() for b in v})
    print("| instance | method | " + " | ".join(f"T={b:g}" for b in budgets) + " |")
    print("|---|---|" + "---|" * len(budgets))
    for (n, m) in sorted(by):
        cells = []
        for b in budgets:
            r = by[(n, m)].get(b)
            if r is None:
                cells.append("")
            else:
                w = r["bag"] if r["bag"] else r["maxop"]
                c = ref.get((n, m, b))
                cells.append(f"{float(w):.0f} / {c:.2f}" if c is not None else f"{float(w):.0f}")
        print(f"| {n} | {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__" and os.environ.get("ANYTIME"):
    anytime(sys.argv[1:])
