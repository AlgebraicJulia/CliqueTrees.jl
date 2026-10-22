#!/usr/bin/env python3
"""Tables for the cuDSS end-to-end benchmark (NOTES.md).
usage: python3 analyze.py results/cudss_b200.tsv [results/ref_b200.tsv ...] > results/tables.md
Each file is compared with the cuDSS-default ("cudss") rows of the same file (same job / node).
Times are medians over the reps. total = order + reorder phase + symbolic + factorization + solve.
Regimes: T(k) = setup + k * (factor + solve), setup = order + reorder + symbolic; k = 1 (one solve), 10, inf.
"""
import sys, math, statistics as st
from collections import defaultdict

COLS = "host gpu name n nnzA cfg rep tord tre tsym tfact tsolve total lu_nnz cflops our_nnz our_flops resid info hostt ttag".split()
BUCKETS = [(1e5, 5e5, "100k-500k"), (5e5, 1e6, "500k-1M"), (1e6, 3e6, "1-3M"), (3e6, 1e12, ">=3M")]


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return math.nan


def load(path):
    rows = defaultdict(list)
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 20:
            continue
        r = dict(zip(COLS, f + [""] * (len(COLS) - len(f))))
        rows[(r["name"], r["cfg"])].append(r)
    agg = {}
    for k, rs in rows.items():
        ok = [r for r in rs if r["info"] == "ok"]
        a = {"n": int(rs[0]["n"]), "status": "ok" if ok else rs[-1]["info"], "reps": len(ok)}
        src = ok if ok else rs
        for c in ("tord", "tre", "tsym", "tfact", "tsolve", "total", "lu_nnz", "cflops", "our_nnz", "our_flops", "resid"):
            v = [fnum(r[c]) for r in src if not math.isnan(fnum(r[c]))]
            a[c] = st.median(v) if v else math.nan
        a["ttag"] = rs[0]["ttag"]
        a["hostt"] = rs[0]["hostt"]
        agg[k] = a
    return agg


def gm(xs):
    xs = [x for x in xs if x and not math.isnan(x) and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else math.nan


def f3(x, d=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def g3(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3g}"


def setup(a):
    return a["tord"] + a["tre"] + a["tsym"]


def per(a):
    return a["tfact"] + a["tsolve"]


def Tk(a, k):
    return setup(a) + k * per(a)


def report(path):
    agg = load(path)
    names = sorted({k[0] for k in agg}, key=lambda nm: agg[(nm, next(c for (m, c) in agg if m == nm))]["n"])
    cfgs = []
    for (_, c) in agg:
        if c not in cfgs:
            cfgs.append(c)
    print(f"## {path}\n")
    # ---------------------------------------------------------------- per matrix, absolute
    print("### Per matrix (medians, seconds; nnz = cuDSS lu_nnz, GF = cuDSS flops / 1e9)\n")
    print("| matrix | n | cfg | order | reorder | symb | factor | solve | total | lu_nnz | GF | resid | status |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm in names:
        for c in cfgs:
            a = agg.get((nm, c))
            if not a:
                continue
            print(f"| {nm} | {a['n']} | {c} | {f3(a['tord'])} | {f3(a['tre'])} | {f3(a['tsym'])} | {f3(a['tfact'])} | {f3(a['tsolve'], 4)} | "
                  f"{f3(a['total'])} | {g3(a['lu_nnz'])} | {g3(a['cflops'] / 1e9)} | {g3(a['resid'])} | {a['status']} |")
    print()
    # ---------------------------------------------------------------- ratios vs cuDSS default
    others = [c for c in cfgs if c not in ("cudss",)]
    print("### Ratios vs cuDSS default (same job): <1 = ours better\n")
    print("| matrix | n | cfg | total (k=1) | T(10) | factor+solve (k=inf) | factor | lu_nnz | flops | order/total | cuDSS reorder/total |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    ratios = defaultdict(lambda: defaultdict(list))
    for nm in names:
        d = agg.get((nm, "cudss"))
        if not d or d["status"] != "ok":
            continue
        for c in others:
            a = agg.get((nm, c))
            if not a or a["status"] != "ok":
                continue
            r = {"t1": Tk(a, 1) / Tk(d, 1), "t10": Tk(a, 10) / Tk(d, 10), "tinf": per(a) / per(d), "fact": a["tfact"] / d["tfact"],
                 "nnz": a["lu_nnz"] / d["lu_nnz"], "fl": a["cflops"] / d["cflops"], "of": a["tord"] / Tk(a, 1), "cf": d["tre"] / Tk(d, 1)}
            for lo, hi, b in BUCKETS:
                if lo <= a["n"] < hi:
                    for k, v in r.items():
                        ratios[(b, c)][k].append(v)
                    ratios[(b, c)]["win1"].append(1.0 if r["t1"] < 1 else 0.0)
            print(f"| {nm} | {a['n']} | {c} | {f3(r['t1'])} | {f3(r['t10'])} | {f3(r['tinf'])} | {f3(r['fact'])} | {f3(r['nnz'])} | {f3(r['fl'])} | "
                  f"{f3(r['of'])} | {f3(r['cf'])} |")
    print()
    print("### Buckets (geomean of ratios vs cuDSS default; wins = #matrices with total < cuDSS)\n")
    print("| bucket | cfg | matrices | total (k=1) | T(10) | k=inf | factor | lu_nnz | flops | order/total (mean) | cuDSS reorder/total (mean) | wins k=1 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lo, hi, b in BUCKETS:
        for c in others:
            r = ratios.get((b, c))
            if not r:
                continue
            m = len(r["t1"])
            print(f"| {b} | {c} | {m} | {f3(gm(r['t1']))} | {f3(gm(r['t10']))} | {f3(gm(r['tinf']))} | {f3(gm(r['fact']))} | {f3(gm(r['nnz']))} | "
                  f"{f3(gm(r['fl']))} | {f3(st.mean(r['of']))} | {f3(st.mean(r['cf']))} | {int(sum(r['win1']))}/{m} |")
    print()
    # ---------------------------------------------------------------- break-even k
    print("### Regimes: numbers k of factorizations (same pattern, one ordering + analysis) for which the config's T(k) beats cuDSS default\n")
    print("| matrix | " + " | ".join(others) + " |")
    print("|---|" + "---|" * len(others))
    for nm in names:
        d = agg.get((nm, "cudss"))
        if not d or d["status"] != "ok":
            continue
        cells = []
        for c in others:
            a = agg.get((nm, c))
            if not a or a["status"] != "ok":
                cells.append("-"); continue
            ds, dp, s_, p_ = setup(d), per(d), setup(a), per(a)
            win1 = Tk(a, 1) < Tk(d, 1)
            if win1 and p_ <= dp:
                cells.append("all k")
            elif win1:
                cells.append(f"k<={math.floor((ds - s_) / (p_ - dp))}")
            elif p_ < dp:
                cells.append(f"k>={math.ceil((s_ - ds) / (dp - p_))}")
            else:
                cells.append("never")
        print(f"| {nm} | " + " | ".join(cells) + " |")
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
