#!/usr/bin/env python3 -I
"""usage: report.py RESULTSDIR  -> markdown (tables + summary) on stdout"""
import os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RD = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results"
SOLV = [("tamaki17", "Tamaki17"), ("septw", "septw"), ("tamaki16", "Tamaki16"), ("uplow22", "UpLow22")]

def hdr(path, tag):
    try:
        for l in open(path):
            if l.startswith("p "): p = l.split(); return int(p[2]), int(p[3])
            if l.startswith(tag): return int(l.split()[3]) - 1
    except OSError: return None
def load(s, st):
    p = f"{RD}/{s}_{st}.tsv"
    if not os.path.exists(p): return None
    return {r["inst"]: r for r in csv.DictReader(open(p), delimiter="\t")}
def cell(r):
    if r is None: return "-"
    if r["status"] == "OK": return f"{float(r['secs']):.1f}s" + ("" if r["valid"] == "y" else " (INVALID)")
    if r["status"] == "TIMEOUT": return "TO" + (f" (ub {r['width']})" if r["width"] != "-" else "")
    return "FAIL"
out = []
for st, tdd in (("exact", "exact_td"), ("bonus", "bonus_td")):
    R = {s: load(s, st) for s, _ in SOLV}; R = {s: v for s, v in R.items() if v}
    if not R: continue
    names = sorted(next(iter(R.values())))
    ok = lambda s, n: R[s].get(n, {}).get("status") == "OK"
    out.append(f"\n## Set `{st}` ({len(names)} instances)\n")
    out.append("| instance | n | m | tw (known) | " + " | ".join(f"{l} time" for s, l in SOLV if s in R) + " | septw width ok? |\n|" + "---|" * (4 + len(R) + 1))
    for n in names:
        nm = hdr(f"{ROOT}/data/plain/{st}/{n}.gr", "p") or ("?", "?")
        tw = hdr(f"{ROOT}/data/plain/{tdd}/{n}.td", "s td"); tw = "?" if tw is None else tw
        r = R.get("septw", {}).get(n)
        wok = "" if not r or r["status"] != "OK" else ("y" if tw == "?" or str(tw) == r["width"] else f"NO ({r['width']})")
        out.append(f"| {n} | {nm[0]} | {nm[1]} | {tw} | " + " | ".join(cell(R[s].get(n)) for s, _ in SOLV if s in R) + f" | {wok} |")
    out.append(f"\n### Summary, `{st}` (solved = optimal TD found and validated)\n")
    for s, l in SOLV:
        if s in R: out.append(f"- {l}: {sum(ok(s, n) and R[s][n]['valid'] == 'y' for n in names)} / {len(names)} solved")
    if "tamaki17" in R and "septw" in R:
        A = [n for n in names if ok("tamaki17", n)]; B = [n for n in names if ok("septw", n)]
        out.append(f"- Neither Tamaki17 nor septw solves ({len(names)-len(set(A)|set(B))}): " + ", ".join(n for n in names if n not in A and n not in B))
        out.append(f"- Only Tamaki17 ({len(set(A)-set(B))}): " + ", ".join(n for n in names if n in A and n not in B))
        out.append(f"- Only septw ({len(set(B)-set(A))}): " + ", ".join(n for n in names if n in B and n not in A))
print("\n".join(out))
