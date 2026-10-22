"""B_FAIR tables from gpu_cutn_fair.tsv (cuTensorNet executor).  python analyzeBfair.py <tsv> <out.md>"""
import csv, math, statistics as st, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
R = defaultdict(dict)
for r in rows:
    R[r["net"]][r["tag"]] = r
f = lambda x: float(x) if x not in ("", "nan", None) else float("nan")
ok = lambda r: r is not None and r["status"] == "ok"
out = []
P = out.append
st_count = defaultdict(int)
for r in rows:
    st_count[r["status"].split(":")[0]] += 1
P("## Status counts\n\n" + ", ".join(f"{k}: {v}" for k, v in sorted(st_count.items())) + "\n")
bad = [r for r in rows if r["status"] not in ("ok", "skip_cost")]
if bad:
    P("Non-ok rows:\n\n| net | tag | status |\n|---|---|---|")
    for r in bad:
        P(f"| {r['net']} | {r['tag']} | {r['status'][:120]} |")
    P("")
errs = [f(r["relerr"]) for r in rows if r["status"] == "ok" and r["tag"] != "cutn_own"]
if errs:
    P(f"relerr vs complex128 reference (ok rows): median {st.median(errs):.2e}, max {max(errs):.2e}, >1e-3: {sum(e > 1e-3 for e in errs)}\n")

def cmp_table(ot, ct, title):
    P(f"## {title}: ours ({ot}) vs cotengra ({ct}), cuTensorNet execution\n")
    P("| net | ours logC | ctg logC | ours W | ctg W | ours #sl | ctg #sl | ours t (s) | ctg t (s) | speedup | ours search | ctg search | TTS speedup | class |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cats = defaultdict(list)
    for net in sorted(R):
        o, c = R[net].get(ot), R[net].get(ct)
        if not (ok(o) and ok(c)):
            continue
        no, nc = int(float(o["nslices"])), int(float(c["nslices"]))
        to, tc = f(o["t_med"]), f(c["t_med"])
        so, sc = f(o["search_s"]), f(c["search_s"])
        sp, tts = tc / to, (sc + tc) / (so + to)
        cls = "a:both unsliced" if no == 1 and nc == 1 else ("b:ctg sliced, ours fewer/none" if nc > 1 and no < nc else "c:other")
        cats[cls].append((net, sp, tts))
        P(f"| {net} | {f(o['log10C_pred']):.2f} | {f(c['log10C_pred']):.2f} | {f(o['W']):.1f} | {f(c['W']):.1f} | {no} | {nc} | {to:.4g} | {tc:.4g} | {sp:.2f} | {so:.1f} | {sc:.1f} | {tts:.2f} | {cls[:1]} |")
    P("")
    P(f"Summary ({ot} vs {ct}; speedup = ctg time / ours time; win >1.05, loss <0.95, else tie):\n")
    P("| class | n | wins | ties | losses | median speedup | geomean speedup | median TTS speedup | geomean TTS |")
    P("|---|---|---|---|---|---|---|---|---|")
    allr = [x for v in cats.values() for x in v]
    for name, v in [(k, cats[k]) for k in sorted(cats)] + [("all", allr)]:
        if not v:
            continue
        s = [x[1] for x in v]; t = [x[2] for x in v]
        gm = lambda a: math.exp(sum(map(math.log, a)) / len(a))
        P(f"| {name} | {len(v)} | {sum(x > 1.05 for x in s)} | {sum(0.95 <= x <= 1.05 for x in s)} | {sum(x < 0.95 for x in s)} | {st.median(s):.2f} | {gm(s):.2f} | {st.median(t):.2f} | {gm(t):.2f} |")
    P("")

cmp_table("ours60", "ctg_hyper60", "60 s budget")
cmp_table("ours10", "ctg_hyper10", "10 s budget")

P("## Slices at the memory cap (2^30 elements per intermediate), all networks with trees\n")
P("| net | ours10 #sl | ours60 #sl | ctg_hyper10 #sl | ctg_hyper60 #sl | ctg_greedy #sl | ours60 logC | ctg60 logC |")
P("|---|---|---|---|---|---|---|---|")
sl = defaultdict(list)
for net in sorted(R):
    g = lambda t, k: R[net][t][k] if t in R[net] else "nan"
    ns = lambda t: int(float(g(t, "nslices"))) if g(t, "nslices") != "nan" else None
    P(f"| {net} | {ns('ours10')} | {ns('ours60')} | {ns('ctg_hyper10')} | {ns('ctg_hyper60')} | {ns('ctg_greedy')} | {g('ours60','log10C_pred')} | {g('ctg_hyper60','log10C_pred')} |")
    if ns("ours60") and ns("ctg_hyper60"):
        sl["60"].append((ns("ours60"), ns("ctg_hyper60"))); 
    if ns("ours10") and ns("ctg_hyper10"):
        sl["10"].append((ns("ours10"), ns("ctg_hyper10")))
P("")
for k, v in sl.items():
    P(f"{k} s: ours fewer slices in {sum(a < b for a, b in v)}, equal in {sum(a == b for a, b in v)}, more in {sum(a > b for a, b in v)} of {len(v)}; "
      f"unsliced (1): ours {sum(a == 1 for a, b in v)}, cotengra {sum(b == 1 for a, b in v)}; median log10(ctg/ours slices) = {st.median([math.log10(b / a) for a, b in v]):.2f}\n")

P("## Executed trees: real time of all tags (s), cuTensorNet\n")
tags = ["ours10", "ours60", "td_MF1", "td_hbt1", "td_exact", "td_hyb300", "ctg_greedy", "ctg_hyper10", "ctg_hyper60", "cutn_own"]
P("| net | " + " | ".join(tags) + " |\n|---|" + "---|" * len(tags))
for net in sorted(R):
    c = []
    for t in tags:
        r = R[net].get(t)
        c.append("" if r is None else (f"{f(r['t_med']):.4g}" if ok(r) else r["status"].split(":")[0]))
    P(f"| {net} | " + " | ".join(c) + " |")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
