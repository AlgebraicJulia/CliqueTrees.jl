"""Cost-aware selection: tables.

    python analyze.py <ref cost dir (exactness resA/cost)> <ref td dir (resA/td)> <pool dir> <gpu dir> <rc dir>
                      <B trees.tsv> <out.md> [unused] [res_contract.tsv]

All costs log10 C (cotengra contraction_cost), stage br+ctg unless said otherwise (= the study's best TD
conversion: branchtree + branchrefine + cotengra subtree_reconfigure). Study numbers come from the exactness
experiment A (same networks, same conversion); "opt" = TDs of certified optimal width.
"""
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np

L2 = math.log10(2.0)
EPS = 1e-3


def family(name):
    for p in ("qaoazz", "qaoaamp", "reg", "rqc", "syc", "lat", "mc3sat", "lm_mera"):
        if name.startswith(p):
            return {"lm_mera": "lm"}.get(p, p)
    return name.split("_")[0]


def read_ref(costdir, tddir):
    ref = {}
    for f in glob.glob(os.path.join(costdir, "*.tsv")):
        net = os.path.basename(f)[:-4]
        rows = defaultdict(dict)
        for l in open(f):
            x = l.rstrip("\n").split("\t")
            if l.startswith(("#", "net\t")) or len(x) < 8:
                continue
            rows[x[1]][x[2]] = (float(x[3]), float(x[7]) if x[7] else float("nan"))
        ex = None; tex = float("nan"); brc = {}
        tdf = os.path.join(tddir, net + ".td")
        if os.path.exists(tdf):
            for l in open(tdf):
                if l.startswith("EXACT"):
                    y = l.split("\t"); ex = (float(y[1]), y[3] == "1"); tex = float(y[4])
                elif l.startswith("ORDER"):
                    y = l.split("\t"); brc[y[1]] = (float(y[3]), float(y[8]) * L2, float(y[2]))   # width, br log10C, seconds
        ref[net] = dict(rows=rows, exact=ex, texact=tex, brc=brc)
    return ref


def read_pool(f):
    d = dict(td={}, sel=[], net=None, atoms=[], enum=[], nouni=False)
    ls = open(f).read().split("\n")
    i = 0
    while i < len(ls):
        x = ls[i].split("\t")
        if x[0] == "NET":
            d["net"] = dict(name=x[1], n=int(x[2]), K=float(x[3]), cert=x[4] == "1", natoms=int(x[5]), nmax=int(x[6]),
                            tfront=float(x[7]), tpool=float(x[8]))
        elif x[0] == "ATOM":
            d["atoms"].append(x)
        elif x[0] == "TD":
            d["td"][x[1]] = dict(mode=x[2], delta=int(x[3]), j=int(x[4]), t=float(x[5]), width=float(x[6]),
                                 lcost=float(x[7]), br=float(x[8]) * L2, bk=float(x[9]) * L2)
            i += 1
        elif x[0] == "SEL":
            d["sel"].append(dict(mode=x[1], delta=int(x[2]), N=int(x[3]), t=float(x[4]), tag_br=x[5], br=float(x[6]) * L2,
                                 tag_bk=x[7], bk=float(x[8]) * L2))
        elif x[0] == "#NOUNI":
            d["nouni"] = True
        elif x[0] == "#ENUM":
            d["enum"].append(x)
        elif x[0] == "SSA":
            i += 1
        i += 1
    return d


def read_gpu(f):
    d = dict(gs=None, gms=None, gmt=[], pol={}, T={})
    if not os.path.exists(f):
        return d
    ls = open(f).read().split("\n")
    for k, l in enumerate(ls):
        x = l.split("\t")
        if x[0] == "GS":
            d["gs"] = dict(n=int(x[1]), secs=float(x[2]), rate=float(x[3]), dcpu=float(x[4]), dw=float(x[5]))
        elif x[0] == "T":
            d["T"][x[1]] = float(x[2]) * L2
        elif x[0] == "GMS":
            d["gms"] = dict(cap=float(x[1]), ncand=int(x[2]), secs=float(x[4]), seed=float(x[5]) * L2, best=float(x[6]) * L2)
        elif x[0] == "GMT":
            d["gmt"] = [tuple(float(z) for z in t.split(":")) for t in x[1].split()]
        elif x[0] == "POL":
            d["pol"][x[1]] = dict(src=x[2], a=float(x[3]) * L2, b=float(x[4]) * L2, secs=float(x[5]))
    return d


def read_rc(f):
    d = defaultdict(dict)
    if not os.path.exists(f):
        return d
    for l in open(f):
        x = l.rstrip("\n").split("\t")
        if l.startswith("#") or len(x) < 7:
            continue
        d[x[1]][x[2]] = float(x[3])
    return d


def read_B(f):
    d = defaultdict(dict)
    if os.path.exists(f):
        for l in open(f):
            x = l.rstrip("\n").split("\t")
            if len(x) > 3:
                d[x[0]][x[1]] = float(x[2])
    return d


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else float("nan")


def wtl(d, tol=0.01):
    d = [x for x in d if x is not None and not math.isnan(x)]
    return f"{sum(x < -tol for x in d)}/{sum(abs(x) <= tol for x in d)}/{sum(x > tol for x in d)}"


def main():
    costdir, tddir, pooldir, gpudir, rcdir, btrees, outp = sys.argv[1:8]
    ptsp = sys.argv[8] if len(sys.argv) > 8 else None
    ref = read_ref(costdir, tddir)
    B = read_B(btrees)
    Bnets = set(B)
    out = []
    P = print

    def pr(s=""):
        out.append(s)

    nets = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(pooldir, "*.pool")))
    R = []   # per-network records
    for net in nets:
        pool = read_pool(os.path.join(pooldir, net + ".pool"))
        if pool["net"] is None:
            continue
        gpu = read_gpu(os.path.join(gpudir, net + ".gpu"))
        rc = read_rc(os.path.join(rcdir, net + ".tsv"))
        rf = ref.get(net)
        r = dict(net=net, fam=family(net), K=pool["net"]["K"], cert=pool["net"]["cert"], tpool=pool["net"]["tpool"],
                 natoms=pool["net"]["natoms"], nmax=pool["net"]["nmax"], ntd=len(pool["td"]), pool=pool, gpu=gpu, rc=rc)
        # study reference
        if rf:
            rows = rf["rows"]; K = r["K"]
            for st in ("bucket", "br", "br+ctg"):
                heur = [v[st][0] for t, v in rows.items() if st in v and not t.startswith(("exact", "hyb", "ctg_"))]
                opt = [v[st][0] for t, v in rows.items() if st in v and not t.startswith("ctg_") and abs(v[st][1] - K) < 1e-3]
                allt = [v[st][0] for t, v in rows.items() if st in v and not t.startswith("ctg_")]
                exs = [v[st][0] for t, v in rows.items() if st in v and t.startswith("exact")]
                ex0 = rows.get("exact", rows.get("hyb300", {})).get(st, (float("nan"),))[0]
                r[f"ref_{st}"] = dict(exact=ex0, exactbest=min(exs) if exs else float("nan"),
                                       heur=min(heur) if heur else float("nan"),
                                       optbest=min(opt) if opt else float("nan"), optworst=max(opt) if opt else float("nan"),
                                       nopt=len(opt), best=min(allt) if allt else float("nan"))
            r["hyper10"] = rows.get("ctg_hyper10", {}).get("+ctg", (float("nan"),))[0]
            r["hyper10raw"] = rows.get("ctg_hyper10", {}).get("raw", (float("nan"),))[0]
            r["texact"] = rf["texact"]
            r["tref"] = sum(v[2] for v in rf["brc"].values())   # seconds of every study TD (they seed the atom pools)
        r["hyper60B"] = B.get(net, {}).get("ctg_hyper60", float("nan"))
        r["ours60B"] = B.get(net, {}).get("ours60", float("nan"))
        R.append(r)

    def selv(r, mode, delta, N, crit="br", stage="br+ctg"):
        """log10 C (cotengra, stage) of the best-of-N TD of the group, chosen by crit (br|bk); None if missing."""
        cand = [s for s in r["pool"]["sel"] if s["mode"] == mode and s["delta"] == delta]
        if not cand:
            return None, None
        if N == "all":
            s = max(cand, key=lambda s: s["N"])
        else:
            le = [s for s in cand if s["N"] <= N]
            if not le:
                return None, None
            s = max(le, key=lambda s: s["N"])
        tag = s["tag_br"] if crit == "br" else s["tag_bk"]
        if stage == "jl":
            return (s["br"] if crit == "br" else s["bk"]), s
        v = r["rc"].get("td:" + tag, {}).get(stage)
        return v, s

    def combo(r, maxd=2, ours=True, study=True):
        """Select by br cost (Julia branchstats, no oracle) among our TDs (best of all, widths K..K+maxd) and the
        study's TD pool (exact, relabels, greedy, ND, HBT) of width <= K+maxd; return br+ctg of the winner."""
        c = []
        if ours:
            for s_ in r["pool"]["sel"]:
                if (maxd == 0 and s_["mode"] == "all0") or (maxd > 0 and s_["mode"] == "all"):
                    pass
            grp = "all0" if maxd == 0 else "all"
            v, s_ = selv(r, grp, -1, "all")
            if v is not None:
                c.append((s_["br"], v))
        rf = ref.get(r["net"])
        if study and rf:
            for t, (w, b, _) in rf["brc"].items():
                if w <= r["K"] + maxd + 1e-3 and "br+ctg" in rf["rows"].get(t, {}):
                    c.append((b, rf["rows"][t]["br+ctg"][0]))
        return min(c)[1] if c else None

    cert = [r for r in R if r["cert"] and "ref_br+ctg" in r]
    pr("# Cost-aware selection among (near-)minimum-width TDs: results")
    pr()
    pr(f"networks with pools: {len(R)}; certified optimal width with study reference: {len(cert)}; "
       f"with GPU results: {sum(1 for r in R if r['gpu']['gs'])}; with cotengra recost: {sum(1 for r in R if r['rc'])}")
    pr()
    pr("TDs per network (median / max): " + f"{med([r['ntd'] for r in R]):.0f} / {max([r['ntd'] for r in R] or [0])}; "
       f"pool seconds (front end + per-atom candidates incl. septw/HBT) median {med([r['tpool'] for r in R]):.1f}")
    g = [r["gpu"]["gs"] for r in R if r["gpu"]["gs"]]
    if g:
        pr(f"GPU scoring: {sum(x['n'] for x in g)} TDs, GPU C vs CPU (Host) max |diff| {max(x['dcpu'] for x in g):.2e} bits, "
           f"GPU maxop vs TD width max |diff| {max(x['dw'] for x in g):.2e} bits")
    pr()

    # ---- 1. headline at stage br+ctg on certified networks
    for stage in ("br+ctg", "bucket"):
        crit = "br" if stage == "br+ctg" else "bk"
        pr(f"## 1{'a' if stage == 'br+ctg' else 'b'}. Certified networks, stage {stage} (selection by "
           f"{'our br cost (Julia branchstats)' if crit == 'br' else 'exact bucket C (GPU kernel)'}); log10 C differences, negative = ours cheaper")
        pr()
        pr("| method | N | n | vs exact solver TD: W/T/L, median d | vs best study heuristic TD: W/T/L, median d | vs best study opt-width TD: W/T/L, median d | vs cotengra hyper10+ctg: W/T/L, median d | median t (s) |")
        pr("|---|---|---|---|---|---|---|---|")
        rows_spec = [("ref", 0, "all")] + [("uni", 0, N) for N in (1, 10, 100, 1000, "all")] + [("dp", 0, 1)] + \
                    [("dpn", 0, N) for N in (10, 100, 1000, "all")] + [("dp", 1, 1), ("dpn", 1, "all"), ("dp", 2, 1), ("dpn", 2, "all")] + \
                    [("all0", -1, N) for N in (10, 100, 1000, "all")] + [("all", -1, "all")]
        for mode, d, N in rows_spec:
            de, dh, do, dc, ts = [], [], [], [], []
            n = 0
            for r in cert:
                v, s = selv(r, mode, d, N, crit, stage)
                if v is None:
                    continue
                rf = r[f"ref_{stage}"]; n += 1
                de.append(v - rf["exact"]); dh.append(v - rf["heur"]); do.append(v - rf["optbest"])
                if stage == "br+ctg":
                    dc.append(v - r["hyper10"])
                ts.append(s["t"])
            if n == 0:
                continue
            lab = {"ref": "study TD pool alone (exact + relabels, greedy, ND, HBT)", "uni": "uniform random opt-width TDs (exact solver relabels, greedy, HBT)", "dp": "cost-aware PMC DP (pmccost)",
                   "dpn": "PMC DP + cost noise", "all0": "all generators, width K", "all": "all generators, widths K..K+2"}[mode]
            dl = "" if d < 0 else f" (width <= K+{d})"
            pr(f"| {lab}{dl} | {N} | {n} | {wtl(de)}, {med(de):+.3f} | {wtl(dh)}, {med(dh):+.3f} | {wtl(do)}, {med(do):+.3f} | "
               f"{wtl(dc) + ', ' + format(med(dc), '+.3f') if dc else '-'} | {med(ts):.1f} |")
        pr()

    # ---- 2. GPU: mass search + polish
    pr("## 2. GPU width-capped mass search (10 s, insertion moves around the cheapest width-K TDs, only maxop = TD width <= K kept) and GPU polish (reconf K=10, ILS, K=12; 10 s), stage as labelled, certified networks")
    pr()
    pr("| tree | n | vs exact solver TD (br+ctg) | vs best study opt TD (br+ctg) | vs best study TD-tree any width (br+ctg) | vs cotengra hyper10+ctg | median candidates |")
    pr("|---|---|---|---|---|---|---|")
    for item, stage, lab in (("gms", "bucket", "GPU mass search, bucket tree"), ("gms", "bucket+ctg", "GPU mass search + ctg"),
                             ("pol:br", "+ctg", "polish(br of selected TD) + ctg"), ("pol:bk", "+ctg", "polish(bucket of GPU-selected TD) + ctg"),
                             ("pol:gms", "+ctg", "polish(mass-search winner) + ctg"), ("polmin", "+ctg", "best polished")):
        de, do, db, dc, nc = [], [], [], [], []
        for r in cert:
            if item == "polmin":
                vs = [r["rc"].get(k, {}).get("+ctg") for k in ("pol:br", "pol:bk", "pol:gms")]
                vs = [x for x in vs if x is not None]
                v = min(vs) if vs else None
            else:
                v = r["rc"].get(item, {}).get(stage)
            if v is None:
                continue
            rf = r["ref_br+ctg"]
            de.append(v - rf["exact"]); do.append(v - rf["optbest"]); db.append(v - rf["best"]); dc.append(v - r["hyper10"])
            if r["gpu"]["gms"]:
                nc.append(r["gpu"]["gms"]["ncand"])
        if de:
            pr(f"| {lab} | {len(de)} | {wtl(de)}, {med(de):+.3f} | {wtl(do)}, {med(do):+.3f} | {wtl(db)}, {med(db):+.3f} | {wtl(dc)}, {med(dc):+.3f} | {med(nc):.2e} |")
    pr()

    # ---- 3. spread recovered
    pr("## 3. How much of the spread among width-optimal TDs is recovered (certified networks with study spread >= 0.01)")
    pr()
    pr("spread = C(worst study opt-width TD) - C(best study opt-width TD); gain = C(exact solver's TD) - C(ours); "
       "'recovered' = gain / (C(exact) - C(best study opt TD)) where that denominator >= 0.01. Stage br+ctg.")
    pr()
    pr("| ours | n | median study spread | median gain over exact TD | median recovered | ours <= best study opt TD (+0.005) | ours < best study opt TD (-0.005) |")
    pr("|---|---|---|---|---|---|---|")
    for lab, fn in (("uniform best-of-all (width K)", lambda r: selv(r, "uni", 0, "all")[0]),
                    ("PMC DP (width K)", lambda r: selv(r, "dp", 0, 1)[0]),
                    ("DP+noise best-of-all (width K)", lambda r: selv(r, "dpn", 0, "all")[0]),
                    ("all generators width K", lambda r: selv(r, "all0", -1, "all")[0]),
                    ("all generators widths K..K+2", lambda r: selv(r, "all", -1, "all")[0]),
                    ("study pool alone, selected by br (width K)", lambda r: combo(r, 0, ours=False)),
                    ("ours + study pool, selected by br (width K)", lambda r: combo(r, 0)),
                    ("ours + study pool, selected by br (widths K..K+2)", lambda r: combo(r, 2)),
                    ("+ GPU mass search + polish (best tree)", lambda r: min([x for x in [selv(r, "all", -1, "all")[0]] +
                     [r["rc"].get(k, {}).get(s) for k, s in (("gms", "bucket+ctg"), ("pol:br", "+ctg"), ("pol:bk", "+ctg"), ("pol:gms", "+ctg"))] if x is not None], default=None))):
        sp, gain, rec, le, lt = [], [], [], 0, 0
        n = 0
        for r in cert:
            rf = r["ref_br+ctg"]
            if rf["nopt"] < 2 or rf["optworst"] - rf["optbest"] < 0.01:
                continue
            v = fn(r)
            if v is None:
                continue
            n += 1
            sp.append(rf["optworst"] - rf["optbest"]); gain.append(rf["exact"] - v)
            den = rf["exact"] - rf["optbest"]
            if den >= 0.01:
                rec.append((rf["exact"] - v) / den)
            le += v <= rf["optbest"] + 0.005; lt += v < rf["optbest"] - 0.005
        if n:
            pr(f"| {lab} | {n} | {med(sp):.3f} | {med(gain):+.3f} | {med(rec):.2f} (n={len(rec)}) | {le}/{n} | {lt}/{n} |")
    pr()

    # ---- 4. per family
    pr("## 4. Per family (certified, br+ctg): best-of-all at width K / any width / + GPU, minus exact solver TD and minus cotengra hyper10+ctg")
    pr()
    pr("| family | n | all0 - exact | all - exact | +GPU - exact | all - hyper10 | +GPU - hyper10 |")
    pr("|---|---|---|---|---|---|---|")
    fams = defaultdict(list)
    for r in cert:
        fams[r["fam"]].append(r)
    for fam, rs in sorted(fams.items()):
        a0, a, gp, ah, gh = [], [], [], [], []
        for r in rs:
            rf = r["ref_br+ctg"]
            v0 = selv(r, "all0", -1, "all")[0]; v = selv(r, "all", -1, "all")[0]
            vg = [x for x in [v] + [r["rc"].get(k, {}).get(s) for k, s in (("gms", "bucket+ctg"), ("pol:br", "+ctg"), ("pol:bk", "+ctg"), ("pol:gms", "+ctg"))] if x is not None]
            if v0 is not None:
                a0.append(v0 - rf["exact"])
            if v is not None:
                a.append(v - rf["exact"]); ah.append(v - r["hyper10"])
            if vg:
                gp.append(min(vg) - rf["exact"]); gh.append(min(vg) - r["hyper10"])
        pr(f"| {fam} | {len(rs)} | {med(a0):+.3f} | {med(a):+.3f} | {med(gp):+.3f} | {med(ah):+.3f} | {med(gh):+.3f} |")
    pr()

    # ---- 5. B networks vs cotengra hyper60
    pr("## 5. The 24 real-contraction networks (experiment B): ours vs cotengra hyper60 and our previous pipeline (ours60), log10 C")
    pr()
    pr("| net | K (cert) | exact TD br+ctg (study A) | sel all br+ctg | +GPU best | gen s | ours60 (B) | ctg hyper60 (B) | best - hyper60 |")
    pr("|---|---|---|---|---|---|---|---|---|")
    dB = []
    for r in R:
        if r["net"] not in Bnets or math.isnan(r["hyper60B"]):
            continue
        v = selv(r, "all", -1, "all")[0]
        vg = [x for x in [v] + [r["rc"].get(k, {}).get(s) for k, s in (("gms", "bucket+ctg"), ("pol:br", "+ctg"), ("pol:bk", "+ctg"), ("pol:gms", "+ctg"))] if x is not None]
        best = min(vg) if vg else None
        ex = r.get("ref_br+ctg", {}).get("exact", float("nan"))
        tg = max([s["t"] for s in r["pool"]["sel"] if s["mode"] == "all"], default=float("nan"))
        if best is not None:
            dB.append(best - r["hyper60B"])
        pr(f"| {r['net']} | {r['K']:.2f} ({int(r['cert'])}) | {ex:.3f} | {v if v is not None else float('nan'):.3f} | "
           f"{best if best is not None else float('nan'):.3f} | {tg:.0f} | {r['ours60B']:.3f} | {r['hyper60B']:.3f} | "
           f"{(best - r['hyper60B']) if best is not None else float('nan'):+.3f} |")
    if dB:
        pr()
        pr(f"best (selection + GPU) vs hyper60: W/T/L {wtl(dB)}, median {med(dB):+.3f}")
    pr()

    # ---- 6. time to quality
    pr("## 6. Time to quality (certified networks, br+ctg of the best-of-N, all generators at width K, by generation time; "
       "+ the seconds of the study's TDs that seed the atom pools: exact solves incl. certification, relabels, greedy, ND, HBT)")
    pr()
    pr("| N | median t_gen (s) | median t_gen + t_ref (s) | median d vs exact TD | median d vs best study opt TD |")
    pr("|---|---|---|---|---|")
    for N in (1, 3, 10, 30, 100, 300, 1000, 3000, "all"):
        ts, te, de, do = [], [], [], []
        for r in cert:
            v, s = selv(r, "all0", -1, N)
            if v is None:
                continue
            rf = r["ref_br+ctg"]
            ts.append(s["t"]); te.append(s["t"] + r.get("tref", float("nan"))); de.append(v - rf["exact"]); do.append(v - rf["optbest"])
        if ts:
            pr(f"| {N} | {med(ts):.1f} | {med(te):.1f} | {med(de):+.3f} | {med(do):+.3f} |")
    pr()
    gm = [r["gpu"]["gmt"] for r in cert if r["gpu"]["gmt"]]
    if gm:
        pr("GPU mass search trace (bucket stage, log10 C drop from its seed, median over networks) at:")
        for T in (0.1, 0.3, 1, 3, 10):
            dr = []
            for tr in gm:
                b = [c for (t, n, c) in tr if t <= T]
                dr.append((b[-1] - tr[0][2]) * L2)
            pr(f"- {T} s: {med(dr):+.3f}")
        pr()

    # ---- 7. per network
    pr("## 7. Per network (certified; br+ctg)")
    pr()
    pr("| net | n | K | atoms (max n) | #TD | exact | best study opt | best study heur | uni all | dp | dpn all | all K..K+2 | +GPU | hyper10 |")
    pr("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in cert:
        rf = r["ref_br+ctg"]

        def f(x):
            return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
        v = selv(r, "all", -1, "all")[0]
        vg = [x for x in [v] + [r["rc"].get(k, {}).get(s) for k, s in (("gms", "bucket+ctg"), ("pol:br", "+ctg"), ("pol:bk", "+ctg"), ("pol:gms", "+ctg"))] if x is not None]
        pr(f"| {r['net']} | {r['pool']['net']['n']} | {r['K']:.2f} | {r['natoms']} ({r['nmax']}) | {r['ntd']} | {f(rf['exact'])} | {f(rf['optbest'])} | "
           f"{f(rf['heur'])} | {f(selv(r, 'uni', 0, 'all')[0])} | {f(selv(r, 'dp', 0, 1)[0])} | {f(selv(r, 'dpn', 0, 'all')[0])} | {f(v)} | "
           f"{f(min(vg) if vg else None)} | {f(r['hyper10'])} |")
    # ---- 8. real contraction (contractB.py on pklB)
    cp = sys.argv[9] if len(sys.argv) > 9 else None
    if cp and os.path.exists(cp):
        rows = defaultdict(dict)
        for l in open(cp):
            x = l.rstrip("\n").split("\t")
            if len(x) < 14 or x[0] == "net" or x[13] != "ok":
                continue
            rows[x[0]][x[1].replace("td_hyb300", "td_exact")] = dict(C=float(x[2]), t=float(x[7]), s=float(x[6]), err=x[11])
        pr("## 8. Real B200 contraction (torch complex64, median of 3) of the networks where selection changed the tree")
        pr()
        pr("| net | csel C / t | csel_gpu C / t | ours60 C / t | hyper60 C / t | exact TD C / t |")
        pr("|---|---|---|---|---|---|")
        for net, d in sorted(rows.items()):
            f = lambda k: f"{d[k]['C']:.3f} / {d[k]['t']:.4g}" if k in d else "-"
            pr(f"| {net} | {f('csel')} | {f('csel_gpu')} | {f('ours60')} | {f('ctg_hyper60')} | {f('td_exact')} |")
        pr()
        for a_, b_ in (("csel", "ours60"), ("csel_gpu", "ours60"), ("csel", "ctg_hyper60"), ("csel_gpu", "ctg_hyper60"), ("csel", "td_exact")):
            rt = []; tts = []; dc = []
            for net, d in rows.items():
                if a_ in d and b_ in d:
                    rt.append(math.log10(d[a_]["t"] / d[b_]["t"])); dc.append(d[a_]["C"] - d[b_]["C"])
                    tts.append((d[a_]["s"] + d[a_]["t"]) / (d[b_]["s"] + d[b_]["t"]))
            if rt:
                W = [sum(x < math.log10(0.95) for x in rt), sum(abs(x) <= math.log10(1.05) for x in rt), sum(x > math.log10(1.05) for x in rt)]
                pr(f"- {a_} vs {b_}: faster/tie/slower {W[0]}/{W[1]}/{W[2]}, median log10 C diff {med(dc):+.3f}, "
                   f"median time ratio x{10 ** med(rt):.2f}, median time-to-solution ratio {med(tts):.2f}")
        pr()
    open(outp, "w").write("\n".join(out) + "\n")
    print("\n".join(out[:60]))


if __name__ == "__main__":
    main()
