"""usage: resume_queue.py <outdir> <old_jobid>  -> writes <outdir>/resume_{einsum,frontier}.txt
= leftover queue_*.<old_jobid> lines + tasks that were popped but have no row (killed in flight / crashed) + the last
task of each old result file (possibly cut mid-ladder). Duplicated rows are harmless (harvest takes the best per budget)."""
import sys, glob, csv, os
out, job = sys.argv[1], sys.argv[2]
for kind in ("einsum", "frontier"):
    tasks = [l.strip() for l in open(f"{out}/tasks_{kind}.txt") if l.strip()]
    left = [l.strip() for l in open(f"{out}/queue_{kind}.{job}") if l.strip()]
    seen, last = set(), set()
    for fn in glob.glob(f"{out}/res/cutn_t*_*.tsv"):
        if fn.endswith(("_smoke_e.tsv", "_smoke_f.tsv")): continue
        rows = list(csv.DictReader(open(fn), delimiter="\t"))
        for r in rows:
            if r["source"] == kind:
                seen.add((r["network"], r["cap"], r["extra"].split(";")[0]))
        if rows and rows[-1]["source"] == kind:
            last.add((rows[-1]["network"], rows[-1]["cap"], rows[-1]["extra"].split(";")[0]))
    res = []
    lset = set(left)
    for t in tasks:
        p = t.split(); memf = f"memf={float(p[4]) if len(p) > 4 else 1.0}"
        key = (p[1], p[3], memf)
        if t in lset or key not in seen or key in last:
            res.append(t)
    open(f"{out}/resume_{kind}.txt", "w").write("\n".join(res) + "\n")
    print(kind, len(tasks), "tasks ->", len(res), "resume (", len(left), "leftover )")
