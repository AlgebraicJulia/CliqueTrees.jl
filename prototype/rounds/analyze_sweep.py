import sys, csv, statistics as st, math
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
by = defaultdict(dict)
for r in rows:
    key = (r['alg'], r['score'], r['relax'])
    by[r['name']][key] = (float(r['width']), float(r['lcost']), float(r['time']), int(r['rounds']), int(r['n']))
names = [nm for nm in by if all(not math.isnan(by[nm][k][0]) for k in by[nm])]
bad = [nm for nm in by if nm not in names]
print(f"instances {len(by)}  complete {len(names)}  with failures {len(bad)}: {bad}")
ser = {'md': ('MMD','-','0'), 'mf': ('MF','-','0'), 'mf1': ('MF1','-','0')}
def summarize(label, f_r, f_s, names):
    d = [f_r(nm)[0] - f_s(nm)[0] for nm in names]
    dc = [f_r(nm)[1] - f_s(nm)[1] for nm in names]
    tr = sum(f_r(nm)[2] for nm in names); ts = sum(f_s(nm)[2] for nm in names)
    better = sum(x < -1e-9 for x in d); worse = sum(x > 1e-9 for x in d)
    print(f"{label:28s} dW mean {st.mean(d):+7.2f} med {st.median(d):+6.2f}  dC mean {st.mean(dc):+7.2f}  better/eq/worse {better:3d}/{len(d)-better-worse:3d}/{worse:3d}  time {tr:8.2f}s vs {ts:8.2f}s")
for big in (False, True):
    ns = [nm for nm in names if (by[nm][ser['mf']][4] >= 1000) == big] if big else names
    print("== all instances" if not big else "== n >= 1000", len(ns))
    for sel in ('luby', 'greedy'):
        for sc in ('md', 'mf', 'mf1'):
            for rx in ('0.0','0.1','0.25','0.5','1.0','2.0','Inf'):
                k = (sel, sc, rx)
                if k not in by[ns[0]]: continue
                summarize(f"{sel} {sc} relax={rx}", lambda nm: by[nm][k], lambda nm: by[nm][ser[sc]], ns)
    # best-of: rounds bw vs serial bw
    for rx in ('0.0','0.25','0.5','1.0','2.0','Inf'):
        def rb(nm):
            c = [by[nm][('luby', sc, rx)] for sc in ('md','mf','mf1')]
            b = min(c, key=lambda x: (x[0], x[1]))
            return (b[0], b[1], sum(x[2] for x in c))
        def sb(nm):
            c = [by[nm][(a,'-','0')] for a in ('MF','MF1','AMF','MMD')]
            b = min(c, key=lambda x: (x[0], x[1]))
            return (b[0], b[1], sum(x[2] for x in c))
        summarize(f"R-bw(md,mf,mf1) relax={rx}", rb, sb, ns)
