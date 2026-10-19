import sys, csv, statistics as st, math
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
by = defaultdict(dict)
for r in rows: by[r['name']][r['alg']] = (float(r['width']), float(r['lcost']), float(r['time']), int(r['n']))
algs = sorted({r['alg'] for r in rows})
names = [nm for nm in by if len(by[nm]) == len(algs) and all(not math.isnan(v[0]) for v in by[nm].values())]
print(len(names), "complete instances;", len(by), "total")
def cmp(a, b, ns):
    d = [by[nm][a][0] - by[nm][b][0] for nm in ns]; dc = [by[nm][a][1] - by[nm][b][1] for nm in ns]
    ta = sum(by[nm][a][2] for nm in ns); tb = sum(by[nm][b][2] for nm in ns)
    gm = math.exp(st.mean(math.log(by[nm][a][2]/by[nm][b][2]) for nm in ns))
    print(f"{a:16s} vs {b:12s}: dW mean {st.mean(d):+6.2f} med {st.median(d):+5.2f} | dC mean {st.mean(dc):+6.2f} | better/eq/worse {sum(x<-1e-9 for x in d)}/{sum(abs(x)<=1e-9 for x in d)}/{sum(x>1e-9 for x in d)} | time {ta:7.2f}s vs {tb:7.2f}s geomean ratio {gm:.2f}")
for a in algs:
    if a != 'nd-bw': cmp(a, 'nd-bw', names)
for a in ('flat-R-mf1','flat-R-bw','flat-R-mf1+AMF'): cmp(a, 'flat-bw', names)
cmp('flat-R-mf1', 'flat-MF1', names); cmp('nd-R-mf1', 'nd-MF1', names)
