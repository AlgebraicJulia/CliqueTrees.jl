#!/usr/bin/env python3
# Differences between two versions in bench/ab.jl results.
#
#   python3 bench/ab_compare.py results.jsonl base new [--top=8]
#
# Tags are matched by prefix (base-r1, base-r2, ... all count as "base"). Per graph: the wall time of
# plain calls (best and median over all rounds), then the steps, kernel families, CUDA API calls and
# copies that moved the most (best of the rounds for each), GPU busy share, the longest idle stretches,
# and whether the checksums agree.
import json, sys, statistics, collections

path, A, B = sys.argv[1:4]
TOP = int(next((a.split('=')[1] for a in sys.argv[4:] if a.startswith('--top=')), 8))
rows = [json.loads(l) for l in open(path) if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    for t in (A, B):
        if r['tag'] == t or r['tag'].startswith(t + '-'):
            by[(t, r['graph'])].append(r)
graphs = []
for r in rows:
    if r['graph'] not in graphs: graphs.append(r['graph'])

ms = lambda x: 1e3 * x
def best(rs, get):
    vals = [v for v in (get(r) for r in rs) if v is not None]
    return min(vals) if vals else None
def merged(rs, key, sub=None):
    out = {}
    for r in rs:
        d = r['profile'][key] if sub is None else r[key]
        for k, v in d.items():
            t = v['time'] if isinstance(v, dict) else v
            out[k] = min(out.get(k, float('inf')), t)
    return out

def fmt_delta(a, b):
    if a is None or b is None: return ''
    d = b - a
    return f'{ms(d):+8.2f}'

summary = []
for g in graphs:
    ra, rb = by.get((A, g)), by.get((B, g))
    if not ra or not rb: continue
    pa = [x for r in ra for x in r['plain']]; pb = [x for r in rb for x in r['plain']]
    ca = {tuple(r['checksum']) for r in ra if r.get('checksum')}; cb = {tuple(r['checksum']) for r in rb if r.get('checksum')}
    same = ca == cb and len(ca) == 1
    print(f"\n=== {g}  n={ra[0]['n']}  checksum {'same' if same else 'DIFFERENT ' + str(ca) + ' vs ' + str(cb)}")
    print(f"  plain call     best {ms(min(pa)):8.1f} → {ms(min(pb)):8.1f} ms ({min(pa)/min(pb):.2f}×)   median {ms(statistics.median(pa)):8.1f} → {ms(statistics.median(pb)):8.1f} ms ({statistics.median(pa)/statistics.median(pb):.2f}×)   GC/call {ms(statistics.median([x for r in ra for x in r['plaingc']])):.1f} → {ms(statistics.median([x for r in rb for x in r['plaingc']])):.1f} ms")
    summary.append((g, min(pa), min(pb), statistics.median(pa), statistics.median(pb), same))
    span_a = best(ra, lambda r: r['profile']['span']); span_b = best(rb, lambda r: r['profile']['span'])
    busy_a = best(ra, lambda r: r['profile']['busy']); busy_b = best(rb, lambda r: r['profile']['busy'])
    print(f"  profiled call  span {ms(span_a):8.1f} → {ms(span_b):8.1f} ms   GPU busy {ms(busy_a):7.1f} → {ms(busy_b):7.1f} ms ({100*busy_a/span_a:.0f}% → {100*busy_b/span_b:.0f}%)")
    for title, da, db in (('steps', merged(ra, 'steps', True), merged(rb, 'steps', True)),
                          ('kernel families', merged(ra, 'kernels'), merged(rb, 'kernels')),
                          ('copies', merged(ra, 'copies'), merged(rb, 'copies')),
                          ('CUDA API (host)', merged(ra, 'api'), merged(rb, 'api'))):
        keys = set(da) | set(db)
        ch = sorted(keys, key=lambda k: -abs(db.get(k, 0) - da.get(k, 0)))[:TOP]
        ch = [k for k in ch if abs(db.get(k, 0) - da.get(k, 0)) >= 2e-4]
        if not ch: continue
        print(f"  {title}:")
        for k in ch:
            a, b = da.get(k), db.get(k)
            print(f"    {k[:60]:60s} {('%8.2f' % ms(a)) if a is not None else '       –'} → {('%8.2f' % ms(b)) if b is not None else '       –'} ms  {fmt_delta(a or 0, b or 0)}")
    rb0 = min(rb, key=lambda r: r['profile']['span'])
    gaps = rb0['profile']['idle'][:4]
    if gaps:
        print("  longest GPU-idle stretches (" + B + "): " + '; '.join(f"{ms(x['length']):.1f} ms at {ms(x['at']):.0f} ms [{', '.join(f'{k} {ms(v):.1f}' for k, v in x['host'].items())}]" for x in gaps))

print("\n=== summary (plain call, ms)")
print(f"{'graph':18s} {'best ' + A:>12s} {'best ' + B:>12s} {'×':>6s} {'median ' + A:>14s} {'median ' + B:>14s} {'×':>6s}  checksum")
for g, a, b, ma, mb, same in summary:
    print(f"{g:18s} {ms(a):12.1f} {ms(b):12.1f} {a/b:6.2f} {ms(ma):14.1f} {ms(mb):14.1f} {ma/mb:6.2f}  {'same' if same else 'DIFFERENT'}")
if summary:
    print(f"geometric mean speedup: best {statistics.geometric_mean([a/b for _, a, b, *_ in summary]):.3f}×, median {statistics.geometric_mean([ma/mb for *_, ma, mb, _ in summary]):.3f}×")
