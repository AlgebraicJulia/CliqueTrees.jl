#!/usr/bin/env python3
"""Tables of the end-to-end batch runs (e2e_gpu.jl + e2e_cpu.jl rows).
usage: analyze_e2e.py e2e_gpu.tsv e2e_cpu.tsv[,e2e_cpu2.tsv,...] [REP]   (REP: GPU pass, default a; b = second pass run during
the CPU baselines). Several CPU files: the serial time is the min over them (the strongest baseline).
GPU times: whole grounds_batch call (host graphs in, host permutations out), min of 3. new = best of the new configurations
(nw 8/16/32, chunks auto/1); serial = Richard's MF1, T workers over the graphs, BipartiteGraph conversion excluded."""
import csv, collections, sys

gf, cf = sys.argv[1], sys.argv[2]
REP = sys.argv[3] if len(sys.argv) > 3 else 'a'
nan = float('nan')
G = collections.defaultdict(dict); C = collections.defaultdict(dict); meta = {}
def ex(s):
    return dict(kv.split('=', 1) for kv in s.split(';') if '=' in kv)
for r in csv.reader(open(gf), delimiter='\t'):
    if len(r) < 11 or r[1] != REP:
        continue
    G[(r[2], r[3])][r[7]] = (float(r[9]), int(r[8]) if r[8] else 0, ex(r[10]))
    meta[(r[2], r[3])] = (int(r[4]), int(r[5]))
for f1 in cf.split(','):
    for r in csv.reader(open(f1), delimiter='\t'):
        if len(r) < 11:
            continue
        old = C[(r[2], r[3])].get(r[7])
        if old is None or float(r[9]) < old[0]:
            C[(r[2], r[3])][r[7]] = (float(r[9]), int(r[8]) if r[8] else 0)

def best_new(d):
    # the default configuration when the run has it (v5: 'new'), else the best of the new configurations (v4)
    if 'new' in d:
        return 'new', d['new'][0]
    ks = [k for k in d if k.startswith('new-')]
    if not ks:
        return None, nan
    k = min(ks, key=lambda k: d[k][0])
    return k, d[k][0]
f = lambda x: float(x)

# 1. stage breakdown, v2 vs new (one block per graph, nw 16, chunks 1, timed run)
print("== stage breakdown (ms): v2 (nw16) vs new (1 chunk: v4 nw16, v5 default block shape); the timed run syncs between stages")
print("%-20s | %6s %6s %6s %6s %6s %3s | %5s %5s %5s %5s %5s %5s %6s %5s %5s %5s %3s" % (
    "set", "csr", "pre", "gpu", "post", "tot", "L", "pack", "h2d", "twin", "live", "alloc", "pre", "kernel", "grow", "rel", "post", "L"))
for key in sorted(G):
    if key[0] != 'batch' or not key[1][0].isupper() and not key[1].startswith(('ecology', 'thermal', 'lat2', 'audikw')):
        continue
    d = G[key]
    nk = 'new-c1' if 'new-c1' in d else 'new-nw16-c1'
    if 'v2-nw16' not in d or nk not in d:
        continue
    o = d['v2-nw16'][2]; n = d[nk][2]
    ms = lambda e, k: 1000 * f(e.get(k, 0))
    print("%-20s | %6.1f %6.1f %6.1f %6.1f %6.1f %3s | %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %6.1f %5.1f %5.1f %5.1f %3s" % (
        key[1], ms(o, 'csr'), ms(o, 'pre'), ms(o, 'gpu'), ms(o, 'cpu_post'), 1000 * d['v2-nw16'][0], o.get('launches', '?'),
        ms(n, 'pre_size') + ms(n, 'pre_pack'), ms(n, 'pre_h2d'), ms(n, 'pre_twins'), ms(n, 'pre_live'), ms(n, 'pre_alloc'), ms(n, 'pre'),
        ms(n, 'kernel'), ms(n, 'grow'), ms(n, 'relaunch'), ms(n, 'cpu_post'), n.get('launches', '?')))

# 2. end to end vs serial
def table(title, pred):
    print()
    print("== " + title)
    print("%-22s %5s %7s | %7s %7s %-15s | %7s %7s %7s | %6s %6s %6s %6s | %5s" % (
        "set", "B", "nsum", "v2", "new", "(cfg)", "ser1t", "ser8t", "ser32t", "v2/new", "x1t", "x8t", "x32t", "nnz"))
    for key in sorted(G, key=lambda k: (k[1].split('_x')[0], meta[k][0])):
        if key[0] != 'batch' or not pred(key[1]):
            continue
        d = G[key]; c = C.get(key, {})
        k, tn = best_new(d)
        if k is None:
            continue
        tv = d.get('v2-nw16', (nan,))[0]
        s1 = c.get('serial-MF1-1t', (nan, 0)); s8 = c.get('serial-MF1-8t', (nan, 0)); s32 = c.get('serial-MF1-32t', (nan, 0))
        q = d[k][1] / s1[1] if s1[1] else nan
        e = d[k][2]; cfg = k[4:] if k != 'new' else "nw%s/%s" % (e.get('nw', '?'), e.get('maxregs', '?'))
        print("%-22s %5d %7d | %7.4f %7.4f %-15s | %7.4f %7.4f %7.4f | %6.2f %6.2f %6.2f %6.2f | %5.3f" % (
            key[1], meta[key][0], meta[key][1], tv, tn, cfg, s1[0], s8[0], s32[0], tv / tn, s1[0] / tn, s8[0] / tn, s32[0] / tn, q))
table("ND leaf sets (METIS separators of 1M-vertex graphs, leaf bound L)", lambda n: not n.startswith('sweep'))
table("sweep sets (random leaves of bound L, B graphs)", lambda n: n.startswith('sweep'))

# 3. crossover grid: speedup of the new batch vs 8 and 32 threads
print()
print("== crossover: new GPU batch / serial MF1, end to end (x8t | x32t); > 1 = GPU faster")
rows = collections.defaultdict(dict)
for key in G:
    if key[0] == 'batch' and key[1].startswith('sweep_L'):
        L, c = key[1][7:].split('_x'); rows[int(L)][int(c)] = key
counts = sorted({c for L in rows for c in rows[L]})
print("%-8s " % "L \\ B" + " ".join("%13d" % c for c in counts))
for L in sorted(rows):
    cells = []
    for cnt in counts:
        key = rows[L].get(cnt)
        if key is None:
            cells.append("%13s" % "-"); continue
        _, tn = best_new(G[key]); c = C.get(key, {})
        cells.append("%6.2f|%6.2f" % (c.get('serial-MF1-8t', (nan,))[0] / tn, c.get('serial-MF1-32t', (nan,))[0] / tn))
    print("%-8d " % L + " ".join(cells))

# 4. single graphs
print()
print("== single graphs (one grounds call, end to end) vs serial MF1 on 1 thread")
print("%-24s %7s | %7s %7s %7s | %7s | %6s %6s %6s | %s" % ("graph", "n", "replay", "auto", "grid", "ser1t", "x rep", "x auto", "x grid", "replay csr/pre/gpu/post ms"))
for key in sorted(G, key=lambda k: meta[k][1]):
    if key[0] != 'single':
        continue
    d = G[key]; s = C.get(key, {}).get('serial-MF1-1t', (nan, 0))
    rep = d.get('replay', (nan, 0, {})); au = d.get('auto', (nan,))[0]
    gr = min([v[0] for k, v in d.items() if k.startswith('grid')] or [nan])
    e = rep[2]
    br = "/".join("%.1f" % (1000 * f(e.get(k, nan))) for k in ('csr', 'pre', 'gpu', 'cpu_post'))
    print("%-24s %7d | %7.4f %7.4f %7.4f | %7.4f | %6.2f %6.2f %6.2f | %s" % (
        key[1], meta[key][1], rep[0], au, gr, s[0], s[0] / rep[0], s[0] / au, s[0] / gr, br))
