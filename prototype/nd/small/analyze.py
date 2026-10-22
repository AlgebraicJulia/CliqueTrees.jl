#!/usr/bin/env python3
"""Tables of the small-graph sweep (sgpu.jl + sser.jl rows).
usage: analyze.py GPU.tsv CPU.tsv
single: serial = Richard's new MF1 on the B200 node CPU (1 core); x/s = serial time / GPU time
batch:  gpu-tot = grounds_batch wall time (CSR, setup, kernel, orders); gpu-k = kernel loop only"""
import csv, collections, sys

gf, cf = sys.argv[1], sys.argv[2]
gpu = collections.defaultdict(dict); meta = {}; bat = collections.defaultdict(dict)
for r in csv.reader(open(gf), delimiter='\t'):
    if len(r) < 12 or r[7] == 'error':
        continue
    ex = dict(kv.split('=', 1) for kv in r[11].split(';') if '=' in kv)
    if r[2] == 'single':
        gpu[r[3]][r[6]] = (float(r[10]), int(r[7]), ex); meta[r[3]] = int(r[4])
    elif r[2] == 'batch':
        bat[r[3]][r[6]] = (float(r[10]), int(r[7]) if r[7] else 0, float(ex.get('gpu', 'nan')), ex)
cpu = {}
for r in csv.reader(open(cf), delimiter='\t'):
    if r[2] == 'single':
        cpu[r[3]] = (float(r[10]), int(r[7]))
    elif r[2] == 'batch':
        bat[r[3]][r[6]] = (float(r[10]), int(r[7]), float('nan'), {})

nan = float('nan')
print("%-28s %7s %8s %8s %8s %8s %8s | %6s %6s %6s %6s | %5s" % (
    "graph", "n", "serial", "replay", "bestblk", "bestgrid", "auto", "rep/s", "grid/s", "auto/s", "best/s", "nnz"))
for g in sorted(gpu, key=lambda g: meta[g]):
    d = gpu[g]; s = cpu.get(g, (nan, 0))
    rep = d.get('replay', (nan,))[0]
    blk = min([v[0] for k, v in d.items() if k.startswith('block')] or [nan])
    grd = min([v[0] for k, v in d.items() if k.startswith('grid')] or [nan])
    au = d.get('auto', (nan,))[0]
    best = min(rep, blk, grd)
    q = d['replay'][1] / s[1] if s[1] and 'replay' in d else nan
    print("%-28s %7d %8.4f %8.4f %8.4f %8.4f %8.4f | %6.2f %6.2f %6.2f %6.2f | %5.3f" % (
        g, meta[g], s[0], rep, blk, grd, au, s[0] / rep, s[0] / grd, s[0] / au, s[0] / best, q))

print()
print("%-20s %6s %7s | %7s %7s %7s | %7s %7s | %6s %6s %6s %6s | %5s" % (
    "leaves", "count", "nmed", "gpu-tot", "gpu-k", "seqauto", "ser1t", "ser8t", "1t/tot", "8t/tot", "1t/k", "8t/k", "nnz"))
for k in sorted(bat):
    d = bat[k]
    bs = [v for kk, v in d.items() if kk.startswith('batch-')]
    if not bs or 'serial-MF1-1t' not in d:
        continue
    b = min(bs, key=lambda v: v[0])
    s1 = d['serial-MF1-1t']; s8 = d.get('serial-MF1-8t', (nan, 0))
    sa = d.get('seq-auto', (nan,))[0]
    print("%-20s %6s %7s | %7.3f %7.3f %7.2f | %7.3f %7.3f | %6.2f %6.2f %6.2f %6.2f | %5.3f" % (
        k, b[3].get('leaves', '?'), b[3].get('nmed', '?'), b[0], b[2], sa, s1[0], s8[0],
        s1[0] / b[0], s8[0] / b[0], s1[0] / b[2], s8[0] / b[2], b[1] / s1[1]))
