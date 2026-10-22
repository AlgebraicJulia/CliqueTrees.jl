#!/usr/bin/env python3
# "Closing the AMD gap" tables (ND_NOTES.md): GPU ND configs vs serial AMD / AMF / MF1 / METIS-ND (+ ParAMD/ParMETIS if present).
# usage: python3 analyze3.py results3/*.tsv  [env BEFORE=gnd:1024 AFTER=gndd:256:hyb=1@32]
# config key = cfg:Lb@threads. Baselines: AMD, METIS = pbase serial.tsv (milan, rmain + AMD.jl, results3/pbase_serial.tsv);
# MF1, AMF = harvest_oct19b/mf/main_table.tsv (milan, rmain); METIS fallback results/cpu_milan.tsv; ParAMD / ParMETIS t32/t64
# from harvest_oct19b/mf/parallel_baselines.tsv. On-node (B200 host CPU) AMD / AMF rows come from the GPU job (amd:0, amf:0).
import sys, csv, math, os
D = os.path.dirname(os.path.abspath(__file__))
H = os.path.join(D, '..', '..', 'harvest_oct19b', 'mf')
base = {}
def put(g, k, nnz, fl, t, tw=None):
    base.setdefault(g, {})[k] = dict(nnz=float(nnz), fl=float(fl), t=float(t))
for r in csv.DictReader(open(os.path.join(H, 'main_table.tsv')), delimiter='\t'):
    if r['cfg'] == 'serial-MF1-new(rmain)': put(r['graph'], 'MF1', r['nnzL'], r['flops'], r['time_s'])
    if r['cfg'] == 'AMF-new(rmain)': put(r['graph'], 'AMF', r['nnzL'], r['flops'], r['time_s'])
for line in open(os.path.join(D, 'results', 'cpu_milan.tsv')):
    f = line.rstrip('\n').split('\t')
    if len(f) >= 10 and f[6] and f[5] == 'METIS': put(f[2], 'METIS', f[6], f[7], f[9])
ps = os.path.join(D, 'results3', 'pbase_serial.tsv')
if os.path.exists(ps):
    for line in open(ps):
        f = line.rstrip('\n').split('\t')
        if len(f) >= 11 and f[10] == 'ok' and f[6]:
            put(f[1], f[4] if f[4] != 'METIS' else 'METIS', f[6], f[7], f[9])
pb = os.path.join(H, 'parallel_baselines.tsv')
if os.path.exists(pb):
    for r in csv.DictReader(open(pb), delimiter='\t'):
        if r.get('status') == 'ok' and r['method'] in ('PARAMD', 'PARMETIS') and r['threads'] in ('32', '64'):
            put(r['graph'], r['method'] + 't' + r['threads'], r['nnzL'], r['flops'], r['time_s'])
gpu = {}; ns = {}
for fn in sys.argv[1:]:
    for line in open(fn):
        f = line.rstrip('\n').split('\t')
        if len(f) < 11: continue
        ex = dict(kv.split('=', 1) for kv in f[11].split(';') if '=' in kv) if len(f) > 11 else {}
        key = f[5] + ':' + f[6] + '@' + ex.get('threads', '?')
        if f[5] in ('amd', 'amf'):
            key = f[5] + '@node'
        gpu[(f[2], key)] = dict(nnz=float(f[7]), fl=float(f[8]), t=float(f[10]), ex=ex)
        ns[f[2]] = int(f[3])
cfgs = sorted({k[1] for k in gpu})
graphs = sorted(ns, key=lambda g: ns[g])
BEFORE = os.environ.get('BEFORE', 'gnd:1024@32'); AFTER = os.environ.get('AFTER', 'gndd:hyb=1:256@32')
def gm(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float('nan')
def bucket(n):
    return '100k-500k' if n < 500_000 else '500k-1M' if n < 1_000_000 else '1-3M' if n < 3_000_000 else '>=3M'
def fx(x, d=2): return '-' if x is None or not math.isfinite(x) else f'{x:.{d}f}'
print(f"## per graph: time ratios = config time / AMD (milan serial); fill = nnz / METIS-ND (BEFORE={BEFORE}, AFTER={AFTER})\n")
print("| graph | n | AMD s (milan / node) | METIS s | MF1 s | before s | after s | after/AMD | after/METIS | after/MF1 | nnz before/METIS | nnz after/METIS | nnz after/MF1 | nnz after/AMD | flops after/METIS |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
rows = []
for g in graphs:
    b = base.get(g, {}); A = b.get('AMD'); M = b.get('METIS'); F = b.get('MF1')
    r0 = gpu.get((g, BEFORE)); r1 = gpu.get((g, AFTER)); an = gpu.get((g, 'amd@node'))
    if not r1: continue
    row = dict(g=g, n=ns[g], b=bucket(ns[g]),
               ta=r1['t'] / A['t'] if A else float('nan'), tm=r1['t'] / M['t'] if M else float('nan'), tf=r1['t'] / F['t'] if F else float('nan'),
               tan=r1['t'] / an['t'] if an else float('nan'), t0a=r0['t'] / A['t'] if (r0 and A) else float('nan'),
               n0=r0['nnz'] / M['nnz'] if (r0 and M) else float('nan'), n1=r1['nnz'] / M['nnz'] if M else float('nan'),
               nf=r1['nnz'] / F['nnz'] if F else float('nan'), na=r1['nnz'] / A['nnz'] if A else float('nan'),
               f1=r1['fl'] / M['fl'] if M else float('nan'))
    rows.append(row)
    print(f"| {g} | {ns[g]} | {fx(A['t'] if A else None, 3)} / {fx(an['t'] if an else None, 3)} | {fx(M['t'] if M else None)} | {fx(F['t'] if F else None)} | "
          f"{fx(r0['t'] if r0 else None, 3)} | {r1['t']:.3f} | {fx(row['ta'])} | {fx(row['tm'])} | {fx(row['tf'])} | {fx(row['n0'], 3)} | {fx(row['n1'], 3)} | {fx(row['nf'], 3)} | {fx(row['na'], 3)} | {fx(row['f1'], 3)} |")
print("\n## per bucket (geometric means)\n")
print("| bucket | graphs | before/AMD | after/AMD | after/AMD(node) | after/METIS | after/MF1 | nnz before/METIS | nnz after/METIS | nnz after/MF1 | nnz after/AMD | flops after/METIS | after <= AMD |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for bk in ('100k-500k', '500k-1M', '1-3M', '>=3M'):
    R = [r for r in rows if r['b'] == bk]
    if not R: continue
    win = sum(1 for r in R if r['ta'] <= 1.0)
    print(f"| {bk} | {len(R)} | {gm([r['t0a'] for r in R]):.2f} | {gm([r['ta'] for r in R]):.2f} | {gm([r['tan'] for r in R]):.2f} | {gm([r['tm'] for r in R]):.3f} | "
          f"{gm([r['tf'] for r in R]):.3f} | {gm([r['n0'] for r in R]):.3f} | {gm([r['n1'] for r in R]):.3f} | {gm([r['nf'] for r in R]):.3f} | {gm([r['na'] for r in R]):.3f} | {gm([r['f1'] for r in R]):.3f} | {win}/{len(R)} |")
print("\n## all configs: time s / nnz vs METIS\n")
print("| graph | " + " | ".join(cfgs) + " |")
print("|---|" + "---|" * len(cfgs))
for g in graphs:
    M = base.get(g, {}).get('METIS')
    cells = []
    for c in cfgs:
        r = gpu.get((g, c))
        cells.append(f"{r['t']:.3f} / {r['nnz'] / M['nnz']:.3f}" if (r and M) else (f"{r['t']:.3f}" if r else '-'))
    print(f"| {g} | " + " | ".join(cells) + " |")
