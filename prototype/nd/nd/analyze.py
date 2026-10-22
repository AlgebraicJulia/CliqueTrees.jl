#!/usr/bin/env python3
# ND_NOTES tables: GPU ND + MF1 leaves vs serial MF1 (main_table), METIS-ND, CliqueTrees ND+MF1, CPU-METIS + GPU leaves.
# usage: python3 analyze.py results/full_*_b200.tsv   (cpu_milan.tsv and main_table.tsv read from fixed paths)
import sys, csv, math, os, statistics as st
D = os.path.dirname(os.path.abspath(__file__))
MT = os.path.join(D, '..', '..', 'harvest_oct19b', 'mf', 'main_table.tsv')
mf1 = {}
for r in csv.DictReader(open(MT), delimiter='\t'):
    if r['cfg'] == 'serial-MF1-new(rmain)':
        mf1[r['graph']] = dict(n=int(r['n']), nnz=float(r['nnzL']), fl=float(r['flops']), tw=int(r['width']), t=float(r['time_s']))
cpu = {}
for line in open(os.path.join(D, 'results', 'cpu_milan.tsv')):
    f = line.rstrip('\n').split('\t')
    if len(f) < 10 or f[6] == '':
        continue
    cpu[(f[2], f[5])] = dict(nnz=float(f[6]), fl=float(f[7]), tw=int(f[8]), t=float(f[9]))
gpu = {}
for fn in sys.argv[1:]:
    for line in open(fn):
        f = line.rstrip('\n').split('\t')
        if len(f) < 11:
            continue
        g, cfg, L = f[2], f[5], f[6]
        gpu[(g, cfg + ':' + L)] = dict(n=int(f[3]), nnz=float(f[7]), fl=float(f[8]), tw=int(f[9]), t=float(f[10]),
                                      extra=dict(kv.split('=', 1) for kv in f[11].split(';') if '=' in kv) if len(f) > 11 else {})
cfgs = sorted({k[1] for k in gpu})
graphs = sorted({k[0] for k in gpu}, key=lambda g: mf1.get(g, {}).get('n', 0))
def bucket(n):
    return '100k-1M' if n < 1_000_000 else ('1-3M' if n < 3_000_000 else '>=3M')
def gm(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float('nan')
main = sys.argv[0] and os.environ.get('MAIN', 'gnd:4096')
print(f"## per graph (main config {main}; ratios: fill = GPU / min(MF1, METIS-ND) [which]; speed = baseline time / GPU time)\n")
print("| graph | n | nnz vs best | flops vs best | best | t GPU s (nd / leaves) | x MF1 | x METIS-ND | x ND+MF1 (CT, best L) | x CPU-METIS+GPU leaves | fill CPU-METIS+GPU leaves vs best |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
rows = []
for g in graphs:
    r = gpu.get((g, main)); m = mf1.get(g); me = cpu.get((g, 'METIS'))
    if not r or not m or not me:
        continue
    bestn = min(m['nnz'], me['nnz']); bestf = min(m['fl'], me['fl'])
    which = 'MF1' if m['nnz'] < me['nnz'] else 'METIS'
    nd = [cpu[(g, a)] for a in ('NDMF1_2048', 'NDMF1_8192') if (g, a) in cpu]
    ndt = min(x['t'] for x in nd) if nd else float('nan')
    mg = gpu.get((g, 'metis:' + main.split(':')[-1])) or gpu.get((g, 'metis:4096'))
    e = r['extra']
    row = dict(g=g, n=r['n'], b=bucket(r['n']), rn=r['nnz'] / bestn, rf=r['fl'] / bestf, t=r['t'], xmf=m['t'] / r['t'], xme=me['t'] / r['t'],
               xnd=ndt / r['t'], xmg=(mg['t'] / r['t']) if mg else float('nan'), mgn=(mg['nnz'] / bestn) if mg else float('nan'),
               rnm=r['nnz'] / m['nnz'], rnme=r['nnz'] / me['nnz'], rfm=r['fl'] / m['fl'], rfme=r['fl'] / me['fl'])
    rows.append(row)
    print(f"| {g} | {r['n']} | {row['rn']:.3f} | {row['rf']:.3f} | {which} | {r['t']:.3f} ({float(e.get('nd', 'nan')):.3f} / {float(e.get('leaves', 'nan')):.3f}) | {row['xmf']:.2f} | {row['xme']:.2f} | {row['xnd']:.1f} | {row['xmg']:.2f} | {row['mgn']:.3f} |")
print("\n## per size bucket (geometric means; fill ratios vs min(MF1, METIS-ND))\n")
print("| bucket | graphs | nnz vs best (max) | flops vs best (max) | nnz vs MF1 | nnz vs METIS | x MF1 (min-max) | x METIS-ND | x ND+MF1 | x CPU-METIS+GPU leaves |")
print("|---|---|---|---|---|---|---|---|---|---|")
for b in ('100k-1M', '1-3M', '>=3M'):
    R = [r for r in rows if r['b'] == b]
    if not R:
        continue
    print(f"| {b} | {len(R)} | {gm([r['rn'] for r in R]):.3f} ({max(r['rn'] for r in R):.3f}) | {gm([r['rf'] for r in R]):.3f} ({max(r['rf'] for r in R):.3f}) | "
          f"{gm([r['rnm'] for r in R]):.3f} | {gm([r['rnme'] for r in R]):.3f} | {gm([r['xmf'] for r in R]):.2f} ({min(r['xmf'] for r in R):.2f}-{max(r['xmf'] for r in R):.2f}) | "
          f"{gm([r['xme'] for r in R]):.2f} | {gm([r['xnd'] for r in R]):.1f} | {gm([r['xmg'] for r in R]):.2f} |")
print("\n## all configs (nnz vs best / x MF1)\n")
print("| graph | " + " | ".join(cfgs) + " |")
print("|---|" + "---|" * len(cfgs))
for g in graphs:
    m = mf1.get(g); me = cpu.get((g, 'METIS'))
    if not m or not me:
        continue
    bestn = min(m['nnz'], me['nnz'])
    cells = []
    for c in cfgs:
        r = gpu.get((g, c))
        cells.append(f"{r['nnz'] / bestn:.3f} / {m['t'] / r['t']:.2f}" if r else "-")
    print(f"| {g} | " + " | ".join(cells) + " |")
