# Merge CPU (septw, HBT) and GPU (PID-BT) results per atom; tables by atom size.
# python3 analyze.py <atoms_all.tsv> <cpu.tsv> <gpu1.tsv> [gpu2.tsv ...]
import sys, statistics as S
atoms = {}
for l in open(sys.argv[1]):
    f = l.rstrip('\n').split('\t')
    if len(f) < 8: continue
    atoms[f[1]] = dict(inst=f[0], n=int(f[2]), m=int(f[3]), lb=float(f[4]), mf=float(f[5]))
cpu = {}; hbt = {}
for l in open(sys.argv[2]):
    f = l.rstrip('\n').split('\t')
    if len(f) < 5: continue
    if f[3] == 'septw' and len(f) >= 8:
        cpu[f[0]] = dict(st=f[4], t=float(f[5]), w=float(f[6]) if f[6] != 'NaN' else None)
    elif f[3] == 'septw':
        cpu[f[0]] = dict(st='KILLED', t=float('nan'), w=None)
    elif f[3] == 'hbt' and len(f) >= 7:
        hbt[f[0]] = float(f[6])
gpu = {}
for p in sys.argv[3:]:
    for l in open(p):
        f = l.rstrip('\n').split('\t')
        if len(f) < 9 or f[2] == 'ERROR':
            if len(f) >= 3 and f[2] == 'ERROR': gpu.setdefault(f[0], dict(st='error', t=float('nan'), w=None, lb=None))
            continue
        st = f[3]; t = float(f[4]); w = float(f[6]) if f[6] not in ('NaN', 'Inf') else None
        lb = float(f[7])
        old = gpu.get(f[0])
        if old is None or old['st'] != 'found' or (st == 'found' and t < old['t']):
            gpu[f[0]] = dict(st=st, t=t, w=w, lb=lb, src=p)
bins = [(0, 60), (61, 100), (101, 150), (151, 200), (201, 300), (301, 400), (401, 600)]
def med(x): return f"{S.median(x):.3g}" if x else "-"
print("| atom vertices | atoms | CPU septw solved (≤100 s / ≤1000 s) | CPU median s | GPU PID solved (≤100 s) | GPU median s | atoms with both | GPU-only certificates (CPU failed) |")
print("|---|---|---|---|---|---|---|---|")
for lo, hi in bins:
    names = [k for k, a in atoms.items() if lo <= a['n'] <= hi and (k in cpu or k in gpu)]
    if not names: continue
    c100 = [cpu[k]['t'] for k in names if k in cpu and cpu[k]['st'] == 'BT_FOUND' and cpu[k]['t'] <= 100]
    c1000 = [cpu[k]['t'] for k in names if k in cpu and cpu[k]['st'] == 'BT_FOUND']
    g = [gpu[k]['t'] for k in names if k in gpu and gpu[k]['st'] == 'found']
    only = [k for k in names if k in gpu and gpu[k]['st'] == 'found' and k in cpu and cpu[k]['st'] != 'BT_FOUND']
    both = [k for k in names if k in gpu and k in cpu]
    ncpu = sum(1 for k in names if k in cpu); ngpu = sum(1 for k in names if k in gpu)
    print(f"| {lo}–{hi} | {len(names)} | {len(c100)} / {len(c1000)} of {ncpu} | {med(c1000)} | {len(g)} of {ngpu} | {med(g)} | {len(both)} | {len(only)} |")
print()
print("| atom | n | m | MMW lb | HBT 300 s | CPU septw | GPU PID | GPU width | check |")
print("|---|---|---|---|---|---|---|---|---|")
for k in sorted(atoms, key=lambda k: atoms[k]['n']):
    a = atoms[k]
    if a['n'] < 90 or (k not in gpu and k not in cpu): continue
    c = cpu.get(k); g = gpu.get(k)
    cs = '-' if c is None else (f"{c['t']:.1f} s" if c['st'] == 'BT_FOUND' else c['st'].replace('BT_', '').lower())
    gs = '-' if g is None else (f"{g['t']:.1f} s" if g['st'] == 'found' else g['st'])
    gw = '-' if g is None or g['w'] is None else f"{g['w']:.4g}"
    agree = ''
    if c and g and c['st'] == 'BT_FOUND' and g['st'] == 'found':
        agree = 'match' if abs(c['w'] - g['w']) < 1e-6 else f"MISMATCH cpu={c['w']}"
    print(f"| {k} | {a['n']} | {a['m']} | {a['lb']:.4g} | {hbt.get(k, float('nan')):.4g} | {cs} | {gs} | {gw} | {agree} |")
