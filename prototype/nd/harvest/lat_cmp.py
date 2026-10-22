import re,math,csv,collections
B='/home/itaykadosh/Projects/GATAS/tn-work/'
base={}  # (graph,cfg)->(tbase,nnzbase,rounds)
for l in open(B+'harvest_oct20/lat/prof1.txt'):
    m=re.match(r'# (\S+) n=(\d+) .* cfg=(\S+) t=(\S+) tbase=(\S+) .* rounds=(\d+) nnz=(\d+) nnzbase=(-?\d+)',l)
    if m:
        tb=float(m[5]) if m[5]!='NaN' else float(m[4]); nb=int(m[8]) if int(m[8])>0 else int(m[7])
        base[(m[1],m[3])]=(tb,nb,int(m[6]),int(m[2]))
mt={}
for r in csv.DictReader(open(B+'harvest_oct19b/mf/main_table.tsv'),delimiter='\t'):
    mt[(r['graph'],r['cfg'])]=(float(r['time_s']),int(float(r['nnzL'])),int(r['n']))
new={}
for l in open(B+'harvest_oct20/lat/rsel_b200.tsv'):
    f=l.split('\t'); new[(f[0],f[3])]=(float(f[8]),int(f[5]),int(f[4]),int(f[1]),int(float(f[6])))
def bk(n): return '<30k' if n<3e4 else '30-100k' if n<1e5 else '100k-1M' if n<1e6 else '1-3M' if n<3e6 else '>3M'
def gm(x): return math.exp(sum(map(math.log,x))/len(x))
rows=[]
for (g,c),(t,nz,rd,n,fl) in sorted(new.items()):
    if c not in('v2','r0.1u0'):continue
    cc={'v2':'v2','r0.1u0':'u0.1'}[c]
    b=base.get((g,cc)); m=mt.get((g,'GPU-MF1-'+cc))
    rows.append((g,n,cc,t,nz,rd,b,m))
print('| graph | n | cfg | new t s | base t s (prof1 job, tbase) | speedup vs base | main_table t s | speedup vs main_table | nnz new==base | nnz new==main_table |')
print('|---|---|---|---|---|---|---|---|---|---|')
S=collections.defaultdict(list);S2=collections.defaultdict(list)
for g,n,cc,t,nz,rd,b,m in rows:
    sb=b[0]/t if b else None; sm=m[0]/t if m else None
    if sb: S[(bk(n),cc)].append(sb)
    if sm: S2[(bk(n),cc)].append(sm)
    print(f"| {g} | {n} | {cc} | {t:.4f} | {b[0]:.4f} | {sb:.2f} | " if b else f"| {g} | {n} | {cc} | {t:.4f} | - | - | ",end='')
    print((f"{m[0]:.4f} | {sm:.2f} | " if m else "- | - | ")+(str(nz==b[1]) if b else '-')+' | '+(str(nz==m[1]) if m else '-')+' |')
print()
print('| bucket | cfg | k(base) | geomean speedup vs prof1 tbase | min | max | k(main_table) | geomean speedup vs main_table | min | max |')
print('|---|---|---|---|---|---|---|---|---|---|')
for b in['<30k','30-100k','100k-1M','1-3M','>3M']:
    for cc in['v2','u0.1']:
        x=S.get((b,cc),[]);y=S2.get((b,cc),[])
        f=lambda v:(f"{gm(v):.2f} | {min(v):.2f} | {max(v):.2f}" if v else "- | - | -")
        print(f"| {b} | {cc} | {len(x)} | {f(x)} | {len(y)} | {f(y)} |")
