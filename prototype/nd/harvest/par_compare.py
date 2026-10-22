import csv,math,collections,sys
B='/home/itaykadosh/Projects/GATAS/tn-work/'
pb=list(csv.DictReader(open(B+'harvest_oct19b/mf/parallel_baselines.tsv'),delimiter='\t'))
mt=list(csv.DictReader(open(B+'harvest_oct19b/mf/main_table.tsv'),delimiter='\t'))
R={}  # (graph,key)->(nnz,flops,time)
N={}
for r in mt:
    N[r['graph']]=int(r['n'])
    k={'GPU-MF1-v2':'GPUMF1','serial-MF1-new(rmain)':'MF1s','GPU-MF1-u0.1':'GPUMF1u'}.get(r['cfg'])
    if k: R[(r['graph'],k)]=(float(r['nnzL']),float(r['flops']),float(r['time_s']))
for r in pb:
    if r['status']!='ok':continue
    k=r['method']+('' if r['method'] in('AMD','METIS') else r['threads'])
    if r['method'] in('AMD','METIS'):k={'AMD':'AMDs','METIS':'METISs'}[r['method']]
    R[(r['graph'],k)]=(float(r['nnzL']),float(r['flops']),float(r['time_s']))
    N[r['graph']]=int(r['n'])
for fn in['r2_t32_b200.tsv']:
    for l in open(B+'rounds/nd/results3/'+fn):
        f=l.rstrip('\n').split('\t')
        if f[5]=='gndd:hyb=1;rmax=4' and f[6]=='256':
            R[(f[2],'GNDD')]=(float(f[7]),float(f[8]),float(f[10]))
graphs=sorted({g for g,k in R if k=='PARAMD1'},key=lambda g:N[g])
def b(n): return '<100k' if n<1e5 else '100k-1M'
keys=['MF1s','GPUMF1','GNDD','AMDs','METISs','PARAMD1','PARAMD8','PARAMD32','PARAMD64','PARMETIS1','PARMETIS8','PARMETIS32','PARMETIS64']
def gm(x):
    x=[v for v in x if v and v>0]; return math.exp(sum(map(math.log,x))/len(x)) if x else None
def f(v,p=3): return '-' if v is None else ('%.'+str(p)+'g')%v
out=[]
for idx,name,fmt in[(2,'time s',3),(0,'nnz(L) / serial MF1',3),(1,'flops / serial MF1',3)]:
    out.append(f'### per graph: {name}\n')
    out.append('| graph | n | '+' | '.join(keys)+' |'); out.append('|'+'---|'*(len(keys)+2))
    for g in graphs:
        row=[]
        for k in keys:
            v=R.get((g,k)); 
            if v is None: row.append('-');continue
            x=v[idx] if idx==2 else v[idx]/R[(g,'MF1s')][idx]
            row.append(f(x,3))
        out.append(f'| {g} | {N[g]} | '+' | '.join(row)+' |')
    out.append('')
# buckets: geomean ratios over graphs having both method and GPUMF1 (and MF1s)
out.append('### per bucket geometric means (graphs with that method present)\n')
out.append('t = time, ratio rows vs serial MF1 (nnz, flops) and time ratio vs GPU-MF1 (tG) and vs serial MF1 (tS)\n')
out.append('| bucket | method | k | geomean time s | tS (time/serial MF1) | tG (time/GPU-MF1) | nnz/MF1 | flops/MF1 | nnz/GPUMF1 | wins on nnz vs GPUMF1 | faster than GPUMF1 |'); out.append('|'+'---|'*11)
for bk in['<100k','100k-1M','all']:
    gs=[g for g in graphs if bk=='all' or b(N[g])==bk]
    for k in keys:
        gg=[g for g in gs if (g,k) in R and (g,'GPUMF1') in R and (g,'MF1s') in R]
        if not gg: continue
        t=gm([R[(g,k)][2] for g in gg])
        tS=gm([R[(g,k)][2]/R[(g,'MF1s')][2] for g in gg]); tG=gm([R[(g,k)][2]/R[(g,'GPUMF1')][2] for g in gg])
        nz=gm([R[(g,k)][0]/R[(g,'MF1s')][0] for g in gg]); fl=gm([R[(g,k)][1]/R[(g,'MF1s')][1] for g in gg])
        nzg=gm([R[(g,k)][0]/R[(g,'GPUMF1')][0] for g in gg])
        w=sum(R[(g,k)][0]<R[(g,'GPUMF1')][0] for g in gg); fa=sum(R[(g,k)][2]<R[(g,'GPUMF1')][2] for g in gg)
        out.append(f'| {bk} | {k} | {len(gg)} | {f(t)} | {f(tS)} | {f(tG)} | {f(nz)} | {f(fl)} | {f(nzg)} | {w}/{len(gg)} | {fa}/{len(gg)} |')
# scaling
out.append('\n### ParAMD / ParMETIS speedup over own 1-thread (geomean, graphs with both)\n')
out.append('| bucket | method | t8 | t32 | t64 | best-thread vs 1 |'); out.append('|---|---|---|---|---|---|')
for bk in['<100k','100k-1M']:
    gs=[g for g in graphs if b(N[g])==bk]
    for m in['PARAMD','PARMETIS']:
        r=[]
        for t in['8','32','64']:
            gg=[g for g in gs if (g,m+'1') in R and (g,m+t) in R]
            r.append(f"{gm([R[(g,m+'1')][2]/R[(g,m+t)][2] for g in gg]):.2f} (k={len(gg)})" if gg else '-')
        out.append(f'| {bk} | {m} | '+' | '.join(r)+' | |')
# quality vs thread count
out.append('\n### Quality drift with threads (geomean nnz vs own t1)\n')
out.append('| method | t8 | t32 | t64 |'); out.append('|---|---|---|---|')
for m in['PARAMD','PARMETIS']:
    r=[]
    for t in['8','32','64']:
        gg=[g for g in graphs if (g,m+'1') in R and (g,m+t) in R]
        r.append(f"{gm([R[(g,m+t)][0]/R[(g,m+'1')][0] for g in gg]):.3f} (k={len(gg)})")
    out.append(f'| {m} | '+' | '.join(r)+' |')
open(sys.argv[1],'w').write('\n'.join(out))
print('\n'.join(out[-60:]))
