import pandas as pd,numpy as np,glob
nm='host gpu graph n nnz method leaf nnzL flops width time'.split()
nd=pd.concat([pd.read_csv(f,sep='\t',header=None,usecols=range(11),names=nm,on_bad_lines='skip') for f in glob.glob('rounds/nd/results3/*_t32_b200.tsv')])
g=nd[(nd.method=='gndd:hyb=1;rmax=4')&(nd.leaf==256)].groupby('graph').agg(n=('n','first'),nnzL=('nnzL','median'),flops=('flops','median'),time=('time','median'))
pb=pd.read_csv('harvest_oct19b/mf/parallel_baselines.tsv',sep='\t'); pb=pb[pb.status=='ok']
mf=pd.read_csv('harvest_oct19b/mf/main_table.tsv',sep='\t')
m1=mf[mf.cfg=='GPU-MF1-u0.1'].set_index('graph'); sm=mf[mf.cfg.str.contains('serial|MF1-serial|rmain',case=False)]
print(sorted(mf.cfg.unique()))
def geo(x): return float(np.exp(np.log(x).mean()))
rows=[]
for meth in ['PARAMD','PARMETIS']:
  for th in [32,64,'best']:
    p=pb[pb.method==meth]
    p=p.loc[p.groupby('graph').time_s.idxmin()] if th=='best' else p[p.threads==th].set_index('graph').reset_index()
    p=p.set_index('graph').join(g,rsuffix='_nd',how='inner')
    for lo,hi,lab in [(0,1e6,'<1M'),(1e6,3e6,'1-3M'),(3e6,1e12,'>=3M')]:
      q=p[(p.n>=lo)&(p.n<hi)]
      if len(q)==0: rows.append((meth,th,lab,0)); continue
      rows.append((meth,th,lab,len(q),geo(q.time_s/q.time),geo(q.nnzL/q.nnzL_nd),geo(q.flops/q.flops_nd),q.time_s.median(),q.time.median()))
print(pd.DataFrame(rows,columns='meth thr bucket k t_ratio nnz_ratio flops_ratio t_par_med t_nd_med'.split()[:9]).round(3).to_string())
# GPU-MF1 vs ND
j=g.join(m1[['nnzL','flops','time_s']],rsuffix='_mf').dropna()
for lo,hi,lab in [(0,1e6,'<1M'),(1e6,3e6,'1-3M'),(3e6,1e12,'>=3M')]:
  q=j[(j.n>=lo)&(j.n<hi)]
  print('MF1u0.1/ND',lab,len(q),'time',round(geo(q.time_s/q.time),2),'nnz',round(geo(q.nnzL_mf/q.nnzL),2),'flops',round(geo(q.flops_mf/q.flops),2),'ND med t',q.time.median(),'MF1 med t',q.time_s.median())
