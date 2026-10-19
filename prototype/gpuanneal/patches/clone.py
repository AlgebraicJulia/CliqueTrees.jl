p='src/BlockAnneal.jl'; s=open(p).read()
a='''                 lam1 = lam, Wt0 = Wt, wfrac = 0.6, pheur = 0.0)'''
b='''                 lam1 = lam, Wt0 = Wt, wfrac = 0.6, pheur = 0.0, resample = 0.0, resevery = 10)'''
assert a in s; s=s.replace(a,b)
a='''        for k in (isodd(ep) ? (1:2:n-1) : (2:2:n-1))'''
b='''        if resample > 0 && ep % resevery == 0
            # population step: the worst `resample` fraction of chains (current energy) become copies of
            # chains drawn from the best fraction; they keep their temperatures
            o = sortperm(E); k = floor(Int, resample * n)
            for (d, s0) in zip(o[end-k+1:end], o[rand(rng, 1:max(k, 1), k)])
                A.L[:, :, d] .= view(A.L, :, :, s0); A.CH[:, :, d] .= view(A.CH, :, :, s0)
                A.SL[:, d] .= view(A.SL, :, s0); E[d] = E[s0]
            end
        end
        for k in (isodd(ep) ? (1:2:n-1) : (2:2:n-1))'''
assert a in s; s=s.replace(a,b)
open(p,'w').write(s)
p='scripts/pipeline.jl'; t=open(p).read()
a='pheur = gf("pheur", 0.75),'
t=t.replace(a,'pheur = gf("pheur", 0.75), resample = gf("resample", 0.0), resevery = gi("resevery", 10),')
open(p,'w').write(t)
print('ok')
