include("/home/itaykadosh/Projects/GATAS/tn-work/tncorpus/load.jl"); using .TNCorpus
for name in readlines(ARGS[1])
    inst = TNCorpus.load(name; graph=false)
    nt = length(inst.hptr)-1
    occ = zeros(Int, inst.n); dup = 0; empt = 0
    for t in 1:nt
        ix = inst.hidx[inst.hptr[t]:inst.hptr[t+1]-1]
        isempty(ix) && (empt += 1)
        length(unique(ix)) < length(ix) && (dup += 1)
        for i in unique(ix); occ[i] += 1; end
    end
    for i in inst.open; occ[i] += 0; end
    h = count(>=(3), occ)
    println(rpad(name, 50), " n=", inst.n, " T=", nt-empt, " open=", length(inst.open), " maxocc=", maximum(occ), " hyper=", h, " single=", count(==(1), occ), " dupT=", dup, " dims=", sort(unique(inst.dims)))
end
