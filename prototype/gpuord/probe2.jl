include(joinpath(@__DIR__, "common.jl"))
ndx(a) = ND(a, METISND(; seed = 0); width = 16, level = 12, scale = 100)
insts = Any[("grid60", synth("grid60")...)]
for nm in ("reg3_N3000_s0", "rqc12x12_m32_s0", "qaoaamp_N1000_p3_s0", "rqc16x16_m48_s0")
    inst = TNCorpus.load(nm); push!(insts, (nm, inst.graph, inst.weights))
end
permutation(ones(100), Graphs.grid([10,10]); alg = ndx(MMD()))
for (nm, g, w) in insts
    print(rpad(nm, 20))
    for (an, a) in (("bw", bw), ("MMD", MMD()), ("AMF", AMF()), ("MF", MF()), ("MFavg", MF(; strategy = 1)), ("MF+MFavg", BestWidth(MF(), MF(; strategy = 1))), ("MMD+MFavg", BestWidth(MMD(), MF(; strategy = 1))))
        t = @elapsed o, _ = permutation(w, g; alg = ndx(a))
        @printf(" %s %.1f (%.2fs) |", an, treewidth(w, g; alg = o), t)
    end
    println(); flush(stdout)
end
