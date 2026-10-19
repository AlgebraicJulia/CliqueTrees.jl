include(joinpath(@__DIR__, "common.jl"))
warmup()
insts = Any[("grid60", synth("grid60")...)]
for nm in ("rqc7x7_m16_s0", "reg3_N500_s0", "qc_circuit_n53_m20_s5_e22_pABCDCDAB")
    inst = TNCorpus.load(nm); push!(insts, (nm, inst.graph, inst.weights))
end
for (nm, g, w) in insts
    println("== $nm  n=$(nv(g)) m=$(ne(g))")
    for (an, a) in (("bw", bw), ("nd", nd()), ("gpu1 p256", gpu(1.0)), ("gpu1 p512", gpu(1.0; pmax = 512)), ("gpu1 p128", gpu(1.0; pmax = 128)), ("safe+gpu1", safe(gpu(1.0))), ("hbt1", hbt(1.0)))
        r = runmethod(a, w, g)
        s = r.stats
        @printf("  %-10s width %8.3f  time %7.3f s  [split %.3f pack %.3f gpu %.3f asm %.3f | pieces %d  rounds %d  scored %d]\n",
            an, r.width, r.t, s.tsplit, s.tpack, s.tgpu, s.tasm, s.npieces, s.rounds, s.nscored)
        flush(stdout)
    end
end
