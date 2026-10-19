include(joinpath(@__DIR__, "common.jl"))
warmup()
insts = Any[("grid60", synth("grid60")...)]
for nm in ("rqc7x7_m16_s0", "reg3_N500_s0", "qc_circuit_n53_m20_s5_e22_pABCDCDAB")
    inst = TNCorpus.load(nm); push!(insts, (nm, inst.graph, inst.weights))
end
for (nm, g, w) in insts
    println("== $nm  n=$(nv(g)) m=$(ne(g))")
    for (an, a) in (("bw", bw), ("nd", nd()), ("gnd0.5 p256 w16", gnd(0.5)), ("gnd0.5 p512 w16", gnd(0.5; pmax = 512)), ("gnd0.5 p256 w64", gnd(0.5; width = 64)), ("gnd0.5 p256 w256", gnd(0.5; width = 256)), ("hbt1", hbt(1.0)))
        r = runmethod(a, w, g)
        s = r.stats
        @printf("  %-17s width %8.3f  time %7.3f s  [rec %.3f pack %.3f gpu %.3f replay %.3f | pieces %d rounds %d scored %d | sepcache %s nodes %s]\n",
            an, r.width, r.t, s.tsplit, s.tpack, s.tgpu, s.tasm, s.npieces, s.rounds, s.nscored, Hybrid.SEPSTATS, Hybrid.NODESTATS)
        flush(stdout)
    end
end
