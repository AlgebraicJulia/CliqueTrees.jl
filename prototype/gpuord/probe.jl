include(joinpath(@__DIR__, "common.jl"))
warmup()
for nm in ("reg3_N3000_s0", "reg3_N10000_s0", "rqc16x16_m48_s0", "qaoaamp_N2000_p3_s0", "mc3sat_N3000")
    rows = filter(r -> r.name == nm, TNCorpus.manifest())
    isempty(rows) && (println("no $nm"); continue)
    inst = TNCorpus.load(rows[1]); g = inst.graph; w = inst.weights
    println("== $(rows[1].name) n=$(nv(g)) m=$(ne(g))")
    for (an, a) in (("bw", bw), ("nd", nd()), ("ndfast", ndfast()), ("ND3", ND{3}(bw, METISND(; seed = 0); width = 16, level = 12, scale = 100)),
                    ("gnd R1 p128", gnd(0.0; pmax = 128)), ("gnd R1 p256", gnd(0.0)), ("gnd R1 p256 fast", gnd(0.0; big = :fast)), ("gnd R3 p256", gnd(0.0; rounds = 3)))
        r = runmethod(a, w, g); s = r.stats
        @printf("  %-17s width %8.2f  time %7.3f s  [rec %.3f pack %.3f gpu %.3f replay %.3f | pieces %d rounds %d | nodes %s]\n",
            an, r.width, r.t, s.tsplit, s.tpack, s.tgpu, s.tasm, s.npieces, s.rounds, Hybrid.NODESTATS)
        flush(stdout)
    end
end
