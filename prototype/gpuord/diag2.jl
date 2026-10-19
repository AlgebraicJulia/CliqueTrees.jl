include(joinpath(@__DIR__, "common.jl"))
for (nm, (g, w)) in (("grid60", synth("grid60")), ("reg3_N500", (TNCorpus.load("reg3_N500_s0").graph, TNCorpus.load("reg3_N500_s0").weights)), ("qc53", (TNCorpus.load("qc_circuit_n53_m20_s5_e22_pABCDCDAB").graph, TNCorpus.load("qc_circuit_n53_m20_s5_e22_pABCDCDAB").weights)))
  for bdw in (false, true), uf in (30, 130, 300), pm in (128, 256)
    Hybrid.BDW[] = bdw; Hybrid.UFACTOR[] = uf
    splits, pieces = Hybrid.splitpieces([g], [w]; pmax = pm)
    porders, pw = Hybrid.gpusearch(pieces, [w]; time = 0.5)
    order = Hybrid.assemble!(Int[], splits[1], splits[1].root, porders, nothing)
    @printf("%-10s bdw=%d uf=%3d pmax=%d  max piece %.2f  final %.2f  pieces %d\n", nm, bdw, uf, pm, maximum(pw) / 1000, treewidth(w, g; alg = order), length(pieces))
  end
end
