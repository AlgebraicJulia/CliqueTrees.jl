using Statistics, Printf
rows = [split(l, '\t') for l in eachline("results.tsv")]
D = Dict{Tuple{String,String},NamedTuple}()
gnames = String[]; meths = ["AMD","AMF","MMD","METIS-ND","MF","MF1","GPU-MF","GPU-MD","ParAMD-t1","ParAMD-t8"]
nn = Dict{String,Int}(); mm = Dict{String,Int}()
for r in rows
    (r[1] in gnames) || push!(gnames, r[1])
    nn[r[1]] = parse(Int, r[2]); mm[r[1]] = parse(Int, r[3])
    D[(r[1], r[4])] = (nnz = parse(Float64, r[5]), fl = parse(Float64, r[6]), tw = parse(Float64, r[7]), t = parse(Float64, r[8]))
end
f(x) = @sprintf("%.3g", x)
io = open("table.md", "w")
println(io, "| graph | n | order | nnz(L) | flops | tw | time (s) | nnz/AMD | flops/AMD |\n|---|---|---|---|---|---|---|---|---|")
for g in gnames, m in meths
    haskey(D, (g, m)) || continue
    d = D[(g, m)]; a = get(D, (g, "AMD"), nothing)
    println(io, "| $g | $(nn[g]) | $m | $(Int(d.nnz)) | $(f(d.fl)) | $(Int(d.tw)) | $(f(d.t)) | ", a === nothing ? "-" : f(d.nnz / a.nnz), " | ", a === nothing ? "-" : f(d.fl / a.fl), " |")
end
close(io)
function summary(set, label)
    println("\n#### $label (", length(set), " graphs)\n")
    println("| ordering | #graphs | nnz ratio median | nnz min..max | flops ratio median | flops min..max | wins nnz | wins flops | tw ratio median |\n|---|---|---|---|---|---|---|---|---|")
    for m in meths
        m == "AMD" && continue
        gs = [g for g in set if haskey(D, (g, m)) && haskey(D, (g, "AMD"))]
        isempty(gs) && continue
        rn = [D[(g, m)].nnz / D[(g, "AMD")].nnz for g in gs]; rf = [D[(g, m)].fl / D[(g, "AMD")].fl for g in gs]
        rt = [(D[(g, m)].tw + 1) / (D[(g, "AMD")].tw + 1) for g in gs]
        println("| $m | $(length(gs)) | $(f(median(rn))) | $(f(minimum(rn)))..$(f(maximum(rn))) | $(f(median(rf))) | $(f(minimum(rf)))..$(f(maximum(rf))) | $(count(<(1), rn))/$(length(gs)) | $(count(<(1), rf))/$(length(gs)) | $(f(median(rt))) |")
    end
end
ss = filter(g -> isfile("data/$g.mtx"), gnames); tn = setdiff(gnames, ss)
summary(ss, "SuiteSparse"); summary(tn, "Tensor-network graphs"); summary(gnames, "All")
# time table
println("\n#### Time ratios (median over graphs)\n")
println("| graph set | AMD t (median s) | MF serial / AMD | GPU-MF / AMD | GPU-MF / serial MF | GPU-MD / AMD | #graphs with GPU-MF faster than serial MF |\n|---|---|---|---|---|---|---|")
for (set, lab) in ((ss, "SuiteSparse"), (tn, "TN"), (gnames, "all"))
    gs = [g for g in set if all(haskey(D, (g, m)) for m in ("AMD", "MF", "GPU-MF", "GPU-MD"))]
    r(a, b) = f(median([D[(g, a)].t / D[(g, b)].t for g in gs]))
    println("| $lab | $(f(median([D[(g,"AMD")].t for g in gs]))) | $(r("MF","AMD")) | $(r("GPU-MF","AMD")) | $(r("GPU-MF","MF")) | $(r("GPU-MD","AMD")) | $(count(g -> D[(g,"GPU-MF")].t < D[(g,"MF")].t, gs))/$(length(gs)) |")
end
# GPU-MF vs MF fill
println("\n#### GPU-MF vs serial MF fill (ratio GPU-MF/MF)\n")
gs = [g for g in gnames if haskey(D, (g, "MF")) && haskey(D, (g, "GPU-MF"))]
rn = [D[(g, "GPU-MF")].nnz / D[(g, "MF")].nnz for g in gs]; rf = [D[(g, "GPU-MF")].fl / D[(g, "MF")].fl for g in gs]
println("nnz(L): median $(f(median(rn))), range $(f(minimum(rn)))..$(f(maximum(rn))), GPU-MF <= MF on $(count(<=(1), rn))/$(length(gs)); flops: median $(f(median(rf))), range $(f(minimum(rf)))..$(f(maximum(rf))), <= on $(count(<=(1), rf))/$(length(gs))")
