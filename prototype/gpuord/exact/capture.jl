# Capture HBT's exact local triangulations (hbt_exa) with a time breakdown.
# In-process instrumentation only (methods redefined in this session; the
# CliqueTrees tree is not modified).  usage: julia capture.jl T out.jls inst...
include(joinpath(@__DIR__, "..", "common.jl"))
using CliqueTrees: compressreduce, rule_pull, hbt_safe, lowerbound, DEFAULT_LOWER_BOUND_ALGORITHM, BT_ABOVE, BT_TIMEOUT, BT_FOUND, sepenum, sepdp, HBT_RULES

const REC = Any[]
const ATOMS = Any[]
const ROUNDS = Any[]
const PH = Dict{Symbol, Float64}()   # phase times inside the current call
addph(k, t) = (PH[k] = get(PH, k, 0.0) + t)

@eval CliqueTrees function septw_run(weights::AbstractVector{W}, graph::AbstractGraph{V}, k::W, lb::W, hi::W, hord::AbstractVector, hk::W, dl::Real, nt::Integer) where {W, V}
    wmin = minimum(view(weights, oneto(nv(graph))))
    while true
        if k >= hk
            return hord, hk, hk <= hi ? BT_FOUND : BT_ABOVE
        end
        k > hi && return hord, k, BT_ABOVE
        t0 = time()
        sep, status = sepenum(weights, graph, k, dl, nt)
        push!(Main.ROUNDS, (k, status == BT_TIMEOUT ? -1 : nv(sep), time() - t0))
        Main.addph(:sepenum, time() - t0); Main.addph(:nsep, length(sep isa AbstractGraph ? (1:nv(sep)) : sep)); Main.addph(:rounds, 1.0)
        status == BT_TIMEOUT && return hord, lb, BT_TIMEOUT
        t0 = time()
        perm, _, mink, status = sepdp(weights, graph, sep, k, dl, nt)
        Main.addph(:sepdp, time() - t0)
        status == BT_TIMEOUT && return hord, lb, BT_TIMEOUT
        status == BT_FOUND && return perm, mink, BT_FOUND
        lb = k = min(mink, k + wmin)
    end
end

@eval CliqueTrees function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::HBTExact; verbose::Bool = false) where {V}
    t0 = time(); empty!(Main.ROUNDS)
    perm, invp, k, status = septw(weights, graph, alg.lo, alg.hi, alg.dl, 1)
    nvg = nv(graph)
    push!(Main.ATOMS, (; weights = collect(weights[1:nvg]), n = nvg, edges = [(Int(src(e)), Int(dst(e))) for e in edges(graph)],
        lo = alg.lo, hi = alg.hi, dlleft = alg.dl - t0, perm = collect(perm), k, status = Symbol(status), t = time() - t0, rounds = copy(Main.ROUNDS)))
    Main.addph(:septw, time() - t0); Main.addph(:atoms, 1.0); Main.addph(:atommax, 0.0)
    Main.PH[:atommax] = max(get(Main.PH, :atommax, 0.0), nv(graph))
    if status == BT_ABOVE || status == BT_TIMEOUT && alg.status[] == BT_FOUND
        alg.status[] = status
    end
    return convert(Vector{V}, perm), convert(Vector{V}, invp)
end

@eval CliqueTrees function hbt_exa(srh::NamedTuple, weights::AbstractVector{W}, H::AbstractGraph, target::W; lb::W = zero(W), cap::W = typemax(W)) where {W}
    empty!(Main.PH)
    T0 = time(); lb0 = lb
    lb = max(lb, convert(W, lowerbound(weights, H, DEFAULT_LOWER_BOUND_ALGORITHM)))
    t1 = time()
    rec = (; n = nv(H), m = ne(H), weights = collect(weights), edges = [(src(e), dst(e)) for e in edges(H)], target, lb0, lb, cap,
        cutoff = srh.cutoff[], deadline_left = srh.deadline[] - T0)
    if lb > cap
        push!(Main.REC, (; rec..., result = :missing, width = lb, order = Int[], t = time() - T0, tlb = t1 - T0, ph = Dict{Symbol, Float64}(), kn = 0))
        return missing
    end
    dl = min(srh.deadline[], time() + srh.cutoff[])
    kerweights, kergraph, result = compressreduce(HBT_RULES, weights, H, lb)
    t2 = time()
    kerorder, status = hbt_safe(kerweights, kergraph, max(lb, result.width), max(lb, result.width, target), cap, dl)
    t3 = time()
    out = if status == BT_ABOVE
        srh.nexact[] += 0; missing
    elseif status == BT_TIMEOUT
        srh.ntimeout[] += 1; nothing
    else
        srh.nexact[] += 1
        order = rule_pull(result, kerorder)
        order, _ = permutation(weights, H, MinimalChordal(order))
        order
    end
    t4 = time()
    ph = copy(Main.PH); ph[:lb] = t1 - T0; ph[:kernel] = t2 - t1; ph[:safe] = t3 - t2; ph[:minimal] = t4 - t3
    w = out isa AbstractVector ? treewidth(weights, H; alg = out) : zero(W)
    push!(Main.REC, (; rec..., result = out isa AbstractVector ? :found : out === nothing ? :timeout : :missing,
        width = w, order = out isa AbstractVector ? collect(out) : Int[], t = t4 - T0, ph, kn = nv(kergraph), kerlb = result.width))
    return out
end

T = parse(Float64, ARGS[1]); outfile = ARGS[2]
permutation(ones(100), Graphs.grid([10, 10]); alg = hbt(0.2))
empty!(REC)
allrec = Dict{String, Any}()
for nm in ARGS[3:end]
    inst = TNCorpus.load(nm)
    empty!(REC); empty!(ATOMS)
    t = @elapsed o, _ = permutation(inst.weights, inst.graph; alg = hbt(T))
    wd = treewidth(inst.weights, inst.graph; alg = o)
    recs = copy(REC); allrec[nm] = (; width = wd, t, recs, atoms = copy(ATOMS))
    tt = sum(r.t for r in recs; init = 0.0)
    nf = count(r -> r.result == :found, recs); nto = count(r -> r.result == :timeout, recs); nm_ = count(r -> r.result == :missing, recs)
    sumph(k) = sum(get(r.ph, k, 0.0) for r in recs; init = 0.0)
    @printf("%-36s n=%6d width %.1f  wall %.1fs | exact calls %d (found %d, timeout %d, above/missing %d) time %.1fs: lb %.1f kernel %.1f safe %.1f [septw %.1f: sepenum %.1f sepdp %.1f] minimal %.1f | local nv med %d max %d, kernel nv med %d max %d\n",
        nm, nv(inst.graph), wd, t, length(recs), nf, nto, nm_, tt, sumph(:lb), sumph(:kernel), sumph(:safe), sumph(:septw), sumph(:sepenum), sumph(:sepdp), sumph(:minimal),
        isempty(recs) ? 0 : sort([r.n for r in recs])[cld(end, 2)], maximum(r -> r.n, recs; init = 0),
        isempty(recs) ? 0 : sort([r.kn for r in recs])[cld(end, 2)], maximum(r -> r.kn, recs; init = 0))
    flush(stdout)
    serialize(outfile, allrec)
end
