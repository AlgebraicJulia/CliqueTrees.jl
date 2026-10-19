# runpipe: one GPU annealing run (v2 block kernel) with an anytime trace and checkpoint trees.
# Included after BlockAnneal.jl (uses const BA = BlockAnneal).
#
# opts (Dict{String,String}); defaults in brackets:
#   time [30]  Wt [Inf]  beta0 [1] beta1 [300] rounds [200] mix [1] evalevery [64] nreconf [0] kmax [6]
#   unconstrained (Wt = Inf): parallel tempering over trees.
#   sliced (finite Wt): width-target annealing (cap from the seeds' width down to Wt over wfrac [0.6] of the
#     time), joint rotations + slice moves (nslice [16] per evaluation, TreeSASlicer-style proposals with
#     probability pheur [0.75]), penalty lam2 [0.3] -> lam2b [3] (geometric in time).
#   nch [0 = fill device]  nth [default_threads]  shared [auto]  rseed [1]
# Returns (best = (E, CH, slice ids, host costs), trace = [(t, best E)], info).
function runpipe(net, seeds, opts::Dict; out = nothing, ckpt = [0.3, 1, 3, 10, 30, 100, 300, 1000, 3000], log = stdout)
    gf(k, d) = haskey(opts, k) ? parse(Float64, opts[k]) : d
    gi(k, d) = round(Int, gf(k, d))
    T = gf("time", 30.0); Wt = gf("Wt", Inf)
    P = (betamin = gf("beta0", 1.0), betamax = gf("beta1", 300.0), rounds = gi("rounds", 200), mix = gf("mix", 1.0),
         evalevery = gi("evalevery", 64), nreconf = gi("nreconf", 0), kmax = gi("kmax", 6))
    shared = haskey(opts, "shared") ? opts["shared"] == "1" : nothing
    ckpt = sort(filter(<(T), ckpt))
    best = Ref((Inf, zeros(Int32, 2, 2net.nt - 1), Int[]))
    trace = Tuple{Float64,Float64}[]
    nck = Ref(1); t0 = Ref(0.0)
    function dump(tag, t)
        E, CH, sl = best[]
        (isfinite(E) && out !== nothing) || return
        h = BA.hostcost(net, CH, BA.slicemask(net, sl); Wt)
        BA.write_tree("$out.$tag.tree", net.nt, BA.ssa_from_children(net, CH), sl; meta = Dict("t" => @sprintf("%.3f", t),
            "gpu_E" => @sprintf("%.6f", E), "host_log2C" => @sprintf("%.6f", h.log2C), "host_W" => @sprintf("%.3f", h.W),
            "host_maxop" => @sprintf("%.3f", h.maxop), "nsliced" => string(length(sl)), "Wt" => string(Wt),
            "device" => CUDA.name(CUDA.device()), "opts" => join(("$k=$v" for (k, v) in opts), ';')))
    end
    function record!(A)
        BE = Array(A.BESTE); g = argmin(BE)
        if BE[g] < best[][1]
            CH, sl, E, _ = BA.best(A)
            best[] = (E, CH, sl)
        end
        t = time() - t0[]
        push!(trace, (t, best[][1]))
        while nck[] <= length(ckpt) && t >= ckpt[nck[]]
            dump("T" * replace(string(ckpt[nck[]]), r"\.0$" => ""), t); nck[] += 1
        end
    end
    t0[] = time()
    A = BA.BAnnealer(net, seeds; nch = gi("nch", 0), seed = gi("rseed", 1), shared, nth = gi("nth", BA.default_threads(net)))
    info = (nch = A.nch, nth = A.nth, shared = A.shared)
    if !isfinite(Wt)
        BA.temper!(A; time = T, P..., trace = (t, A) -> record!(A))
    else
        W0 = maximum(BA.hostcost(net, CHs, BA.slicemask(net, sl)).W for (CHs, sl) in seeds)
        BA.temper!(A; time = T - (time() - t0[]), P..., lam = gf("lam2", 0.3), lam1 = gf("lam2b", 3.0), Wt, Wt0 = W0,
                   wfrac = gf("wfrac", 0.6), nslice = gi("nslice", 16), pheur = gf("pheur", 0.75),
                   resample = gf("resample", 0.0), resevery = gi("resevery", 10), trace = (t, A) -> record!(A))
    end
    record!(A)
    ttot = time() - t0[]
    dump("final", ttot)
    E, CH, sl = best[]
    h = isfinite(E) ? BA.hostcost(net, CH, BA.slicemask(net, sl); Wt) : (log2C = Inf, W = Inf, maxop = Inf, excess = Inf, nsliced = 0)
    A = nothing; GC.gc(); CUDA.reclaim()
    (best = (E, CH, sl, h), trace, info, ttot)
end

"best value of an anytime trace at time t (Inf if nothing yet)"
traceat(trace, t) = (v = Inf; for (s, e) in trace; s <= t && (v = min(v, e)); end; v)
