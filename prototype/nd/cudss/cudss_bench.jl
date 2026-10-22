# End-to-end sparse Cholesky with NVIDIA cuDSS (CUDSS.jl 0.8.1, cuDSS 0.8.0): does a GPU fill-reducing ordering
# (GND hybrid ND, GPU-MF1) make the whole solve faster than cuDSS's own reordering?  (NOTES.md)
# usage: julia --project=ENV -tT cudss_bench.jl
# env:
#   SRC      dir with GND.jl hybfm.jl kway.jl ndutil.jl Rounds.jl GRounds.jl GSmall.jl gens.jl (flat copy)
#            (ndtree.jl is taken from this script's directory)
#   DATA     ':'-separated dirs with NAME.bin (pattern: n, I, J); lat2_L / lat3_L are generated
#   NAMES    comma list;  OUT tsv (appended);  REPS (3) timed cuDSS runs per config (median reported)
#   CFGS     comma list of
#     cudss    cuDSS default reordering (CUDSS_REORDERING_ALG_DEFAULT = METIS-based ND), host_nthreads = HOSTT
#     cudss1   same, host_nthreads = 1
#     cdamd    cuDSS reordering algo3 (its AMD)
#     own      user_perm = the permutation cuDSS's default reordering produced (control: is the user_perm path
#              as fast as the internal one for the same order?); needs cudss earlier in CFGS
#     X+t      (X = own, gnd, gmf1, amd, metis, mf1) user_perm = X postordered + the ND partition tree derived from
#              its elimination tree (ndtree.jl; tree time counted in the ordering time). Without the tree cuDSS
#              factorizes a user_perm ~10x slower (laptop lat2_200: 48 ms vs 5 ms for its own order).
#     gnd      user_perm = GPU ND hybrid gndd:256:hyb=1;rmax=4 (GND.jl + device MF1 leaves), ordered here
#     gmf1     user_perm = GPU-MF1 on the whole graph (GRounds, mf1, relax 0.1), ordered here
#     amd / metis / mf1   user_perm = precomputed CPU orders PERMDIR/NAME.ALG.perm (perms.jl; time from that file)
#   HOSTT    cuDSS host threads (default Threads.nthreads()); MTLAYER=0 disables the threading layer
#   MEMCAP   bytes: skip a config when 8 * nnz(L) (ours, or cuDSS lu_nnz after analysis) exceeds it (default 0.8 x GPU mem)
#   NOTREEMAX  user_perm configs without "+t" are skipped above this n (default 2M; they only show the slow path)
#   OREPS    timed ordering runs for gnd/gmf1 (default 3, median), after one untimed run
# matrix: the graph's pattern with SPD values (graph Laplacian + I: off-diagonal -1, diagonal deg + 1), lower
# triangle CSR (Int32, 1-based) on the device, 'SPD' / 'L' -> cuDSS LL^T. b = A * ones.
# rows: host gpu name n nnzA cfg rep tord tre tsym tfact tsolve total lu_nnz cudss_flops our_nnz our_flops resid info hostt
using CUDA, CUDSS, CliqueTrees, Graphs, SparseArrays, Serialization, Statistics, LinearAlgebra, Printf
const SRC = ENV["SRC"]
include(joinpath(SRC, "gens.jl"))
include(joinpath(SRC, "Rounds.jl")); using .Rounds
include(joinpath(SRC, "GRounds.jl")); using .GRounds
include(joinpath(SRC, "ndutil.jl"))
include(joinpath(SRC, "GND.jl")); using .GND
include(joinpath(@__DIR__, "ndtree.jl"))

const host = gethostname()
const gpuname = CUDA.name(CUDA.device())
const HOSTT = parse(Int, get(ENV, "HOSTT", string(Threads.nthreads())))
const MEMCAP = parse(Float64, get(ENV, "MEMCAP", string(0.8 * CUDA.totalmem(CUDA.device()))))
const PERMDIR = get(ENV, "PERMDIR", "")
const NOTREEMAX = parse(Int, get(ENV, "NOTREEMAX", "2000000"))   # user_perm configs without the tree (slow path) only below this n
const ALG = RoundsAlg(; score = :mf1, relax = 0.1)

# ---------------------------------------------------------------- orderings (elimination orders, 1-based)
function gpu_leafdev(tm)
    return (uptr, uadj, loffH) -> begin
        bt = Dict{Symbol, Any}()
        dord, info = GRounds.grounds_batch_dev(uptr, uadj, loffH, ALG; timing = bt)
        dord
    end
end
const GNDO = GNDOpts(; Lb = 256, hyb = 1, rmax = 4)
order_gnd(ptr, adj) = gnd_perm(ptr, adj; opts = GNDO, leafdev = gpu_leafdev(Dict{Symbol, Any}()), timing = Dict{Symbol, Any}())
order_gmf1(ptr, adj) = GRounds.grounds_csr(ones(length(ptr) - 1), ptr, adj, ALG)[1]

# ---------------------------------------------------------------- matrix
# lower triangle CSR (diagonal last in each row) of L(G) + I, Int32
function spd_lower(ptr, adj)
    n = length(ptr) - 1
    rp = Vector{Int32}(undef, n + 1); rp[1] = 1
    @inbounds for i in 1:n
        c = 1
        for q in ptr[i]:(ptr[i + 1] - 1)
            adj[q] < i && (c += 1)
        end
        rp[i + 1] = rp[i] + c
    end
    nz = rp[n + 1] - 1
    cv = Vector{Int32}(undef, nz); vv = Vector{Float64}(undef, nz)
    @inbounds for i in 1:n
        k = rp[i]
        for q in ptr[i]:(ptr[i + 1] - 1)     # adj rows are sorted
            j = adj[q]
            if j < i
                cv[k] = j; vv[k] = -1.0; k += 1
            end
        end
        cv[k] = i; vv[k] = (ptr[i + 1] - ptr[i]) + 1.0
    end
    return rp, cv, vv
end
# full symmetric matrix on the host (residual check)
function spd_full(ptr, adj)
    n = length(ptr) - 1
    cp = Vector{Int}(undef, n + 1); cp[1] = 1
    for i in 1:n; cp[i + 1] = cp[i] + (ptr[i + 1] - ptr[i]) + 1; end
    rv = Vector{Int}(undef, cp[n + 1] - 1); nzv = Vector{Float64}(undef, cp[n + 1] - 1)
    @inbounds for i in 1:n
        k = cp[i]; placed = false
        for q in ptr[i]:(ptr[i + 1] - 1)
            j = adj[q]
            if !placed && j > i
                rv[k] = i; nzv[k] = (ptr[i + 1] - ptr[i]) + 1.0; k += 1; placed = true
            end
            rv[k] = j; nzv[k] = -1.0; k += 1
        end
        placed || (rv[k] = i; nzv[k] = (ptr[i + 1] - ptr[i]) + 1.0)
    end
    return SparseMatrixCSC(n, n, cp, rv, nzv)
end

# ---------------------------------------------------------------- cuDSS
const MTLIB = joinpath(dirname(CUDSS.libcudss), "libcudss_mtlayer_gomp.so")
const MTON = Ref(false)
function mt_init!()
    if get(ENV, "MTLAYER", "1") == "1" && isfile(MTLIB)
        CUDSS.cudssSetThreadingLayer(CUDSS.handle(), MTLIB)
        MTON[] = true
    end
    println("cuDSS ", CUDSS.version(), " threading layer: ", MTON[] ? MTLIB : "off", " host_nthreads=", HOSTT); flush(stdout)
end
function cudss_flops(s)
    r = Ref{Int64}(-1)          # cuDSS 0.8: CUDSS_DATA_FLOPS is a 64-bit integer (checked on the laptop)
    try
        CUDSS.cudssDataGet(s.data.handle, s.data, CUDSS.CUDSS_DATA_FLOPS, r, 8, s.nbytes_written)
    catch
    end
    return r[] < 0 ? NaN : Float64(r[])
end
# cuDSS's reordering permutation (as returned by perm_reorder_row), host Vector{Int32}
function cudss_perm(s, n)
    p = Vector{Int32}(undef, n)
    cudss_set(s, "perm_reorder_row", p)
    cudss_get(s, "perm_reorder_row")
    return p
end
# one solve: (times, lu_nnz, flops, x, info, solver perm or nothing)
function cudss_once(rp, cv, vv, n, b; uperm = nothing, tree = nothing, reorder = "default", nthreads = HOSTT, getperm = false, nnzcap = Inf)
    s = CudssSolver(rp, cv, vv, "SPD", 'L')
    MTON[] && cudss_set(s, "host_nthreads", nthreads)
    reorder == "default" || cudss_set(s, "reordering_alg", reorder)
    x = CUDA.zeros(Float64, n)
    CUDA.synchronize()
    t0 = time()
    uperm === nothing || cudss_set(s, "user_perm", uperm)
    tree === nothing || CUDSS.cudssDataSet(s.data.handle, s.data, CUDSS.CUDSS_DATA_USER_ND_PARTITION_TREE, tree, sizeof(tree))
    cudss("reordering", s, x, b); CUDA.synchronize()
    t1 = time()
    cudss("symbolic_factorization", s, x, b); CUDA.synchronize()
    t2 = time()
    lnz = cudss_get(s, "lu_nnz")
    fl = cudss_flops(s)
    p = getperm ? cudss_perm(s, n) : nothing
    if 8.0 * lnz > nnzcap
        finalize(s.data); finalize(s.matrix)
        return (t1 - t0, t2 - t1, NaN, NaN), lnz, fl, nothing, -1, p
    end
    cudss("factorization", s, x, b); CUDA.synchronize()
    t3 = time()
    cudss("solve", s, x, b); CUDA.synchronize()
    t4 = time()
    info = cudss_get(s, "info")
    isfinite(fl) || (fl = cudss_flops(s))
    xh = Array(x)
    finalize(s.data); finalize(s.matrix); finalize(s.config)
    return (t1 - t0, t2 - t1, t3 - t2, t4 - t3), lnz, fl, xh, info, p
end

# ---------------------------------------------------------------- self-check
# small matrix: user_perm = elimination order (1-based, as perm_reorder_row returns it); cuDSS's own order fed back must
# reproduce its lu_nnz; with the derived partition tree (ndtree.jl) the residual must stay at machine precision
function selfcheck()
    for (ptr, adj) in (graph_csr(lat2(30)), graph_csr(lat3(10)))
        n = length(ptr) - 1
        rp, cv, vv = spd_lower(ptr, adj)
        d = (CuVector(rp), CuVector(cv), CuVector(vv))
        A = spd_full(ptr, adj); bh = A * ones(n); b = CuVector(bh)
        g = csr_graph(ptr, adj)
        _, l0, f0, _, _, p0 = cudss_once(d..., n, b; getperm = true)
        println("selfcheck n=$n: cuDSS default lu_nnz=$l0 flops=$f0 perm extrema=$(extrema(p0)) fillq(perm)=$(fillq(g, Int.(p0))[1]) fillq(invperm)=$(fillq(g, invperm(Int.(p0)))[1])")
        o = order_gnd(ptr, adj)
        for (lab, oo) in (("own", Int.(p0)), ("gnd", Int.(o)))
            no, tr = nd_tree(ptr, adj, oo)
            for (lab2, u, t) in (("perm", Int32.(oo), nothing), ("perm+tree", Int32.(no), tr))
                _, l, f, xh, info, _ = cudss_once(d..., n, b; uperm = u, tree = t)
                r = norm(A * xh - bh) / norm(bh)
                println("  $lab $lab2: lu_nnz=$l flops=$f info=$info resid=$r fillq=$(fillq(g, Int.(u))[1]) zeros in tree=$(t === nothing ? 0 : count(iszero, t))")
                (r < 1e-10 && info == 0) || error("selfcheck failed")
            end
        end
    end
    flush(stdout)
end

# ---------------------------------------------------------------- main
function loadperm(name, alg)
    f = joinpath(PERMDIR, "$name.$alg.perm")
    isfile(f) || return nothing
    return deserialize(f)       # (perm::Vector{Int32} elimination order 1-based, time::Float64, host::String, tag::String)
end
fmt(x) = x isa AbstractFloat ? (isfinite(x) ? @sprintf("%.6g", x) : "NaN") : string(x)

function main()
    get(ENV, "MTLATE", "1") == "1" || mt_init!()
    # warm-up: every path on small graphs
    let (ptr, adj) = graph_csr(lat2(40))
        for h in (1,); order_gnd(ptr, adj); end
        gnd_perm(ptr, adj; opts = GNDOpts(; Lb = 64, hyb = 1, rmax = 4), leafdev = gpu_leafdev(Dict{Symbol, Any}()))
        order_gmf1(ptr, adj)
    end
    tcal = @elapsed GND.hyb_calibrate!()
    println("threads=", Threads.nthreads(), " GND calibration ", round(tcal, digits = 2), " s"); flush(stdout)
    get(ENV, "MTLATE", "1") == "1" && mt_init!()
    selfcheck()
    get(ENV, "SELFCHECK", "") == "only" && exit(0)

    out = open(ENV["OUT"], "a")
    cfgs = [String(c) for c in split(get(ENV, "CFGS", "cudss,own,gnd,gmf1,amd,metis,mf1"), ",") if !isempty(c)]
    reps = parse(Int, get(ENV, "REPS", "3")); oreps = parse(Int, get(ENV, "OREPS", "3"))
    for name in split(get(ENV, "NAMES", ""), ",")
        isempty(name) && continue
        local ptr, adj
        tl = @elapsed (ptr, adj) = try load_csr(String(name)) catch err; println("LOAD FAILED $name: $err"); continue end
        n = length(ptr) - 1
        rp, cv, vv = spd_lower(ptr, adj)
        nnzA = 2 * (length(rp) > 0 ? rp[end] - 1 - n : 0) + n
        A = spd_full(ptr, adj); bh = A * ones(n)
        d = (CuVector(rp), CuVector(cv), CuVector(vv)); b = CuVector(bh)
        g = csr_graph(ptr, adj)
        println(name, " n=", n, " nnz(A)=", nnzA, " load ", round(tl, digits = 1), " s"); flush(stdout)
        ownperm = nothing; ocache = Dict{String, Any}()
        for cfg in cfgs
            try
                # ---- ordering ("X+t": user_perm = postordered X + derived ND partition tree, ndtree.jl)
                usetree = endswith(cfg, "+t"); base = usetree ? cfg[1:end - 2] : cfg
                if !usetree && !(base in ("cudss", "cudss1", "cdamd")) && n > NOTREEMAX
                    println("  $cfg: user_perm without tree skipped above n = $NOTREEMAX"); continue
                end
                o = nothing; tord = 0.0; ttag = ""; ttree = 0.0; tr = nothing
                if base in ("gnd", "gmf1") && haskey(ocache, base)
                    o, tord, ttag = ocache[base]                  # same order for X and X+t
                elseif base in ("gnd", "gmf1")
                    f = base == "gnd" ? order_gnd : order_gmf1
                    f(ptr, adj)                                   # untimed (pools, clocks)
                    ts = Float64[]
                    for _ in 1:oreps
                        GC.gc(false); CUDA.reclaim()
                        t = @elapsed (o = f(ptr, adj)); push!(ts, t)
                    end
                    tord = median(ts); ttag = "b200:" * join(fmt.(ts), "/")
                    ocache[base] = (o, tord, ttag)
                elseif base in ("amd", "metis", "mf1")
                    pf = loadperm(name, uppercase(base))
                    pf === nothing && (println("  $cfg: no perm file, skipped"); continue)
                    o, tord = pf[1], pf[2]; ttag = pf[3] * ":" * pf[4]
                elseif base == "own"
                    ownperm === nothing && (println("  own: no cuDSS perm, skipped"); continue)
                    o = ownperm; tord = 0.0; ttag = "cudss"
                end
                if usetree && o !== nothing
                    nd_tree(ptr, adj, Int.(o))                 # untimed (compile)
                    ttree = @elapsed ((o2, tr) = nd_tree(ptr, adj, Int.(o)))
                    o = o2; tord += ttree; ttag *= ";tree=" * fmt(ttree)
                end
                oq = o === nothing ? (-1, NaN, -1) : fillq(g, Int.(o))
                if o !== nothing && 8.0 * oq[1] > MEMCAP
                    println(rpad(name, 14), rpad(cfg, 7), " nnz(L)=", oq[1], " exceeds MEMCAP: skipped")
                    println(out, join((host, gpuname, name, n, nnzA, cfg, 0, fmt(tord), "", "", "", "", "", "", "", oq[1], fmt(oq[2]), "", "skip_mem", HOSTT, ttag), '\t')); flush(out)
                    continue
                end
                uperm = o === nothing ? nothing : Int32.(o)
                reorder = cfg == "cdamd" ? "algo3" : "default"
                nth = cfg == "cudss1" ? 1 : HOSTT
                rows = []
                for rep in 1:reps
                    GC.gc(); CUDA.reclaim()
                    tt, lnz, fl, xh, info, p = cudss_once(d..., n, b; uperm = uperm === nothing ? nothing : copy(uperm), tree = tr === nothing ? nothing : copy(tr), reorder, nthreads = nth,
                        getperm = (cfg == "cudss" && rep == 1), nnzcap = MEMCAP)
                    if p !== nothing
                        ownperm = p
                        oq = fillq(g, Int.(p))
                    end
                    r = xh === nothing ? NaN : norm(A * xh - bh) / norm(bh)
                    tot = tord + sum(tt)
                    st = xh === nothing ? "skip_lu_nnz" : (info == 0 ? "ok" : "info$info")
                    println(out, join((host, gpuname, name, n, nnzA, cfg, rep, fmt(tord), fmt.(tt)..., fmt(tot), lnz, fmt(fl), oq[1], fmt(oq[2]), fmt(r), st, nth, ttag), '\t')); flush(out)
                    @printf("%-14s %-7s rep%d ord=%.3f re=%.3f sym=%.3f fact=%.3f solve=%.4f total=%.3f lu_nnz=%.4g flops=%.4g our_nnz=%.4g resid=%.2g %s\n",
                        name, cfg, rep, tord, tt..., tot, lnz, fl, oq[1], r, st); flush(stdout)
                    xh === nothing && break
                end
            catch err
                println("FAILED $name $cfg: ", first(sprint(showerror, err, catch_backtrace()), 2000)); flush(stdout)
            end
        end
        d = nothing; b = nothing; A = nothing; g = nothing; GC.gc(); CUDA.reclaim()
    end
    close(out)
    println("DONE")
end
main()
