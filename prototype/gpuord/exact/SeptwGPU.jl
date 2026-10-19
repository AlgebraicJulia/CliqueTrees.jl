# septw (CliqueTrees bt/septw.jl: bt_tw + septw_run) with every round's
# sepenum + sepdp on the GPU (EnumW, DPW); same rounds, same results.
# The deadline is checked between the GPU phases (a kernel is not
# interrupted).
using CUDA, CliqueTrees, Graphs
isdefined(Main, :EnumW) || include(joinpath(@__DIR__, "EnumW.jl"))
isdefined(Main, :DPW) || include(joinpath(@__DIR__, "DPW.jl"))
using .EnumW, .DPW
using CliqueTrees: BT_FOUND, BT_ABOVE, BT_TIMEOUT, bt_int, bt_bas

const GPUSTATS = Dict{Symbol, Float64}(:calls => 0, :rounds => 0, :tgpu => 0.0)

gbucket(n) = n <= 64 ? 1 : n <= 128 ? 2 : n <= 256 ? 4 : 8

function gpu_round(weights, graph, k, dl = Inf)
    n = nv(graph); W = gbucket(n)
    TE = W == 1 ? UInt64 : EnumW.Bits{W}; TD = W == 1 ? UInt64 : DPW.Bits{W}
    bt = EnumW.prepare(TE, [graph], [collect(Float64, weights[1:n])], [k])
    lists, ovf, _, info = EnumW.gpu_sepenum(bt; split = 3, dl = Float64(dl))
    info.aborted && return :timeout
    ovf[1] && return nothing
    time() > dl && return :timeout
    sep = EnumW.sepgraph(lists[1], n)
    res, info = DPW.gpu_sepdp(TD, [collect(Float64, weights[1:n])], [graph], [sep], [k]; dl = Float64(dl))
    info.aborted && return :timeout
    info.novf > 0 && return nothing
    return res[1]   # perm, invp, mink, :found/:above
end

function septw_gpu(weights::AbstractVector{W}, graph00::AbstractGraph{V}, lo, hi, dl::Real) where {W, V}
    GPUSTATS[:calls] += 1
    graph = bt_int(graph00); nvtx = nv(graph)
    lo = convert(W, lo); hi = convert(W, hi)
    base = bt_bas(weights, graph)
    if base > hi
        order = collect(1:nvtx); k = base; status = BT_ABOVE
    else
        k = max(base, min(lo, hi))
        hord, _ = permutation(weights, graph, MF())
        hk = treewidth(weights, graph, hord)
        if hk <= k
            order = hord; k = hk; status = BT_FOUND
        else
            order, k, status = septw_run_gpu(weights, graph, k, base, hi, hord, hk, dl)
        end
    end
    perm = Vector{V}(undef, nvtx); invp = Vector{V}(undef, nvtx)
    if status == BT_FOUND
        for i in 1:nvtx
            perm[i] = order[i]; invp[order[i]] = i
        end
    else
        for v in 1:nvtx
            perm[v] = invp[v] = v
        end
    end
    return perm, invp, k, status
end

function septw_run_gpu(weights::AbstractVector{W}, graph, k::W, lb::W, hi::W, hord, hk::W, dl::Real) where {W}
    wmin = minimum(view(weights, 1:nv(graph)))
    while true
        k >= hk && return hord, hk, hk <= hi ? BT_FOUND : BT_ABOVE
        k > hi && return hord, k, BT_ABOVE
        time() > dl && return hord, lb, BT_TIMEOUT
        GPUSTATS[:rounds] += 1
        t0 = time()
        r = gpu_round(weights, graph, k, dl)
        GPUSTATS[:tgpu] += time() - t0
        (r === nothing || r === :timeout) && return hord, lb, BT_TIMEOUT   # deadline (or GPU stack overflow)
        perm, _, mink, st = r
        time() > dl && return hord, lb, BT_TIMEOUT
        st == :found && return perm, convert(W, mink), BT_FOUND
        lb = k = min(convert(W, mink), k + wmin)
    end
end

# HBT hook (in this process only): atoms of HBTExact with at least `nmin`
# vertices go to the GPU, the others stay on the CPU.
const GPU_NMIN = Ref(30)
function hook_hbt!()
    @eval CliqueTrees function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::HBTExact; verbose::Bool = false) where {V}
        n = nv(graph)
        perm, invp, _, status = if Main.GPU_NMIN[] <= n <= 512
            Main.septw_gpu(weights, graph, alg.lo, alg.hi, alg.dl)
        else
            septw(weights, graph, alg.lo, alg.hi, alg.dl, 1)
        end
        if status == BT_ABOVE || status == BT_TIMEOUT && alg.status[] == BT_FOUND
            alg.status[] = status
        end
        return convert(Vector{V}, perm), convert(Vector{V}, invp)
    end
end

# Escalation: CPU septw first, for delta seconds; calls still running go to
# the GPU. delta comes from the measured GPU overhead of one round
# (autotuned on the device at hand: no per-GPU constants).
const DELTA = Ref(0.0)
const ESC = Dict{Symbol, Int}(:calls => 0, :cpu => 0, :gpu => 0, :gpu_found => 0, :gpu_timeout => 0)
function calibrate!()
    g = BipartiteGraph(Graphs.grid([5, 6])); w = ones(nv(g))
    k = 5.0
    gpu_round(w, g, k)                     # compile
    t = minimum(@elapsed(gpu_round(w, g, k)) for _ in 1:5)
    DELTA[] = 2t
    return DELTA[]
end
function hook_hbt_escalate!()
    @eval CliqueTrees function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::HBTExact; verbose::Bool = false) where {V}
        n = nv(graph); E = Main.ESC; E[:calls] += 1
        if n <= 512
            perm, invp, k, status = septw(weights, graph, alg.lo, alg.hi, min(alg.dl, time() + Main.DELTA[]), 1)
            if status == BT_TIMEOUT && time() < alg.dl
                E[:gpu] += 1
                perm, invp, k, status = Main.septw_gpu(weights, graph, alg.lo, alg.hi, alg.dl)
                status == BT_FOUND && (E[:gpu_found] += 1); status == BT_TIMEOUT && (E[:gpu_timeout] += 1)
            else
                E[:cpu] += 1
            end
        else
            perm, invp, k, status = septw(weights, graph, alg.lo, alg.hi, alg.dl, 1)
        end
        if status == BT_ABOVE || status == BT_TIMEOUT && alg.status[] == BT_FOUND
            alg.status[] = status
        end
        return convert(Vector{V}, perm), convert(Vector{V}, invp)
    end
end

# CPU-only counting hook (stock behaviour; counts calls and outcomes)
const CPUC = Dict{Symbol, Int}(:calls => 0, :found => 0, :timeout => 0)
function hook_hbt_count!()
    @eval CliqueTrees function permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::HBTExact; verbose::Bool = false) where {V}
        perm, invp, _, status = septw(weights, graph, alg.lo, alg.hi, alg.dl, 1)
        C = Main.CPUC; C[:calls] += 1; status == BT_FOUND && (C[:found] += 1); status == BT_TIMEOUT && (C[:timeout] += 1)
        if status == BT_ABOVE || status == BT_TIMEOUT && alg.status[] == BT_FOUND
            alg.status[] = status
        end
        return convert(Vector{V}, perm), convert(Vector{V}, invp)
    end
end
