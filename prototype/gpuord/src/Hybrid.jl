module Hybrid

# CPU+GPU hybrid ordering. The CPU regularizes: filled nested dissection
# (METIS vertex separators, as in CliqueTrees' ND: children G[A∪S] + K(S))
# splits every graph into pieces of at most `pmax` vertices; pieces are
# bucketed by size and padded. The GPU then runs a batched search on all
# pieces at once: randomized greedy orderings, then local search focused on
# the pieces that set the width. Orders compose as (A, B, S) up the tree.

using CUDA, Graphs, Random, Metis, Printf
using CliqueTrees
using CliqueTrees: EliminationAlgorithm
using ..GPUOrd
using ..GPUOrd: bucketwords

export GPUSearch, hybrid, splitpieces, Stats, STATS, gpusearch

const Q = 1000          # exact GPU weights are round(Q * w): widths in 1/Q bits
const BDW = Ref(true)  # boundary vertices weigh 1 in METIS
const UFACTOR = Ref(200)
const SQ = 4            # greedy score weights round(SQ * w)

mutable struct Stats
    tsplit::Float64     # CPU: nested dissection into pieces
    tpack::Float64      # CPU: bucketing, padding, upload
    tgpu::Float64       # GPU search (wall, synchronized)
    tasm::Float64       # CPU: download and compose
    npieces::Int
    work::Float64       # sum over pieces of n^3/64^2 (padded word-ops per ordering)
    nscored::Int        # orderings built or scored on the GPU
    rounds::Int
    pieces_n::Vector{Int}
end
Stats() = Stats(0, 0, 0, 0, 0, 0, 0, 0, Int[])
const STATS = Ref(Stats())

# ------------------------------------------------------------ the pieces --

struct Piece
    verts::Vector{Int}          # graph-local vertex ids; the first nint are interior
    nint::Int
    g::SimpleGraph{Int}         # on 1:length(verts), in that order
    graph::Int                  # which graph of the batch (pieces of a graph share a width)
    ew::Vector{Int32}           # exact weights (widths are sums of them)
    sw::Vector{Int32}           # greedy score weights
end
Piece(verts, nint, g, graph) = Piece(verts, nint, g, graph, Int32[], Int32[])

mutable struct Node
    piece::Int                  # > 0: a GPU piece; < 0: a CPU leaf (oversize, ordered on the CPU)
    a::Int
    b::Int
    sep::Vector{Int}            # graph-local ids of S \ boundary, in elimination order
end

struct Split
    nodes::Vector{Node}
    root::Int
end

function tosimple(g0)
    g0 isa SimpleGraph{Int} && return g0
    g = SimpleGraph{Int}(Int(nv(g0)))
    for v in 1:nv(g0), u in neighbors(g0, v)
        u > v && add_edge!(g, Int(v), Int(u))
    end
    return g
end

metisweights(w::AbstractVector{Float64}) = Metis.idx_t[max(1, trunc(Int, 100w[v])) for v in eachindex(w)]

# vertex separator of g (weights iw): part 0 = A, 1 = B, 2 = S
function metissep(g::SimpleGraph, iw::Vector{Metis.idx_t}, seed::Int)
    n = nv(g)
    xadj = Vector{Metis.idx_t}(undef, n + 1); adjncy = Metis.idx_t[]
    xadj[1] = 0
    for v in 1:n
        for u in neighbors(g, v)
            push!(adjncy, u - 1)
        end
        xadj[v + 1] = length(adjncy)
    end
    options = fill(Metis.idx_t(-1), Metis.METIS_NOPTIONS)
    options[Metis.METIS_OPTION_SEED + 1] = seed
    options[Metis.METIS_OPTION_UFACTOR + 1] = UFACTOR[]
    part = Vector{Metis.idx_t}(undef, n); sepsize = Metis.idx_t[0]
    Metis.@check Metis.METIS_ComputeVertexSeparator(Ref{Metis.idx_t}(n), xadj, adjncy, iw, options, sepsize, part)
    return part
end

# subgraph of h on vs (h-local ids) plus a clique on the h-local ids in clq
function subfill(h::SimpleGraph, vs::Vector{Int}, clq::Vector{Int})
    sub, vmap = induced_subgraph(h, vs)
    loc = Dict(v => i for (i, v) in enumerate(vmap))
    cl = [loc[v] for v in clq]
    for i in eachindex(cl), j in i+1:length(cl)
        add_edge!(sub, cl[i], cl[j])
    end
    return sub, vmap
end

"""
    splitpieces(graphs, weights; pmax) -> (splits, pieces)

Filled nested dissection of every graph down to pieces of at most `pmax`
vertices. Piece vertices: interior first, then the separator vertices of
ancestors (boundary).
"""
function splitpieces(graphs::AbstractVector, weights::AbstractVector; pmax::Int = 256, seed::Int = 1)
    pieces = Piece[]; splits = Split[]; cpuleaves = Tuple{Int, SimpleGraph{Int}, Vector{Int}, Vector{Int}}[]
    for (k, g0) in enumerate(graphs)
        g = tosimple(g0)
        w = weights[k]
        nodes = Node[]
        root = splitnode!(nodes, pieces, g, collect(1:nv(g)), falses(nv(g)), w, pmax, seed, k)
        push!(splits, Split(nodes, root))
    end
    return splits, pieces
end

function splitnode!(nodes, pieces, h::SimpleGraph{Int}, verts::Vector{Int}, isbd::BitVector, w, pmax, seed, k)
    n = nv(h)
    if n <= pmax
        return leaf!(nodes, pieces, h, verts, isbd, k)
    end
    iw = metisweights(w[verts])
    BDW[] && (iw[isbd] .= 1)   # boundary vertices are eliminated higher up: free in a separator
    part = metissep(h, iw, seed)
    A = findall(==(0), part); B = findall(==(1), part); S = findall(==(2), part)
    if isempty(A) || isempty(B)
        # cannot split (e.g. near-complete): a CPU leaf
        push!(nodes, Node(-1, 0, 0, [verts[v] for v in 1:n if !isbd[v]]))
        return length(nodes)
    end
    ca = 0; cb = 0
    for (X, which) in ((A, :a), (B, :b))
        vs = sort!(vcat(X, S))
        sub, vmap = subfill(h, vs, S)
        bd = BitVector(isbd[v] for v in vmap)
        sset = Set(S)
        for (i, v) in enumerate(vmap)
            v in sset && (bd[i] = true)
        end
        c = splitnode!(nodes, pieces, sub, verts[vmap], bd, w, pmax, seed, k)
        which === :a ? (ca = c) : (cb = c)
    end
    # S \ boundary, fewest boundary neighbours first
    Sint = [v for v in S if !isbd[v]]
    sort!(Sint; by = v -> count(u -> isbd[u], neighbors(h, v)))
    push!(nodes, Node(0, ca, cb, verts[Sint]))
    return length(nodes)
end

function leaf!(nodes, pieces, h, verts, isbd, k)
    n = nv(h)
    perm = vcat(findall(!, isbd), findall(isbd))
    nint = count(!, isbd)
    if nint == 0
        push!(nodes, Node(0, 0, 0, Int[]))
        return length(nodes)
    end
    g, _ = induced_subgraph(h, perm)   # vertex i of g is perm[i] of h
    push!(pieces, Piece(verts[perm], nint, g, k))
    push!(nodes, Node(length(pieces), 0, 0, Int[]))
    return length(nodes)
end

function assemble!(out::Vector{Int}, s::Split, node::Int, porders, cpuorders)
    nd = s.nodes[node]
    if nd.piece > 0
        append!(out, porders[nd.piece])
    elseif nd.piece < 0
        append!(out, nd.sep)   # CPU leaf (not reached with pmax >= piece sizes)
    else
        nd.a > 0 && assemble!(out, s, nd.a, porders, cpuorders)
        nd.b > 0 && assemble!(out, s, nd.b, porders, cpuorders)
        append!(out, nd.sep)
    end
    return out
end

# --------------------------------------------------------------- search --

mutable struct BState{W}
    b::Bucket{W}
    ids::Vector{Int}            # piece ids
    best::CuMatrix{Int16}
    bestw::CuVector{Int32}
    bestc::CuVector{Float32}
    besttmax::CuVector{Int32}
end

# resident warps on the device for a per-warp shared slice: the job batch
# size of a launch is a multiple of this (device attributes only)
function capacity(bytes_per_warp)
    dev = CUDA.device()
    sms = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    smsm = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
    maxthr = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    return sms * max(1, min(smsm ÷ bytes_per_warp, maxthr ÷ 32))
end

function greedyparams(rng, n)
    out = Vector{GreedyParams}(undef, n)
    for i in 1:n
        r = rand(rng, 1:6)
        noise = rand(rng, (0.0f0, 0.02f0, 0.1f0, 0.3f0))
        out[i] = r == 1 ? GreedyParams(1, 0, 0, noise, rand(rng, UInt32)) :
                 r == 2 ? GreedyParams(0, 1, 0, noise, rand(rng, UInt32)) :
                 r <= 4 ? GreedyParams(0, 0, 1, noise, rand(rng, UInt32)) :
                 GreedyParams(rand(rng, Float32), rand(rng, Float32) * 0.1f0, rand(rng, Float32), noise, rand(rng, UInt32))
    end
    return out
end

"""
    gpusearch(pieces, weights; time, seed, mode) -> (orders per piece, widths)

Batched search over all pieces. Round 1: K randomized greedy orderings per
piece. Then, until `time` runs out: rounds of local-search mutations and
fresh randomized greedies, on the pieces whose best width is the largest of
their graph (the width of a graph is the max over its pieces).
"""
function gpusearch(pieces::Vector{Piece}, weights; time::Float64 = 1.0, seed::Int = 1, minrounds::Int = 1, verbose::Bool = false, focus::Bool = true)
    st = STATS[]
    rng = Xoshiro(seed)
    t0 = Base.time()
    # bucket and pack
    byW = Dict{Int, Vector{Int}}()
    for (i, pc) in enumerate(pieces)
        push!(get!(byW, bucketwords(nv(pc.g)), Int[]), i)
    end
    states = BState[]
    for W in sort!(collect(keys(byW)))
        ids = byW[W]
        gs = [pieces[i].g for i in ids]
        ew = [isempty(pieces[i].ew) ? Int32[round(Int32, Q * weights[pieces[i].graph][v]) for v in pieces[i].verts] : pieces[i].ew for i in ids]
        sw = [isempty(pieces[i].sw) ? Int32[round(Int32, SQ * weights[pieces[i].graph][v]) for v in pieces[i].verts] : pieces[i].sw for i in ids]
        b = pack(gs, ew, W; nints = [pieces[i].nint for i in ids], sweights = sw)
        P = length(ids); N = 64W
        push!(states, BState{W}(b, ids, CUDA.zeros(Int16, N, P), CUDA.fill(typemax(Int32), P), CUDA.fill(Inf32, P), CUDA.zeros(Int32, P)))
        st.work += sum(nv(g)^3 for g in gs) / 4096
    end
    CUDA.synchronize()
    st.tpack += Base.time() - t0
    st.npieces += length(pieces)
    append!(st.pieces_n, [nv(pc.g) for pc in pieces])

    t1 = Base.time()
    ngraphs = maximum(pc.graph for pc in pieces; init = 0)
    pwidth = fill(typemax(Int32), length(pieces))
    rnd = 0
    while true
        rnd += 1
        # which pieces get work this round: all in round 1, then the pieces at
        # the max width of their graph (ties: all of them)
        gmax = fill(Int32(-1), ngraphs)
        for (i, pc) in enumerate(pieces)
            gmax[pc.graph] = max(gmax[pc.graph], pwidth[i])
        end
        for s in states
            searchround!(s, pieces, pwidth, gmax, rnd, rng, focus)
        end
        CUDA.synchronize()
        for s in states
            hw = Array(s.bestw)
            for (k, i) in enumerate(s.ids)
                pwidth[i] = hw[k]
            end
        end
        elapsed = Base.time() - t1
        if verbose
            @printf("  round %d  t=%.3fs  max piece width %d\n", rnd, elapsed, maximum(pwidth; init = 0))
        end
        (elapsed >= time && rnd >= minrounds) && break
    end
    st.rounds += rnd
    st.tgpu += Base.time() - t1

    t2 = Base.time()
    orders = Vector{Vector{Int}}(undef, length(pieces))
    for s in states
        hb = Array(s.best)
        for (k, i) in enumerate(s.ids)
            pc = pieces[i]
            orders[i] = [pc.verts[Int(hb[t, k]) + 1] for t in 1:pc.nint]
        end
    end
    st.tasm += Base.time() - t2
    return orders, pwidth
end

function searchround!(s::BState{W}, pieces, pwidth, gmax, round, rng, focus) where {W}
    st = STATS[]
    N = 64W; b = s.b
    if round == 1 || !focus
        active = collect(1:b.P)
    else
        active = [k for (k, i) in enumerate(s.ids) if pwidth[i] >= gmax[pieces[i].graph]]
    end
    isempty(active) && return
    cap = capacity(N * W * 8 + 128)
    # jobs per active piece: fill the device a few times over, at least 8
    K = max(8, cld(4cap, length(active)))
    K = min(K, 4096)
    njobs = K * length(active)
    pidx = CuArray(Int32.(repeat(active; inner = K)))
    bstart = CuArray(Int32.(collect(1:K:njobs + 1)))
    bpiece = CuArray(Int32.(active))
    cand = CUDA.zeros(Int16, N, njobs)
    width = CUDA.zeros(Int32, njobs); lcost = CUDA.zeros(Float32, njobs); tmaxs = CUDA.zeros(Int32, njobs)
    if round == 1 || isodd(round)
        greedy!(cand, width, lcost, tmaxs, b, pidx, CuArray(greedyparams(rng, njobs)); invq = 1 / Q)
    end
    if round > 1
        # local search: a few moves around the incumbent
        if isodd(round)
            selectbest!(bstart, bpiece, cand, width, lcost, tmaxs, s.best, s.bestw, s.bestc, s.besttmax, b)
        end
        mp = [MutateParams(rand(rng, (1, 1, 2, 3)), 0.5f0, rand(rng, (0, 4, 16)), rand(rng, UInt32)) for _ in 1:njobs]
        mutate!(cand, width, lcost, tmaxs, b, pidx, s.best, s.besttmax, CuArray(mp); invq = 1 / Q)
        st.nscored += njobs
    end
    selectbest!(bstart, bpiece, cand, width, lcost, tmaxs, s.best, s.bestw, s.bestc, s.besttmax, b)
    st.nscored += njobs
    return
end

# --------------------------------------------------------- the algorithm --

"""
    GPUSearch(; time, pmax, seed)

CliqueTrees elimination algorithm: a batch of graphs (e.g. the atoms of
SafeSeparators) is split into pieces of at most `pmax` vertices on the CPU and
searched on the GPU for `time` seconds in total.
"""
Base.@kwdef struct GPUSearch <: EliminationAlgorithm
    time::Float64 = 1.0
    pmax::Int = 256
    seed::Int = 1
end

CliqueTrees.isconcurrent(::GPUSearch) = false

function hybrid(weights::AbstractVector{<:AbstractVector}, graphs::AbstractVector, alg::GPUSearch)
    st = STATS[]
    t0 = time()
    splits, pieces = splitpieces(graphs, weights; pmax = alg.pmax, seed = alg.seed)
    st.tsplit += time() - t0
    porders, _ = isempty(pieces) ? (Vector{Int}[], Int32[]) : gpusearch(pieces, weights; time = alg.time, seed = alg.seed)
    t1 = time()
    out = [assemble!(Int[], splits[k], splits[k].root, porders, nothing) for k in eachindex(graphs)]
    st.tasm += time() - t1
    for (k, o) in enumerate(out)
        @assert length(o) == nv(graphs[k]) && isperm(o)
    end
    return out
end

function CliqueTrees.batch_permutation(weights::AbstractVector{<:AbstractVector}, graphs::AbstractVector{<:AbstractGraph}, alg::GPUSearch; verbose::Bool = false)
    V = eltype(graphs[1])
    return [convert(Vector{V}, o) for o in hybrid([collect(Float64, w) for w in weights], graphs, alg)]
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::GPUSearch; verbose::Bool = false) where {V}
    o = convert(Vector{V}, only(hybrid([collect(Float64, weights)], [graph], alg)))
    return o, invperm(o)
end

include("GPUND.jl")

end
