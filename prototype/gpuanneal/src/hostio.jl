# Host-side network / tree I/O and exact re-costing (no GPU dependency); included by GPUAnneal.jl

# ------------------------------------------------------------------ network --

struct Net
    name::String
    n::Int                       # indices (dim > 1)
    nt::Int                      # tensors (leaves; scalar tensors dropped)
    W::Int                       # 64-bit words per bitset
    dims::Vector{Int}
    tens::Vector{Vector{Int}}    # 1-based index ids
    open::Vector{Int}
    K::Int
    cw::Vector{Float64}          # class weights log2(dim)
    cmask::Matrix{UInt64}        # (W, K)
    leaf::Matrix{UInt64}         # (W, nt) leaf leg sets
    outm::Vector{UInt64}         # output mask
    wt::Vector{Float64}          # per index weight
end

function readtnc(path)
    open(path) do io
        @assert read(io, 8) == b"TNCORP1\0"
        h = Vector{Int64}(undef, 5); read!(io, h)
        n, nt, nnz, nopen, hnnz = h
        rd(T, k) = (v = Vector{T}(undef, k); read!(io, v); v)
        dims = rd(Int64, n); rd(Int64, n + 1); rd(Int32, nnz)
        openv = rd(Int32, nopen); hptr = rd(Int64, nt + 1); hidx = rd(Int32, hnnz)
        tens = [Int.(hidx[hptr[t]:hptr[t+1]-1]) for t in 1:nt]
        (; dims = Int.(dims), tens, open = Int.(openv))
    end
end

setbit!(b::AbstractVector{UInt64}, i) = (b[(i-1)>>6+1] |= UInt64(1) << ((i - 1) & 63); b)
getbit(b::AbstractVector{UInt64}, i) = (b[(i-1)>>6+1] >> ((i - 1) & 63)) & 1 == 1

function readnet(path::AbstractString; name = splitext(basename(path))[1])
    r = readtnc(path)
    tens = [t for t in r.tens if !isempty(t)]
    n = length(r.dims); nt = length(tens); W = cld(n, 64)
    ud = sort(unique(r.dims)); K = length(ud)
    cw = log2.(Float64.(ud))
    cmask = zeros(UInt64, W, K)
    for i in 1:n
        setbit!(view(cmask, :, searchsortedfirst(ud, r.dims[i])), i)
    end
    cnt = zeros(Int, n)
    for t in tens, i in t
        cnt[i] += 1
    end
    isout = falses(n); isout[r.open] .= true
    leaf = zeros(UInt64, W, nt)
    for (k, t) in enumerate(tens), i in t
        (cnt[i] >= 2 || isout[i]) && setbit!(view(leaf, :, k), i)
    end
    outm = zeros(UInt64, W)
    for i in r.open
        setbit!(outm, i)
    end
    Net(name, n, nt, W, r.dims, tens, r.open, K, cw, cmask, leaf, outm, log2.(Float64.(r.dims)))
end

wof(net::Net, b::AbstractVector{UInt64}) = sum(net.cw[c] * sum(count_ones(b[w] & net.cmask[w, c]) for w in 1:net.W) for c in 1:net.K)

# -------------------------------------------------------------------- trees --

"children (2, nn) Int32 from a cotengra ssa path (0-based pairs)"
function tree_from_ssa(net::Net, ssa::Vector{Tuple{Int,Int}})
    nt = net.nt; nn = 2nt - 1
    @assert length(ssa) == nt - 1 "ssa path has $(length(ssa)) contractions, expected $(nt-1)"
    CH = zeros(Int32, 2, nn)
    for (k, (a, b)) in enumerate(ssa)
        CH[1, nt+k] = a + 1; CH[2, nt+k] = b + 1
    end
    CH
end

"post-order ssa path of a child array (root = nn)"
function ssa_from_children(net::Net, CH::AbstractMatrix)
    nt = net.nt; nn = 2nt - 1
    id = zeros(Int, nn); for i in 1:nt; id[i] = i - 1; end
    nxt = nt; ssa = Tuple{Int,Int}[]
    stack = [(nn, false)]
    while !isempty(stack)
        v, done = pop!(stack)
        if v <= nt
            continue
        elseif done
            push!(ssa, (id[CH[1, v]], id[CH[2, v]])); id[v] = nxt; nxt += 1
        else
            push!(stack, (v, true)); push!(stack, (Int(CH[2, v]), false)); push!(stack, (Int(CH[1, v]), false))
        end
    end
    ssa
end

"leg sets of all nodes, (W, nn): bottom-up unions U, then top-down L(a) = U(a) & (U(b) | L(v))"
function legs(net::Net, CH::AbstractMatrix)
    nt = net.nt; nn = 2nt - 1; W = net.W
    U = zeros(UInt64, W, nn); U[:, 1:nt] .= net.leaf
    # ssa order: children of v have smaller ids? not after rotations -> use explicit post-order
    order = Int[]; stack = [nn]
    while !isempty(stack)
        v = pop!(stack); push!(order, v)
        if v > nt
            push!(stack, CH[1, v]); push!(stack, CH[2, v])
        end
    end
    for v in Iterators.reverse(order)
        v > nt && (U[:, v] .= U[:, CH[1, v]] .| U[:, CH[2, v]])
    end
    L = similar(U); L[:, nn] .= net.outm   # root legs = output
    for v in order
        v <= nt && continue
        a = CH[1, v]; b = CH[2, v]
        L[:, a] .= U[:, a] .& (U[:, b] .| L[:, v])
        L[:, b] .= U[:, b] .& (U[:, a] .| L[:, v])
    end
    L
end

"""Exact host (re)costing in Float64: log2 C_S, W (max node size after slicing, all nodes),
maxop (largest single contraction), excess over Wt."""
function hostcost(net::Net, CH::AbstractMatrix, S::AbstractVector{UInt64}; Wt = Inf)
    nt = net.nt; nn = 2nt - 1
    L = legs(net, CH)
    ns = .~S
    ws = wof(net, S)
    mx = -Inf; terms = Float64[]
    Wmax = -Inf; exc = 0.0
    for v in 1:nn
        sz = wof(net, L[:, v] .& ns)
        Wmax = max(Wmax, sz)
        if v > nt
            exc += max(0.0, sz - Wt)
            inv = (L[:, CH[1, v]] .| L[:, CH[2, v]]) .& ns
            push!(terms, wof(net, inv))
        end
    end
    m = maximum(terms)
    lc = ws + m + log2(sum(exp2(t - m) for t in terms))
    (; log2C = lc, W = Wmax, maxop = m, excess = exc, nsliced = sum(count_ones, S))
end

function read_tree(path)
    lines = readlines(path)
    h = split(lines[1]); nt = parse(Int, h[2])
    flat = parse.(Int, split(lines[2]))
    sl = length(lines) >= 3 ? parse.(Int, split(lines[3])) : Int[]
    ssa = [(flat[2k-1], flat[2k]) for k in 1:length(flat)÷2]
    nt, ssa, sl
end

function write_tree(path, nt, ssa, sliced; meta = Dict{String,String}())
    open(path * ".tmp", "w") do io
        println(io, "TREE $nt $(length(sliced))")
        println(io, join(("$a $b" for (a, b) in ssa), ' '))
        println(io, join(sliced, ' '))
        for (k, v) in meta
            println(io, "# $k $v")
        end
    end
    mv(path * ".tmp", path; force = true)
end

slicemask(net::Net, sl) = (S = zeros(UInt64, net.W); foreach(i -> setbit!(S, i), sl); S)
sliceids(net::Net, S) = [i for i in 1:net.n if getbit(S, i)]


"""
    greedy_slice(net, CH, Wt; S0) -> slice ids

Greedy slicing of a fixed tree to width <= Wt: repeatedly slice the index (among the legs of the currently
widest intermediates) that minimizes log2 C_S + 2 * (sum over nodes of the excess width), until feasible.
"""
function greedy_slice(net::Net, CH::AbstractMatrix, Wt::Real; S0 = Int[])
    nt = net.nt; nn = 2nt - 1; W = net.W
    L = legs(net, CH)
    S = slicemask(net, S0)
    inv = zeros(UInt64, W, nn - nt)
    for v in nt+1:nn
        inv[:, v-nt] .= L[:, CH[1, v]] .| L[:, CH[2, v]]
    end
    function score(S)
        ns = .~S
        ws = wof(net, S)
        t = [wof(net, view(inv, :, j) .& ns) for j in 1:nn-nt]
        m = maximum(t)
        lc = ws + m + log2(sum(exp2(x - m) for x in t))
        sz = [wof(net, view(L, :, v) .& ns) for v in nt+1:nn]
        exc = sum(max(0.0, x - Wt) for x in sz)
        lc, exc, maximum(sz), sz
    end
    lc, exc, mw, sz = score(S)
    while exc > 1e-9
        cand = Int[]
        for (j, v) in enumerate(nt+1:nn)
            if sz[j] >= mw - 1e-9
                for i in 1:net.n
                    getbit(view(L, :, v), i) && !getbit(S, i) && push!(cand, i)
                end
            end
        end
        unique!(cand)
        best = (Inf, 0)
        for i in cand
            S2 = copy(S); setbit!(S2, i)
            l2, e2, _, _ = score(S2)
            best = min(best, (l2 + 2e2, i))
        end
        setbit!(S, best[2])
        lc, exc, mw, sz = score(S)
    end
    sliceids(net, S)
end
