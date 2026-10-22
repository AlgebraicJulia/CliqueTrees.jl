# Host-side (CPU) reference for gpucost: networks, the contraction tree induced by an elimination
# ordering, its exact cotengra cost, legs, ssa export, and subtree (chunk) extraction/replacement.
#
# Cost convention = cotengra 0.8.2 ContractionTree.contraction_cost(): C = sum over internal nodes p
# of prod(dims of legs(left) U legs(right)), where legs(x) = indices with some but not all
# occurrences inside x (open indices count one extra occurrence, so they stay legs up to the root);
# an index occurring in one tensor only and not open is summed in preprocessing (no cost).
# Equivalently: index i is involved in contraction p iff p lies in the Steiner tree spanned by the
# leaves holding i (plus the root if i is open). That is what `cost!` computes.
module Host

export Net, readnet, Tree, bucket_tree, cost!, tree_ssa, tree_from_ssa, renumber!, open_last

include(get(ENV, "TNCORPUS_LOAD", "/home/itaykadosh/Projects/GATAS/tn-work/tncorpus/load.jl"))
using .TNCorpus

struct Net
    name::String
    family::String
    n::Int                       # indices = line-graph vertices 1..n
    T::Int                       # tensors (non-scalar), leaves 1..T
    w::Vector{Float64}           # log2 dims
    dims::Vector{Int}
    isopen::BitVector
    open::Vector{Int}
    tens::Vector{Vector{Int}}    # tensor -> unique indices (tnc order)
    occptr::Vector{Int}          # index -> tensors holding it (sorted, unique)
    occ::Vector{Int}
end

function readnet(name::AbstractString)
    inst = TNCorpus.load(name; graph = false)
    tens = Vector{Int}[]
    for t in 1:length(inst.hptr)-1
        ix = inst.hidx[inst.hptr[t]:inst.hptr[t+1]-1]
        isempty(ix) && continue
        push!(tens, ix)
    end
    n = inst.n; T = length(tens)
    lists = [Int[] for _ in 1:n]
    for (t, ix) in enumerate(tens), i in unique(ix)
        push!(lists[i], t)
    end
    occptr = cumsum(vcat(1, length.(lists)))
    isopen = falses(n); isopen[inst.open] .= true
    Net(name, inst.family, n, T, log2.(Float64.(inst.dims)), inst.dims, isopen, inst.open, tens, occptr, reduce(vcat, lists; init = Int[]))
end

occof(net::Net, i) = @view net.occ[net.occptr[i]:net.occptr[i+1]-1]

"""Binary contraction tree. Leaves 1..T, internal nodes T+1..2T-1 in topological order (children
before parents), root = 2T-1. left/right/parent are indexed by node (0 = none)."""
mutable struct Tree
    T::Int
    left::Vector{Int}
    right::Vector{Int}
    parent::Vector{Int}
    acc::Vector{Float64}       # log2 flops of the contraction creating p (0 for leaves)
    leg::Vector{Float64}       # log2 size of node p
    log2C::Float64
    W::Float64
end
Tree(T) = Tree(T, zeros(Int, 2T - 1), zeros(Int, 2T - 1), zeros(Int, 2T - 1), zeros(2T - 1), zeros(2T - 1), NaN, NaN)

function findroot!(rep, x)
    while rep[x] != x
        rep[x] = rep[rep[x]]; x = rep[x]
    end
    x
end

"""Tree induced by the elimination ordering `order` (index ids, first eliminated first): bucket
elimination; open indices are skipped; each bucket's classes are merged left to right in increasing
order of their current tree node id (a caterpillar); leftover classes are merged the same way at the
end. This rule is replicated exactly by the GPU kernel."""
function bucket_tree(net::Net, order::AbstractVector{<:Integer})
    T = net.T; tr = Tree(T)
    rep = collect(1:T); cur = collect(1:T); nxt = T
    roots = Int[]
    for v in order
        net.isopen[v] && continue
        empty!(roots)
        for t in occof(net, v)
            r = findroot!(rep, t)
            r in roots || push!(roots, r)
        end
        length(roots) < 2 && continue
        sort!(roots; by = r -> cur[r])
        node = cur[roots[1]]
        for j in 2:length(roots)
            nxt += 1; tr.left[nxt] = node; tr.right[nxt] = cur[roots[j]]; node = nxt
            rep[roots[j]] = roots[1]
        end
        cur[roots[1]] = node
    end
    empty!(roots)
    for t in 1:T
        findroot!(rep, t) == t && push!(roots, t)
    end
    sort!(roots; by = r -> cur[r])
    node = cur[roots[1]]
    for j in 2:length(roots)
        nxt += 1; tr.left[nxt] = node; tr.right[nxt] = cur[roots[j]]; node = nxt
    end
    @assert nxt == 2T - 1
    for p in T+1:2T-1
        tr.parent[tr.left[p]] = p; tr.parent[tr.right[p]] = p
    end
    tr
end

"""Exact cost: fills tr.acc, tr.leg, tr.log2C, tr.W. If `legs` is given (Vector{Vector{Int}}),
also collects the leg index list of every node (root legs = open indices)."""
function cost!(tr::Tree, net::Net; legs = nothing)
    T = tr.T; N = 2T - 1; root = N
    sz = ones(Int, N); pre = zeros(Int, N)
    for p in T+1:N
        sz[p] = 1 + sz[tr.left[p]] + sz[tr.right[p]]
    end
    for p in N:-1:T+1
        l = tr.left[p]; r = tr.right[p]
        pre[l] = pre[p] + 1; pre[r] = pre[p] + 1 + sz[l]
    end
    isanc(a, b) = pre[a] <= pre[b] < pre[a] + sz[a]
    fill!(tr.acc, 0.0); fill!(tr.leg, 0.0)
    legs === nothing || foreach(empty!, legs)
    buf = Int[]
    for i in 1:net.n
        oc = occof(net, i); op = net.isopen[i]
        (length(oc) < 2 && !op) && continue
        wi = net.w[i]
        empty!(buf); append!(buf, oc); sort!(buf; by = x -> pre[x])
        # first leaf: up to the LCA of all (inclusive), or to the root if open
        x = buf[1]; last = buf[end]
        while true
            top = op ? x == root : isanc(x, last)
            x > T && (tr.acc[x] += wi)
            if !top || op
                tr.leg[x] += wi; legs === nothing || push!(legs[x], i)
            end
            top && break
            x = tr.parent[x]
        end
        for j in 2:length(buf)
            x = buf[j]; prev = buf[j-1]
            while !isanc(x, prev)
                x > T && (tr.acc[x] += wi)
                tr.leg[x] += wi; legs === nothing || push!(legs[x], i)
                x = tr.parent[x]
            end
        end
    end
    m = maximum(@view tr.acc[T+1:N])
    s = 0.0
    for p in T+1:N
        s += exp2(tr.acc[p] - m)
    end
    tr.log2C = m + log2(s)
    tr.W = maximum(tr.leg)
    tr
end

"""Port of frontier ftn.open_last: a perfect elimination ordering of the same filled graph that ends
with the open indices (maximum cardinality search on the filled graph, open clique first, ties by
original position; reversed). Identity when the network has no open index."""
function open_last(net::Net, order::AbstractVector{<:Integer})
    isempty(net.open) && return collect(order)
    n = net.n
    adj = [Set{Int}() for _ in 1:n]
    for t in net.tens, a in t
        union!(adj[a], t)
    end
    for a in net.open
        union!(adj[a], net.open)
    end
    for v in 1:n
        delete!(adj[v], v)
    end
    pos = zeros(Int, n); for (k, v) in enumerate(order); pos[v] = k; end
    fill = adj
    for v in order
        later = [u for u in fill[v] if pos[u] > pos[v]]
        for u in later, x in later
            x != u && push!(fill[u], x)
        end
    end
    card = zeros(Int, n); seen = falses(n); visit = Int[]
    for _ in 1:n
        bv = 0
        for v in order
            seen[v] && continue
            if bv == 0 || (card[v], net.isopen[v], -pos[v]) > (card[bv], net.isopen[bv], -pos[bv])
                bv = v
            end
        end
        seen[bv] = true; push!(visit, bv)
        for u in fill[bv]
            seen[u] || (card[u] += 1)
        end
    end
    peo = reverse(visit)
    @assert all(net.isopen[v] for v in peo[end-length(net.open)+1:end]) "open clique not last"
    peo
end

"""ssa path (0-based, cotengra convention) of a tree in topological order."""
function tree_ssa(tr::Tree)
    [(tr.left[p] - 1, tr.right[p] - 1) for p in tr.T+1:2tr.T-1]
end

function tree_from_ssa(T::Int, ssa)
    tr = Tree(T)
    for (k, (a, b)) in enumerate(ssa)
        p = T + k; tr.left[p] = a + 1; tr.right[p] = b + 1
        tr.parent[a+1] = p; tr.parent[b+1] = p
    end
    tr
end

"""Renumber internal nodes so that children precede parents (postorder from the root `root`),
given arbitrary left/right arrays over node ids (leaves keep ids 1..T)."""
function renumber!(tr::Tree, root::Int; newid_out = nothing)
    T = tr.T; N = 2T - 1
    newid = zeros(Int, length(tr.left)); nxt = T
    stack = [(root, false)]
    while !isempty(stack)
        x, done = pop!(stack)
        x <= T && continue
        if done
            nxt += 1; newid[x] = nxt
        else
            push!(stack, (x, true)); push!(stack, (tr.right[x], false)); push!(stack, (tr.left[x], false))
        end
    end
    @assert nxt == N
    nl = zeros(Int, N); nr = zeros(Int, N)
    mp(y) = y <= T ? y : newid[y]
    for x in eachindex(tr.left)
        newid[x] == 0 && continue
        nl[newid[x]] = mp(tr.left[x]); nr[newid[x]] = mp(tr.right[x])
    end
    tr.left = nl; tr.right = nr; tr.parent = zeros(Int, N)
    for p in T+1:N
        tr.parent[nl[p]] = p; tr.parent[nr[p]] = p
    end
    tr.acc = zeros(N); tr.leg = zeros(N)
    if newid_out !== nothing
        resize!(newid_out, N)
        for x in 1:N
            newid_out[x] = x <= T ? x : newid[x]
        end
    end
    tr
end

function write_tree(path, tr::Tree; meta = Dict())
    open(path * ".tmp", "w") do f
        println(f, "TREE $(tr.T) 0")
        println(f, join(("$(a) $(b)" for (a, b) in tree_ssa(tr)), ' '))
        println(f)
        for (k, v) in meta
            println(f, "# $k $v")
        end
    end
    mv(path * ".tmp", path; force = true)
end

function read_tree(path)
    ls = readlines(path)
    T = parse(Int, split(ls[1])[2])
    fl = parse.(Int, split(ls[2]))
    tree_from_ssa(T, collect(zip(fl[1:2:end], fl[2:2:end])))
end

end
