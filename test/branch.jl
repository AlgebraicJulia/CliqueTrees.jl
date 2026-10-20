using CliqueTrees
using CliqueTrees: BipartiteGraph
using Graphs
using Random
using Test

# a pointed hypergraph from the index lists of the tensors and the output
function branchhypergraph(ixs::Vector{Vector{Int}}, iy::Vector{Int}, n::Int)
    ptr = [1]; tgt = Int[]

    for ix in ixs
        append!(tgt, ix); push!(ptr, length(tgt) + 1)
    end

    append!(tgt, iy); push!(ptr, length(tgt) + 1)
    return BipartiteGraph{Int, Int}(n, length(ixs) + 1, length(tgt), ptr, tgt), length(ixs) + 1
end

# a random contraction tree over the tensors 1:m - 1 (the point is m)
function branchrandtree(rng::AbstractRNG, m::Int)
    parent = zeros(Int, 2m - 2); pool = collect(1:m - 1); next = m + 1

    while length(pool) > 1
        shuffle!(rng, pool); a = pop!(pool); b = pop!(pool)
        parent[a] = next; parent[b] = next; push!(pool, next); next += 1
    end

    parent[only(pool)] = m
    return parent
end

# a parent vector of a binary contraction tree rooted at the point, with every tensor a leaf
function branchvalid(parent::AbstractVector{Int}, point::Int)
    nn = length(parent); m = (nn + 2) ÷ 2
    nchild = zeros(Int, nn)

    for x in 1:nn
        if parent[x] != 0
            nchild[parent[x]] += 1
        end
    end

    parent[point] == 0 || return false
    nchild[point] == 1 || return false
    all(x -> x == point || nchild[x] == 0, 1:m) || return false
    all(x -> nchild[x] == 2, m + 1:nn) || return false

    # every node reaches the point
    for x in 1:nn
        y = x; k = 0

        while y != point
            y = parent[y]; k += 1
            (y == 0 || k > nn) && return false
        end
    end

    return true
end

# internal nodes numbered after their children
function branchchildrenfirst(parent::AbstractVector{Int}, point::Int)
    return all(x -> parent[x] == 0 || parent[x] == point || parent[x] > x, eachindex(parent))
end

# every step contracts two distinct ids that already exist
function branchssavalid(path::AbstractVector, ntensors::Int)
    next = ntensors

    for (a, b) in path
        (a < next && b < next && a != b) || return false
        next += 1
    end

    return true
end

@testset "branch decompositions" begin
    rng = Xoshiro(1)

    # a lattice partition function and a random 3-regular independent-set network
    nets = Tuple{Vector{Vector{Int}}, Vector{Int}, Vector{Float64}}[]
    g = Graphs.grid([6, 6])
    ixs = [Int[] for _ in vertices(g)]

    for (k, e) in enumerate(edges(g))
        push!(ixs[src(e)], k); push!(ixs[dst(e)], k)
    end

    push!(nets, (ixs, Int[], fill(1.0, ne(g))))
    g = random_regular_graph(40, 3; rng)
    push!(nets, ([[[v] for v in vertices(g)]; [[src(e), dst(e)] for e in edges(g)]], [1, 2], fill(1.0, nv(g))))

    # random hypergraphs: many words, many weight classes (the unaligned path), outputs,
    # vertices of degree 1, tensors without vertices
    for _ in 1:20
        n = rand(rng, 1:200)
        classes = rand(rng, ([1.0], [1.0, log2(3)], collect(3.0:14.0) ./ 3, [0.0, 1.0, 2.0]))
        ixs = [unique(rand(rng, 1:n, rand(rng, 0:6))) for _ in 1:rand(rng, 3:60)]
        iy = unique(rand(rng, 1:n, rand(rng, 0:3)))
        push!(nets, (ixs, iy, [rand(rng, classes) for _ in 1:n]))
    end

    @testset "anneal" for (i, (ixs, iy, weights)) in enumerate(nets)
        H, point = branchhypergraph(ixs, iy, length(weights))
        m = nv(H)

        for parent in (branchtree(weights, H, point), branchrandtree(rng, m))
            stats = branchstats(weights, H, point, parent)

            # default: never wider; children numbered first; a valid SSA path
            new, info = branchanneal(weights, H, point, parent; sweeps = 50, seed = i, info = true)
            newstats = branchstats(weights, H, point, new)
            @test branchvalid(new, point)
            @test branchchildrenfirst(new, point)
            @test branchssavalid(branchssa(new, point), m - 1)
            @test newstats.cost <= stats.cost + 1.0e-9
            @test newstats.width <= stats.width + 1.0e-9
            @test info.feasible && info.sweeps == 50 && info.proposals == 50 * (m - 2)
            @test info.cost ≈ newstats.cost && info.width == newstats.width

            # deterministic
            @test new == branchanneal(weights, H, point, parent; sweeps = 50, seed = i)

            # dense and sparse legs, a single cycle, no refinement, no cap, mixed criterion
            for dense in (true, false)
                new = branchanneal(weights, H, point, parent; sweeps = 50, beta0 = 1.0, W = Inf,
                    mix = 0.5, refine = false, seed = i, dense)
                @test branchvalid(new, point)
                @test branchchildrenfirst(new, point)
                @test branchstats(weights, H, point, new).cost <= stats.cost + 1.0e-9
            end

            # an explicit cap above the start
            W = stats.width + 2
            new = branchanneal(weights, H, point, parent; sweeps = 50, W, mix = 0.0, seed = i)
            @test branchvalid(new, point)
            @test branchstats(weights, H, point, new).width <= W + 1.0e-9

            # an explicit cap below the start: within it if feasible, else the narrowest seen
            W = stats.width - 2
            new, info = branchanneal(weights, H, point, parent; sweeps = 50, W, seed = i, info = true)
            width = branchstats(weights, H, point, new).width
            @test branchvalid(new, point)
            @test info.feasible == (width <= W + 1.0e-9)
            @test width <= stats.width + 1.0e-9

            # a time limit (the sweeps are not limited)
            new, info = branchanneal(weights, H, point, parent; time = 0.01, seed = i, info = true)
            @test branchvalid(new, point)
        end
    end

    @testset "anneal dense = sparse" begin
        # with unit weights the two kinds of legs make the same moves
        for (ixs, iy, weights) in nets[1:2]
            H, point = branchhypergraph(ixs, iy, length(weights))
            parent = branchrandtree(rng, nv(H))
            a = branchanneal(weights, H, point, parent; sweeps = 100, dense = true, refine = false)
            b = branchanneal(weights, H, point, parent; sweeps = 100, dense = false, refine = false)
            @test a == b
        end
    end

    @testset "anneal time" begin
        ixs, iy, weights = nets[1]
        H, point = branchhypergraph(ixs, iy, length(weights))
        parent = branchtree(weights, H, point)
        new, info = branchanneal(weights, H, point, parent; time = 0.2, refine = false, info = true)
        @test info.sweeps > 1000
        @test info.time >= 0.2
    end

    @testset "anneal improves" begin
        ixs, iy, weights = nets[1]
        H, point = branchhypergraph(ixs, iy, length(weights))
        parent = branchrandtree(rng, nv(H))
        new = branchanneal(weights, H, point, parent; sweeps = 500, W = Inf)
        @test branchstats(weights, H, point, new).cost < branchstats(weights, H, point, parent).cost - 10
    end

    @testset "anneal tiny" begin
        H, point = branchhypergraph([[1, 2], [2]], [1], 2)
        parent = branchtree([1.0, 1.0], H, point)
        @test branchanneal([1.0, 1.0], H, point, parent) == parent
    end
end
