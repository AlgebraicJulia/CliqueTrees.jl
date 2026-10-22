# Step 1 data: per network, a pool of orderings (all seed .ord files + GPU-perturbed variants of the
# seeds), each scored by (a) lcost = log2 sum_v 2^{w(bag_v)} (gpuord's score on the line graph, here
# computed exactly in Float64 on the CPU) and (b) our exact C of the induced bucket tree (GPU kernel).
# cotengra's C of ftn.tree_from_order is added by corr_ctg.py.
#   julia --project=env scripts/corr_orders.jl <list> <orddir> <outdir> <nvariants>
include(joinpath(@__DIR__, "..", "src", "Host.jl")); using .Host
include(joinpath(@__DIR__, "..", "src", "GPUTree.jl")); using .GPUTree
include("/home/itaykadosh/Projects/GATAS/tn-work/tncorpus/load.jl"); using .TNCorpus
using CUDA, Random

function lcost(inst, order)
    n = inst.n; w = inst.weights
    adj = [Set{Int}(inst.rowval[inst.colptr[v]:inst.colptr[v+1]-1]) for v in 1:n]
    alive = trues(n); s = 0.0; m = -Inf; bags = Float64[]
    for v in order
        nb = [u for u in adj[v] if alive[u]]
        b = w[v] + sum((w[u] for u in nb); init = 0.0)
        push!(bags, b)
        for u in nb
            union!(adj[u], nb); delete!(adj[u], u)
        end
        alive[v] = false
    end
    m = maximum(bags)
    m + log2(sum(exp2.(bags .- m))), m
end

names = readlines(ARGS[1]); orddir = ARGS[2]; out = ARGS[3]; nv = parse(Int, ARGS[4])
mkpath(out); Random.seed!(1)
for name in names
    files = sort(filter(f -> startswith(f, name * "__") && endswith(f, ".ord"), readdir(orddir)))
    isempty(files) && continue
    inst = TNCorpus.load(name; graph = false); net = readnet(name); g = gnet(net)
    tags = [split(replace(f, ".ord" => ""), "__")[2] for f in files]
    base = reduce(hcat, [Int32.(open_last(net, parse.(Int, split(readlines(joinpath(orddir, f))[2])))) .- Int32(1) for f in files])
    nb = size(base, 2); J = nb + nv
    sc = scratch(g, J)
    # variants: perturb the HBT/hbt-nd seeds and the nd seeds, a range of strengths
    pick = findall(t -> startswith(t, "hbt") || startswith(t, "nd"), tags)
    bidx = Int32[j <= nb ? j : pick[rand(1:length(pick))] for j in 1:J]
    nm = Int32[j <= nb ? 0 : rand((1, 2, 4, 8, 16, 32, 64)) for j in 1:J]
    md = Int32[rand((2, 8, 32)) for j in 1:J]
    perturb!(sc, g, CuArray(base), CuArray(bidx), CuArray(nm), CuArray(md), CuArray(rand(UInt64, J)))
    score_orders!(sc, g)
    orders = Array(sc.orders) .+ 1; lc = Array(sc.log2C); W = Array(sc.W)
    open(joinpath(out, name * ".tsv"), "w") do io
        println(io, "k\ttag\tnmoves\tmaxd\tlcost\tbagw\tgpu_log2C\tgpu_W")
        for j in 1:J
            l, bw = lcost(inst, orders[:, j])
            tag = j <= nb ? tags[j] : "var_" * tags[bidx[j]]
            println(io, join((j, tag, nm[j], md[j], round(l; digits = 5), round(bw; digits = 4), lc[j], W[j]), '\t'))
        end
    end
    open(joinpath(out, name * ".orders"), "w") do io
        for j in 1:J
            println(io, join(orders[:, j], ' '))
        end
    end
    println(name, " ", J, " orderings"); flush(stdout)
end
