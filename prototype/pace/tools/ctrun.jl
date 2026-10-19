# usage: julia --project=<mtenv> ctrun.jl <mt_seconds> <in.gr> <out.td> [nt]
# Prints one line: status width solve_seconds   (status: FOUND | TIMEOUT | ABOVE)
using CliqueTrees, Graphs
using CliqueTrees: BT_FOUND, BT_ABOVE, BT_TIMEOUT

function readgr(path)
    n = 0; edges = Tuple{Int,Int}[]
    for line in eachline(path)
        isempty(line) && continue
        c = line[1]
        (c == 'c') && continue
        t = split(line)
        if c == 'p'; n = parse(Int, t[3])
        else
            u, v = parse(Int, t[1]), parse(Int, t[2])
            u != v && push!(edges, (u, v))
        end
    end
    g = SimpleGraph(n)
    for (u, v) in edges; add_edge!(g, u, v); end
    return g
end

# exact treewidth (unit weights) of each connected component, shared deadline
function solve(g, mt; nt = 1)
    dl = time() + mt
    n = nv(g); order = Int[]; status = BT_FOUND; width = 0
    for comp in connected_components(g)
        if length(comp) <= 2 # trivial; width = |comp| - 1
            append!(order, comp); width = max(width, length(comp)); continue
        end
        h, _ = induced_subgraph(g, comp)
        perm, invp, k, st = CliqueTrees.septw(ones(Int, length(comp)), h, 0, typemax(Int), dl, nt)
        if st != BT_FOUND
            return nothing, k, st
        end
        append!(order, comp[perm]); width = max(width, k)
    end
    return order, width - 1, status # septw's k is the bag weight, tw + 1
end

# elimination game -> bags {v} U higher neighbours, tree by first higher neighbour
function writetd(path, g, order)
    n = nv(g); pos = zeros(Int, n); pos[order] = 1:n
    adj = [sort!([pos[w] for w in neighbors(g, order[i])]) for i in 1:n] # in positions
    bags = Vector{Vector{Int}}(undef, n); parent = zeros(Int, n)
    higher = [filter(>(i), adj[i]) for i in 1:n]
    for i in 1:n
        hi = higher[i]
        bags[i] = vcat(i, hi)
        if !isempty(hi)
            p = hi[1]; parent[i] = p
            rest = @view hi[2:end]
            higher[p] = sort!(unique!(vcat(higher[p], rest)))
        end
    end
    roots = [i for i in 1:n if parent[i] == 0]
    w = maximum(length, bags) - 1
    open(path, "w") do io
        println(io, "s td $n $(w + 1) $n")
        for i in 1:n
            println(io, "b $i ", join((order[j] for j in bags[i]), " "))
        end
        for i in 1:n; parent[i] != 0 && println(io, "$i $(parent[i])"); end
        for k in 2:length(roots); println(io, "$(roots[k-1]) $(roots[k])"); end
    end
    return w
end

function main()
    mt = parse(Float64, ARGS[1]); nt = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    solve(path_graph(5), 10.0; nt) # warm-up
    g = readgr(ARGS[2])
    t = @elapsed (order, width, st) = solve(g, mt; nt)
    if order === nothing
        println(st == BT_TIMEOUT ? "TIMEOUT" : "ABOVE", " ", width - 1, " ", round(t; digits = 3))
    else
        w = writetd(ARGS[3], g, order)
        println("FOUND ", w, " ", round(t; digits = 3), w == width ? "" : " WIDTHMISMATCH($width)")
    end
end
main()
