# PACE 2017 Track B I/O and the solution checker.
# Input: one edge per line, two vertex labels separated by whitespace, '#' comments.
# Output: one fill edge per line (two labels).

# Returns (labels, adj) with adj::Vector{BitSet} over 1..n (loops and multi-edges dropped).
function read_graph(path::AbstractString)
    id = Dict{String, Int}(); labels = String[]; edges = Tuple{Int, Int}[]
    getid(s) = get!(id, s) do
        push!(labels, s); length(labels)
    end
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, '#')) && continue
        f = split(s)
        length(f) < 2 && continue
        a = getid(String(f[1])); b = getid(String(f[2]))
        a != b && push!(edges, (a, b))
    end
    n = length(labels); adj = [BitSet() for _ in 1:n]
    for (a, b) in edges
        push!(adj[a], b); push!(adj[b], a)
    end
    return labels, adj
end

nedges(adj) = sum(length, adj) ÷ 2

# Is the graph chordal? MCS order + perfect elimination check (Tarjan-Yannakakis).
function is_chordal(adj::Vector{BitSet})
    n = length(adj); n == 0 && return true
    # maximum cardinality search with buckets
    w = zeros(Int, n); numbered = falses(n); order = zeros(Int, n)   # order[i] = vertex numbered i (from n down to 1)
    buckets = [BitSet() for _ in 0:n]; union!(buckets[1], 1:n); top = 0
    for i in n:-1:1
        while isempty(buckets[top + 1]); top -= 1; end
        v = first(buckets[top + 1]); delete!(buckets[top + 1], v)
        numbered[v] = true; order[i] = v
        for u in adj[v]
            numbered[u] && continue
            delete!(buckets[w[u] + 1], u); w[u] += 1; push!(buckets[w[u] + 1], u)
            top = max(top, w[u])
        end
    end
    # order[1..n] is a PEO iff chordal: for each v, its later neighbours minus the first one are adjacent to that one
    pos = zeros(Int, n); for (i, v) in enumerate(order); pos[v] = i; end
    for v in order
        later = [u for u in adj[v] if pos[u] > pos[v]]
        isempty(later) && continue
        p = later[argmin([pos[u] for u in later])]
        for u in later
            u == p || u in adj[p] || return false
        end
    end
    return true
end

# Check a fill edge list (label pairs) for an input graph: each edge new, no duplicates, result chordal.
function check_solution(labels, adj, fill_labels::Vector{Tuple{String, String}})
    id = Dict(l => i for (i, l) in enumerate(labels))
    h = deepcopy(adj); seen = Set{Tuple{Int, Int}}()
    for (a, b) in fill_labels
        (haskey(id, a) && haskey(id, b)) || return (ok = false, k = -1, msg = "unknown vertex $a or $b")
        u, v = minmax(id[a], id[b])
        u == v && return (ok = false, k = -1, msg = "loop $a")
        v in adj[u] && return (ok = false, k = -1, msg = "existing edge $a $b")
        (u, v) in seen && return (ok = false, k = -1, msg = "duplicate $a $b")
        push!(seen, (u, v)); push!(h[u], v); push!(h[v], u)
    end
    is_chordal(h) || return (ok = false, k = length(seen), msg = "not chordal")
    return (ok = true, k = length(seen), msg = "ok")
end

function read_fill(path::AbstractString)
    out = Tuple{String, String}[]
    isfile(path) || return out
    for line in eachline(path)
        f = split(strip(line))
        length(f) >= 2 && !startswith(f[1], '#') && push!(out, (String(f[1]), String(f[2])))
    end
    return out
end
