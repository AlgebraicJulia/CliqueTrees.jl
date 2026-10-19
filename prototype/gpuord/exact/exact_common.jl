using CliqueTrees, Graphs, Serialization, Printf, Random
function loadrounds(dir; nmin = 2, nmax = 512)
    files = filter(f -> startswith(basename(f), "atoms_") && endswith(f, ".jls"), readdir(dir; join = true))
    out = []
    for f in sort(files), (nm, v) in sort(collect(deserialize(f)); by = first), a in v.atoms
        (nmin <= a.n <= nmax) || continue
        g = SimpleGraph(a.n); for (x, y) in a.edges; add_edge!(g, x, y); end
        for r in a.rounds
            r[2] >= 0 && push!(out, (; inst = nm, g = BipartiteGraph(g), w = a.weights, k = r[1]))
        end
    end
    return out
end
function cpu_round(r, nt)
    sep, _ = CliqueTrees.sepenum(r.w, r.g, r.k, Inf, nt)
    perm, invp, mink, st = CliqueTrees.sepdp(r.w, r.g, sep, r.k, Inf, nt)
    return collect(perm), mink, st == CliqueTrees.BT_FOUND ? :found : :above, nv(sep)
end
