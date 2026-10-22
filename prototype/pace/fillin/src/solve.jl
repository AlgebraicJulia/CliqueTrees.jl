# Whole-instance driver: read, preprocess (prep.jl), solve each atom exactly, assemble, check.
# Load either fillref.jl (CPU) or gpufill.jl (GPU) first; `decide` picks the decision procedure.
include("graphio.jl")
include("prep.jl")
include("brute.jl")

function solve_instance(path::AbstractString, decide; tlimit = 30.0, verbose = false, outpath = nothing, kw...)
    t0 = time(); deadline = t0 + tlimit
    labels, adj = read_graph(path)
    atoms, forced, G = preprocess(adj; verbose)
    tprep = time() - t0
    fill = Tuple{Int, Int}[]; append!(fill, forced)
    status = :found; nat = length(atoms); maxat = isempty(atoms) ? 0 : maximum(length, atoms); runs = 0
    # small atoms first
    for verts in sort(atoms; by = length)
        adjl = atom_graph(G, verts)
        if length(verts) > 64 * 8
            status = :toolarge; break
        end
        r = solve_atom(adjl, decide; deadline, verbose, kw...)
        runs += r.runs
        if r.status != :found
            status = r.status; break
        end
        for (a, b) in r.fill
            push!(fill, minmax(verts[a], verts[b]))
        end
        time() > deadline && (status = :timeout; break)
    end
    t = time() - t0
    fl = [(labels[a], labels[b]) for (a, b) in fill]
    chk = status == :found ? check_solution(labels, adj, fl) : (ok = false, k = -1, msg = string(status))
    if outpath !== nothing && status == :found
        open(outpath, "w") do io
            for (a, b) in fl
                println(io, a, " ", b)
            end
        end
    end
    return (; status, k = status == :found ? length(fill) : -1, t, tprep, n = length(adj), m = nedges(adj), atoms = nat, maxatom = maxat,
        forced = length(forced), runs, check = chk)
end

# Compile every code path for set widths 1..maxW: a dense random 12-vertex core plus a pendant path
# (so the graph needs W words), decided at K = OPT - 1 (NO, phases) and K = OPT (YES, root, rebuild).
function warmup(decide; maxW = 8)
    for W in 1:maxW
        n = 64 * (W - 1) + 12
        core = [[1, 2], [1, 3], [2, 4], [3, 4], [3, 5], [4, 6], [5, 7], [6, 7], [5, 8], [7, 9], [8, 9], [8, 10], [9, 11], [10, 11], [10, 12], [11, 12], [1, 12], [2, 6]]
        adjl = [Int[] for _ in 1:n]
        for (a, b) in core; push!(adjl[a], b); push!(adjl[b], a); end
        for v in 13:n; push!(adjl[v - 1], v); push!(adjl[v], v - 1); end
        foreach(sort!, adjl)
        opt = brute_minfill([filter(<=(12), a) for a in adjl[1:12]])
        pg = relabel(adjl, ones(n), vorder(adjl, :rbfs))
        decide(pg, opt - 1); r = decide(pg, opt)
        r.yes && r.opt == opt || error("warmup W=$W: $(r.opt) != $opt")
        solve_atom(adjl, decide)
    end
end
