# GPU PID-BT on the PACE 2017 exact instances (unit weights).
#   julia -t 2 --project=<env> run_pace_gpu.jl <out.tsv> <tlimit s> <ex001 ex002 ... | @file | all> [-- ]
# Env: PACE_DATA (default ../data/plain), GPUPID_SRC (default ../../gpupid/src), PACE_TD_DIR (write .td files here),
#      PACE_VALIDATE (path of td-validate; if set and the .td is written, the .td is validated, time not charged).
#
# Pipeline per instance (the paper's preprocessing): SafeRules(SafeSeparators(<atom solver>, 4 minimal triangulations))
# splits the graph into atoms (reduction rules + safe separators). The atom solver is GPU PID-BT: for each atom,
#   * MF upper bound and MMW lower bound; if they meet, the atom is solved without the GPU ("trivial");
#   * otherwise PGraph on a reversed-BFS vertex order, gpu_tw(pg, lo) with the remaining time budget.
# CliqueTrees stitches the atom orders into one elimination order of the whole graph, so the reported width is the
# treewidth of an actual ordering (the max over atoms, trivial atoms and rule-eliminated vertices included), and a
# .td can be written and checked with td-validate. Status optimal iff every non-trivial atom returned :found.
# Time charged = wall time of preprocessing + GPU solves; GPU kernel compilation (warm-up per bitset width) is not charged.
# Every width is checked against the published optimum (data/plain/exact_td/*.td header); a mismatch is status MISMATCH
# and is never a solve.
using CliqueTrees, Graphs, Printf
const CT = CliqueTrees
const HERE = @__DIR__
const NT = parse(Int, get(ENV, "SEPTW_NT", "1"))
const DATA = get(ENV, "PACE_DATA", joinpath(HERE, "..", "data", "plain"))
const TDDIR = get(ENV, "PACE_TD_DIR", "")
const VALIDATE = get(ENV, "PACE_VALIDATE", "")

# ---------------------------------------------------------------- I/O
function readgr(path)
    n = 0; edges = Tuple{Int, Int}[]
    for line in eachline(path)
        isempty(line) && continue
        c = line[1]; c == 'c' && continue
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

optimum(name) = parse(Int, split(readline(joinpath(DATA, "exact_td", name * ".td")))[4]) - 1

# elimination order -> tree decomposition (bags {v} U higher neighbours), as in tools/ctrun.jl
function writetd(path, g, order)
    n = nv(g); pos = zeros(Int, n); pos[order] = 1:n
    adj = [sort!([pos[w] for w in neighbors(g, order[i])]) for i in 1:n]
    bags = Vector{Vector{Int}}(undef, n); parent = zeros(Int, n)
    higher = [filter(>(i), adj[i]) for i in 1:n]
    for i in 1:n
        hi = higher[i]; bags[i] = vcat(i, hi)
        if !isempty(hi)
            p = hi[1]; parent[i] = p
            higher[p] = sort!(unique!(vcat(higher[p], @view hi[2:end])))
        end
    end
    roots = [i for i in 1:n if parent[i] == 0]
    w = maximum(length, bags) - 1
    open(path, "w") do io
        println(io, "s td $n $(w + 1) $n")
        for i in 1:n; println(io, "b $i ", join((order[j] for j in bags[i]), " ")); end
        for i in 1:n; parent[i] != 0 && println(io, "$i $(parent[i])"); end
        for k in 2:length(roots); println(io, "$(roots[k-1]) $(roots[k])"); end
    end
    return w
end

# ---------------------------------------------------------------- the atom solver
# one record per atom handed to the solver
const ATOMS = NamedTuple{(:n, :m, :status, :width, :t), Tuple{Int, Int, Symbol, Float64, Float64}}[]
const DEADLINE = Ref(Inf)       # absolute time at which the charged budget ends
const WARM = Ref(0.0)           # seconds spent compiling (not charged)
const WARMED = Set{Int}()
const ERRORS = String[]
const STICKY = Ref(false)

sticky(e) = (m = sprint(showerror, e); occursin("ILLEGAL", m) || occursin("KernelException", m) || occursin("launch failure", m) || (occursin("ERROR_", m) && !occursin("OUT_OF_MEMORY", m)))
remaining() = DEADLINE[] + WARM[] - time()

function warmup(W)
    W in WARMED && return
    t0 = time()
    n = 64 * (W - 1) + 10
    g = cycle_graph(n); add_edge!(g, 1, n ÷ 2)
    pg = PGraph([sort(neighbors(g, v)) for v in 1:n], [isodd(v) ? 1.0 : 2.0 for v in 1:n]); gpu_tw(pg, 2.0)
    push!(WARMED, W); WARM[] += time() - t0
end

struct GPUExact <: CT.EliminationAlgorithm end
CT.isconcurrent(::GPUExact) = false

function CT.permutation(weights::AbstractVector, graph::AbstractGraph, ::GPUExact; verbose::Bool = false)
    n = nv(graph); m = ne(graph); t0 = time()
    ordmf, _ = CT.permutation(weights, graph, MF())
    ordmf = collect(Int, ordmf)
    ub = treewidth(weights, graph, ordmf)
    lb = lowerbound(weights, graph, MMW())
    fin(status, w, order) = (push!(ATOMS, (; n, m, status, width = Float64(w), t = time() - t0)); (order, invperm(order)))
    lb >= ub - 1e-9 && return fin(:trivial, ub, ordmf)
    # CPU septw on the atom (ablation of the GPU run: same pipeline, CPU exact solver)
    remaining() <= 0 && return fin(:timeout, ub, ordmf)
    perm, _, k, st = CT.septw(round.(Int, weights), graph, 0, typemax(Int), time() + remaining(), NT)
    if st == CT.BT_FOUND
        order = collect(Int, perm); w = treewidth(weights, graph, order)
        return w <= ub ? fin(:found, w, order) : fin(:error, ub, ordmf)
    end
    return fin(st == CT.BT_TIMEOUT ? :timeout : :unsolved, ub, ordmf)
end

atompipe() = SafeRules(SafeSeparators(GPUExact(),
    (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()));
    maxmissing = 50))

# ---------------------------------------------------------------- driver
function solve_instance(name, tl)
    g = readgr(joinpath(DATA, "exact", name * ".gr")); n = nv(g); m = ne(g); opt = optimum(name)
    empty!(ATOMS); empty!(ERRORS); STICKY[] = false; WARM[] = 0.0
    t0 = time(); DEADLINE[] = t0 + tl
    order = Int[]; err = ""
    try
        order, _ = CT.permutation(ones(Float64, n), g, atompipe())
        order = collect(Int, order)
    catch e
        err = replace(sprint(showerror, e)[1:min(end, 200)], '\n' => ' ')
        sticky(e) && (STICKY[] = true)
    end
    t = time() - t0 - WARM[]
    isempty(ERRORS) || (err = join(unique(ERRORS), " | "))
    natoms = length(ATOMS); maxn = maximum((a.n for a in ATOMS); init = 0)
    width = -1; valid = "-"
    if !isempty(order)
        width = Int(treewidth(ones(Float64, n), g, order)) - 1
        if !isempty(TDDIR)
            mkpath(TDDIR); td = joinpath(TDDIR, name * ".td")
            w2 = writetd(td, g, order); w2 == width || (err *= " TDWIDTH($w2)")
            if !isempty(VALIDATE)
                p = run(pipeline(ignorestatus(`$VALIDATE $(joinpath(DATA, "exact", name * ".gr")) $td`); stdout = devnull, stderr = devnull))
                valid = p.exitcode == 0 ? "y" : "NOT_VALID"
            end
        end
    end
    allsolved = !isempty(order) && all(a -> a.status in (:trivial, :found), ATOMS) && isempty(err)
    status = allsolved ? "optimal" : (any(a -> a.status == :oom, ATOMS) ? "oom" : any(a -> a.status == :error, ATOMS) || !isempty(err) ? "error" : "timeout")
    if width >= 0 && ((allsolved && width != opt) || width < opt || valid == "NOT_VALID")
        status = "MISMATCH"
        err = "MISMATCH width=$width optimum=$opt valid=$valid " * err
        printstyled(stderr, "!!!!!! $name: $err\n"; color = :red, bold = true)
    end
    return (; name, n, m, natoms, maxn, width, opt, status, t, valid, err = isempty(err) ? "-" : err)
end

function main()
    out = ARGS[1]; tl = parse(Float64, ARGS[2])
    names = ARGS[3] == "all" ? [@sprintf("ex%03d", i) for i in 1:200] : startswith(ARGS[3], "@") ? filter(!isempty, strip.(readlines(ARGS[3][2:end]))) : ARGS[3:end]
    done = isfile(out) ? Set(first(split(l, '\t')) for l in eachline(out) if !isempty(l)) : Set{String}()
    names = filter(k -> !(k in done), names)
    # compile the pipeline once on a tiny graph (not charged)
    for w in ("ex070", "ex001", "ex013")   # JIT warm-up of the pipeline and the GPU path (uncharged)
        g = readgr(joinpath(DATA, "exact", w * ".gr")); DEADLINE[] = Inf; WARM[] = 0.0
        CT.permutation(ones(Float64, nv(g)), g, atompipe())
    end
    new = !isfile(out) || filesize(out) == 0
    io = open(out, "a")
    new && (println(io, "instance\tn\tm\tnatoms\tmax_atom_n\twidth\toptimum\tstatus\ttime\tvalid\terror"); flush(io))
    for name in names
        r = solve_instance(name, tl)
        @printf(io, "%s\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%.3f\t%s\t%s\n", r.name, r.n, r.m, r.natoms, r.maxn, r.width, r.opt, r.status, r.t, r.valid, r.err)
        flush(io)
        println(r.name, " ", r.status, " ", round(r.t; digits = 2), " w=", r.width, " opt=", r.opt, " atoms=", r.natoms, " maxn=", r.maxn); flush(stdout)
        STICKY[] && (println("sticky CUDA error, exiting for restart"); exit(3))
        GC.gc(true)
    end
    close(io)
end
main()
