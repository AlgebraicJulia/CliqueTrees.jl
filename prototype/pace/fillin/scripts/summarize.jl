# Check every solver's output (chordal completion, new edges only, no duplicates; count = k) and
# tabulate: julia --project=env scripts/summarize.jl <tlimit>  -> results/summary.tsv + stdout counts
include(joinpath(@__DIR__, "..", "src", "graphio.jl"))
D = joinpath(@__DIR__, ".."); R = joinpath(D, "results"); tl = parse(Float64, get(ARGS, 1, "30"))
files = readlines(joinpath(D, "data", "all.txt"))
solvers = [("tamaki", "sota_tamaki.tsv"), ("laakeri", "sota_laakeri.tsv"), ("cpu", "cpu30.tsv"), ("gpu", "gpu30.tsv")]
function load(sv, fn)
    p = joinpath(R, fn); out = Dict{String, Tuple{Bool, Float64}}()
    isfile(p) || return out
    for l in eachline(p)
        f = split(l, '\t')
        if sv in ("tamaki", "laakeri")
            out[f[1]] = (f[2] == "OK", parse(Float64, f[4]))
        else
            length(f) >= 12 || continue
            out[f[1]] = (f[7] == "found" && parse(Float64, f[9]) <= tl, parse(Float64, f[9]))
        end
    end
    return out
end
tabs = Dict(sv => load(sv, fn) for (sv, fn) in solvers)
io = open(joinpath(R, "summary.tsv"), "w")
println(io, join(["name", "set", "n", "m", ["$(sv)_$x" for (sv, _) in solvers for x in ("k", "t")]..., "agree"], '\t'))
cnt = Dict((sv, set) => 0 for (sv, _) in solvers for set in ("public", "hidden")); bad = String[]; disagree = String[]
ran = Dict((sv, set) => 0 for (sv, _) in solvers for set in ("public", "hidden"))
for f in files
    name = basename(f); set = occursin("hidden", f) ? "hidden" : "public"
    labels, adj = read_graph(f)
    row = Any[name, set, length(adj), nedges(adj)]; ks = Int[]
    for (sv, _) in solvers
        r = get(tabs[sv], name, nothing)
        r === nothing && (push!(row, "-", "-"); continue)
        ran[(sv, set)] += 1
        ok, t = r
        if ok
            c = check_solution(labels, adj, read_fill(joinpath(R, "sol_$sv", name * ".fill")))
            if c.ok
                push!(row, c.k, round(t; digits = 2)); push!(ks, c.k); cnt[(sv, set)] += 1
            else
                push!(row, "BAD:" * c.msg, round(t; digits = 2)); push!(bad, "$sv $name $(c.msg)")
            end
        else
            push!(row, "TO", round(t; digits = 2))
        end
    end
    ag = length(unique(ks)) <= 1
    ag || push!(disagree, "$name $(ks)")
    push!(row, ag ? "yes" : "NO")
    println(io, join(row, '\t'))
end
close(io)
for (sv, _) in solvers
    println(rpad(sv, 8), " hidden ", cnt[(sv, "hidden")], "/", ran[(sv, "hidden")], "  public ", cnt[(sv, "public")], "/", ran[(sv, "public")])
end
println("invalid outputs: ", isempty(bad) ? "none" : bad)
println("value disagreements: ", isempty(disagree) ? "none" : disagree)
