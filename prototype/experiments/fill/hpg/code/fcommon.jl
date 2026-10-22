# shared loader / evaluator for the large-scale fill harness (wrapper only; kernels live in code/{Rounds,GRounds,evalutil,gens}.jl)
using CliqueTrees, Graphs, SparseArrays, Serialization
const CODE = get(ENV, "CODE", @__DIR__)
const DATA = get(ENV, "DATA", "/blue/fairbanksj/itaykadosh/gpu/fill/data")
const RES = get(ENV, "RES", "/blue/fairbanksj/itaykadosh/gpu/fill/results")
include(joinpath(CODE, "evalutil.jl")); include(joinpath(CODE, "gens.jl"))

graph_from_edges(n, I, J) = (B = sparse(vcat(I, J), vcat(J, I), true, n, n, |); SimpleGraph(B))
function edges_of(g)
    I = Int32[]; J = Int32[]
    for e in edges(g); push!(I, src(e)); push!(J, dst(e)); end
    return I, J
end
function gen_graph(name)
    startswith(name, "lat2_") && return lat2(parse(Int, name[6:end]))
    startswith(name, "lat3_") && return lat3(parse(Int, name[6:end]))
    error("unknown generated instance $name")
end
function loadg(name)
    f = joinpath(DATA, name * ".bin")
    if isfile(f)
        n, I, J = deserialize(f)
        return graph_from_edges(n, I, J)
    end
    return gen_graph(name)
end
permfile(name, meth) = joinpath(DATA, "perms", name * "." * meth * ".perm")
saveperm(name, meth, o) = (mkpath(dirname(permfile(name, meth))); write(permfile(name, meth) * ".tmp", Int32.(o)); mv(permfile(name, meth) * ".tmp", permfile(name, meth); force = true))
function loadperm(name, meth, n)
    f = permfile(name, meth); isfile(f) || return nothing
    raw = Vector{Int32}(undef, n); read!(f, raw); return Int.(raw)
end
# same metrics as run.jl::rec
function evalrow(name, g, meth, o, t)
    n = nv(g); ui = ones(Int, n)
    @assert length(o) == n && isperm(o) "invalid permutation: $meth"
    cw = colweights(ui, g, o)
    return join((name, n, ne(g), meth, sum(cw), sum(Float64(c)^2 for c in cw), maximum(cw) - 1, t), '\t')
end
appendline(file, s) = (mkpath(dirname(file)); open(io -> (write(io, s * "\n"); flush(io)), file, "a"))
