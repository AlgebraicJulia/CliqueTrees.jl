# shared helpers for the ND scripts: graph loading, CSR, fill evaluation (sbench.jl fillq), CPU leaf solvers
using CliqueTrees, Graphs, SparseArrays, Serialization
const NDDIR = @__DIR__
const RDIR = joinpath(NDDIR, "..")
isdefined(Main, :lat2) || include(joinpath(RDIR, "gens.jl"))

const DATADIRS = split(get(ENV, "DATA", joinpath(RDIR, "..", "fill", "data")), ":")
const HAVEMM = Base.find_package("MatrixMarket") !== nothing
HAVEMM && @eval using MatrixMarket

# sorted CSR without self loops: (ptr::Vector{Int64}, adj::Vector{Int32})
function graph_csr(g::SimpleGraph)
    fl = g.fadjlist; n = length(fl)
    ptr = Vector{Int64}(undef, n + 1); ptr[1] = 1
    for v in 1:n
        ptr[v + 1] = ptr[v] + length(fl[v]) - (insorted(v, fl[v]) ? 1 : 0)
    end
    adj = Vector{Int32}(undef, ptr[n + 1] - 1); k = 1
    for v in 1:n, u in fl[v]
        u == v && continue
        adj[k] = u; k += 1
    end
    return ptr, adj
end

function sparse_csr(B::SparseMatrixCSC)
    # B symmetric pattern; drop diagonal, sorted rows
    n = size(B, 1)
    ptr = Vector{Int64}(undef, n + 1); ptr[1] = 1
    for v in 1:n
        c = 0
        for q in B.colptr[v]:(B.colptr[v + 1] - 1)
            B.rowval[q] != v && (c += 1)
        end
        ptr[v + 1] = ptr[v] + c
    end
    adj = Vector{Int32}(undef, ptr[n + 1] - 1); k = 1
    for v in 1:n, q in B.colptr[v]:(B.colptr[v + 1] - 1)
        u = B.rowval[q]
        u == v && continue
        adj[k] = u; k += 1
    end
    return ptr, adj
end

csr_graph(ptr, adj) = (n = length(ptr) - 1; SimpleGraph(SparseMatrixCSC(n, n, collect(Int, ptr), collect(Int, adj), ones(Bool, length(adj)))))

# load: name.bin (n, I, J serialized), name.mtx, lat2_L, lat3_L; returns (ptr, adj)
function load_csr(name::AbstractString)
    for d in DATADIRS
        f = joinpath(d, name * ".bin")
        if isfile(f)
            n, I, J = deserialize(f)
            B = sparse(vcat(I, J), vcat(J, I), true, n, n, |)
            return sparse_csr(B)
        end
        f = joinpath(d, name * ".mtx")
        if isfile(f) && HAVEMM
            A = mmread(f); B = SparseMatrixCSC(A .!= 0); B = B .| B'
            return sparse_csr(B)
        end
    end
    startswith(name, "lat2_") && return graph_csr(lat2(parse(Int, name[6:end])))
    startswith(name, "lat3_") && return graph_csr(lat3(parse(Int, name[6:end])))
    error("no data for $name")
end

# nnz(L) (diagonal included), flops (sbench definition), tw: supernodal recount with CliqueTrees
function fillq(g, o)
    n = nv(g)
    @assert length(o) == n && isperm(o) "invalid permutation"
    n == 0 && return 0, 0.0, -1
    _, tree = cliquetree(g; alg = collect(Int, o))
    nz = 0; fl = 0.0; mx = 0
    for bag in tree
        r = length(residual(bag)); s = length(separator(bag))
        nz += r * s + (r * (r + 1)) >> 1
        fl += r * Float64(s)^2 + Float64(s) * r * (r + 1) + r * (r + 1) * (2r + 1) / 6
        mx = max(mx, r + s)
    end
    return nz, fl, mx - 1
end

# CPU leaf solver: an elimination algorithm on every leaf (local CSR)
function cpu_leafsolve(alg)
    return csrs -> begin
        out = Vector{Vector{Int32}}(undef, length(csrs))
        for (i, (p, t)) in enumerate(csrs)
            n = length(p) - 1
            if n <= 1
                out[i] = Int32.(1:n)
            else
                A = SparseMatrixCSC(n, n, collect(Int, p), collect(Int, t), ones(length(t)))
                out[i] = Int32.(permutation(A; alg = alg)[1])
            end
        end
        out
    end
end
natural_leafsolve(csrs) = [Int32.(1:(length(p) - 1)) for (p, _) in csrs]
