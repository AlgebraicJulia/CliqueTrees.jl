module AMDExt

using AMD: AMD as AMDLib
using CliqueTrees
using Graphs

function CliqueTrees.permutation(graph, alg::Union{AMD, SymAMD})
    return permutation(BipartiteGraph(graph), alg)
end

function CliqueTrees.permutation(graph::BipartiteGraph{V}, alg::AMD) where {V}
    # set parameters
    meta = AMDLib.Amd()
    meta.control[AMDLib.AMD_DENSE] = alg.dense
    meta.control[AMDLib.AMD_AGGRESSIVE] = alg.aggressive

    # construct AMD graph
    nrow = convert(Cint, nv(graph))
    colptr = Vector{Cint}(pointers(graph))
    rowval = Vector{Cint}(targets(graph))
    colptr .-= one(Cint) # 0-based indexing
    rowval .-= one(Cint) # 0-based indexing

    # construct permutation
    p = zeros(Cint, nv(graph))
    AMDLib.amd_order(nrow, colptr, rowval, p, meta.control, meta.info)
    p .+= one(Cint) # 1-based indexing

    # restore vertex type
    order::Vector{V} = p
    return order, invperm(order)
end

function CliqueTrees.permutation(graph::BipartiteGraph{Int64}, alg::AMD)
    # set parameters
    meta = AMDLib.Amd()
    meta.control[AMDLib.AMD_DENSE] = alg.dense
    meta.control[AMDLib.AMD_AGGRESSIVE] = alg.aggressive

    # construct AMD graph
    nrow = nv(graph)
    colptr = Vector{Int64}(pointers(graph))
    rowval = Vector{Int64}(targets(graph))
    colptr .-= one(Int64) # 0-based indexing
    rowval .-= one(Int64) # 0-based indexing

    # construct permutation
    p = zeros(Int64, nv(graph))
    AMDLib.amd_l_order(nrow, colptr, rowval, p, meta.control, meta.info)
    p .+= one(Int64) # 1-based indexing

    # restore vertex type
    order = p
    return order, invperm(order)
end

function CliqueTrees.permutation(graph::BipartiteGraph{V}, alg::SymAMD) where {V}
    # set parameters
    meta = AMDLib.Colamd{Cint}()
    meta.knobs[AMDLib.COLAMD_DENSE_ROW] = alg.dense_row
    meta.knobs[AMDLib.COLAMD_DENSE_COL] = alg.dense_col
    meta.knobs[AMDLib.COLAMD_AGGRESSIVE] = alg.aggressive

    # construct AMD graph
    nrow = convert(Cint, nv(graph))
    colptr = Vector{Cint}(pointers(graph))
    rowval = Vector{Cint}(targets(graph))
    colptr .-= one(Cint) # 0-based indexing
    rowval .-= one(Cint) # 0-based indexing

    # construct permutation
    p = zeros(Cint, nv(graph) + 1)
    cfun_calloc = @cfunction(Base.Libc.calloc, Ptr{Cvoid}, (Cint, Cint))
    cfun_free = @cfunction(Base.Libc.free, Cvoid, (Ptr{Cvoid},))
    AMDLib.symamd(nrow, rowval, colptr, p, meta.knobs, meta.stats, cfun_calloc, cfun_free)
    p .+= one(Cint) # 1-based indexing

    # restore vertex type
    order::Vector{V} = @view p[begin:(end - 1)]
    return order, invperm(order)
end

function CliqueTrees.permutation(graph::BipartiteGraph{Int64}, alg::SymAMD)
    # set parameters
    meta = AMDLib.Colamd{Int64}()
    meta.knobs[AMDLib.COLAMD_DENSE_ROW] = alg.dense_row
    meta.knobs[AMDLib.COLAMD_DENSE_COL] = alg.dense_col
    meta.knobs[AMDLib.COLAMD_AGGRESSIVE] = alg.aggressive

    # construct AMD graph
    nrow = nv(graph)
    colptr = Vector{Int64}(pointers(graph))
    rowval = Vector{Int64}(targets(graph))
    colptr .-= one(Int64) # 0-based indexing
    rowval .-= one(Int64) # 0-based indexing

    # construct permutation
    p = zeros(Int64, nv(graph) + 1)
    cfun_calloc = @cfunction(Base.Libc.calloc, Ptr{Cvoid}, (Int64, Int64))
    cfun_free = @cfunction(Base.Libc.free, Cvoid, (Ptr{Cvoid},))
    AMDLib.symamd_l(nrow, rowval, colptr, p, meta.knobs, meta.stats, cfun_calloc, cfun_free)
    p .+= one(Int64) # 1-based indexing

    # restore vertex type
    order = p[begin:(end - 1)]
    return order, invperm(order)
end

end
