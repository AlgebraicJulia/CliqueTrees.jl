module AMDExt

using Base: oneto
using CliqueTrees
using CliqueTrees.Utilities
using Graphs

import AMD as AMDLib

const AMD_DENSE = AMDLib.AMD_DENSE
const AMD_AGGRESSIVE = AMDLib.AMD_AGGRESSIVE
const COLAMD_DENSE_ROW = AMDLib.COLAMD_DENSE_ROW
const COLAMD_DENSE_COL = AMDLib.COLAMD_DENSE_COL
const COLAMD_AGGRESSIVE = AMDLib.COLAMD_AGGRESSIVE

const VECTOR{T} = Union{Vector{T}, FVector{T}}

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph, alg::Union{AMD, SymAMD})
    order = amd(graph, alg)
    return order, invperm(order)
end

function amd(graph::AbstractGraph{V}, alg::Union{AMD, SymAMD}) where {V}
    new = BipartiteGraph{Cint, Cint}(graph)
    order = convert(Vector{V}, amd(new, alg))
    return order
end

function amd(graph::AbstractGraph{Int64}, alg::Union{AMD, SymAMD})
    new = BipartiteGraph{Int64, Int64}(graph)
    return amd(new, alg)
end

function amd(graph::BipartiteGraph{Cint, Cint, <:VECTOR{Cint}, <:VECTOR{Cint}}, alg::AMD)
    n = nv(graph); m = ne(graph); nn = n + one(Cint)

    # set parameters
    meta = AMDLib.Amd()
    setmeta!(meta, alg)

    # construct AMD graph
    xadj = pointers(graph)
    adjncy = targets(graph)

    @inbounds for v in oneto(nn)
        xadj[v] -= one(Cint)
    end

    @inbounds for v in oneto(m)
        adjncy[v] -= one(Cint)
    end

    # construct permutation
    order = zeros(Cint, n)
    AMDLib.amd_order(n, xadj, adjncy, order, meta.control, meta.info)

    @inbounds for v in oneto(n)
        xadj[v] += one(Cint)
        order[v] += one(Cint)
    end

    xadj[nn] += one(Cint)

    @inbounds for v in oneto(m)
        adjncy[v] += one(Cint)
    end

    return order
end


function amd(graph::BipartiteGraph{Int64, Int64, <:VECTOR{Int64}, <:VECTOR{Int64}}, alg::AMD)
    n = nv(graph); m = ne(graph); nn = n + one(Int64)

    # set parameters
    meta = AMDLib.Amd()
    setmeta!(meta, alg)

    # construct AMD graph
    xadj = pointers(graph)
    adjncy = targets(graph)

    @inbounds for v in oneto(nn)
        xadj[v] -= one(Int64)
    end

    @inbounds for v in oneto(m)
        adjncy[v] -= one(Int64)
    end

    # construct permutation
    order = zeros(Int64, n)
    AMDLib.amd_l_order(n, xadj, adjncy, order, meta.control, meta.info)

    @inbounds for v in oneto(n)
        xadj[v] += one(Int64)
        order[v] += one(Int64)
    end

    xadj[nn] += one(Int64)

    @inbounds for v in oneto(m)
        adjncy[v] += one(Int64)
    end

    return order
end

function amd(graph::BipartiteGraph{Cint, Cint, <:VECTOR{Cint}, <:VECTOR{Cint}}, alg::SymAMD)
    n = nv(graph); m = ne(graph); nn = n + one(Cint)

    # set parameters
    meta = AMDLib.Colamd{Cint}()
    setmeta!(meta, alg)

    # construct AMD graph
    xadj = pointers(graph)
    adjncy = targets(graph)

    @inbounds for v in oneto(nn)
        xadj[v] -= one(Cint)
    end

    @inbounds for v in oneto(m)
        adjncy[v] -= one(Cint)
    end

    # construct permutation
    order = zeros(Cint, nn)
    cfun_calloc = @cfunction(Base.Libc.calloc, Ptr{Cvoid}, (Cint, Cint))
    cfun_free = @cfunction(Base.Libc.free, Cvoid, (Ptr{Cvoid},))
    AMDLib.symamd(n, adjncy, xadj, order, meta.knobs, meta.stats, cfun_calloc, cfun_free)

    @inbounds for v in oneto(n)
        xadj[v] += one(Cint)
        order[v] += one(Cint)
    end

    xadj[nn] += one(Cint)

    @inbounds for v in oneto(m)
        adjncy[v] += one(Cint)
    end

    return resize!(order, n)
end

function amd(graph::BipartiteGraph{Int64, Int64, <:VECTOR{Int64}, <:VECTOR{Int64}}, alg::SymAMD)
    n = nv(graph); m = ne(graph); nn = n + one(Int64)

    # set parameters
    meta = AMDLib.Colamd{Int64}()
    setmeta!(meta, alg)

    # construct AMD graph
    xadj = pointers(graph)
    adjncy = targets(graph)

    @inbounds for v in oneto(nn)
        xadj[v] -= one(Int64)
    end

    @inbounds for v in oneto(m)
        adjncy[v] -= one(Int64)
    end

    # construct permutation
    order = zeros(Int64, nn)
    cfun_calloc = @cfunction(Base.Libc.calloc, Ptr{Cvoid}, (Int64, Int64))
    cfun_free = @cfunction(Base.Libc.free, Cvoid, (Ptr{Cvoid},))
    AMDLib.symamd_l(n, adjncy, xadj, order, meta.knobs, meta.stats, cfun_calloc, cfun_free)

    @inbounds for v in oneto(n)
        xadj[v] += one(Int64)
        order[v] += one(Int64)
    end

    xadj[nn] += one(Int64)

    @inbounds for v in oneto(m)
        adjncy[v] += one(Int64)
    end

    return resize!(order, n)
end

function setmeta!(meta::AMDLib.Amd, alg::AMD)
    meta.control[AMD_DENSE] = alg.dense
    meta.control[AMD_AGGRESSIVE] = alg.aggressive
    return
end

function setmeta!(meta::AMDLib.Colamd, alg::SymAMD)
    meta.knobs[COLAMD_DENSE_ROW] = alg.dense_row
    meta.knobs[COLAMD_DENSE_COL] = alg.dense_col
    meta.knobs[COLAMD_AGGRESSIVE] = alg.aggressive
    return
end

end
