# ===== sgetrs! =====

function sgetrs!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat;
        nt::Integer = nthreads(),
    ) where {T, I, SIDE, TRANS}
    S = L.S

    if B isa AbstractVector
        nrhs = one(I)
        pool = nothing
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
        pool = spool_mt(T, nt)
    else
        nrhs = convert(I, size(B, 1))
        pool = spool_mt(T, nt)
    end

    W = DivisionWorkspace{T}(S, nrhs)
    return sgetrs_mt!(s, side, trans, L, U, B, W, pool, nt)
end

function sgetrs_mt!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
    ) where {T, I, SIDE, TRANS}
    if isforward(:L, TRANS, SIDE)
        strsx_mt!(s, side, trans, L, B, W, pool, nt)
        strsx_mt!(s, side, trans, U, B, W, pool, nt)
    else
        strsx_mt!(s, side, trans, U, B, W, pool, nt)
        strsx_mt!(s, side, trans, L, B, W, pool, nt)
    end

    return B
end
