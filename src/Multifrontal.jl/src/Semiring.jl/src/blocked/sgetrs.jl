# ===== sgetrs! =====

function sgetrs!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        Bptr::AbstractVector{I},
        Fptr::AbstractVector{I},
        nBptr::I,
        Nptr::AbstractVector{I},
        Ntgt::AbstractVector{I},
        Nval::AbstractVector{T},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
    ) where {T, I, SIDE, TRANS}
    if isforward(:U, TRANS, SIDE)
        for c in oneto(nBptr)
            fstrt = Fptr[c]
            fstop = Fptr[c + one(I)] - one(I)

            jstrt = Bptr[c]
            jstop = Bptr[c + one(I)] - one(I)

            if SIDE === :L
                sgemx_sparse!(s, trans, Val(:N), B, Nptr, Ntgt, Nval, jstrt, jstop, B, nt)
            else
                sgemx_sparse!(s, Val(:N), trans, B, B, Nptr, Ntgt, Nval, jstrt, jstop, nt)
            end

            sgetrs_mt!(s, side, trans, L, U, B, W, pool, nt, fstrt, fstop)
        end
    else
        for c in reverse(oneto(nBptr))
            fstrt = Fptr[c]
            fstop = Fptr[c + one(I)] - one(I)

            jstrt = Bptr[c]
            jstop = Bptr[c + one(I)] - one(I)

            sgetrs_mt!(s, side, trans, L, U, B, W, pool, nt, fstrt, fstop)

            if SIDE === :L
                sgemx_sparse!(s, trans, Val(:N), B, Nptr, Ntgt, Nval, jstrt, jstop, B, nt)
            else
                sgemx_sparse!(s, Val(:N), trans, B, B, Nptr, Ntgt, Nval, jstrt, jstop, nt)
            end
        end
    end

    return B
end
