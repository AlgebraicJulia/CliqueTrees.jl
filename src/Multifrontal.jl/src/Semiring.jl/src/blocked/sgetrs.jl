# ===== sgetrs_mt! =====

function sgetrs_mt!(
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
        pool,
        nt::Integer,
    ) where {T, I, SIDE, TRANS}
    S = L.S

    if B isa AbstractVector
        nrhs = 1
    elseif SIDE === :L
        nrhs = size(B, 2)
    else
        nrhs = size(B, 1)
    end

    if B isa AbstractMatrix && nt > 1 && nrhs >= 4nt
        tsize = fld(nrhs, nt)

        @threads for t in 1:nt
            tstrt = (t - 1) * tsize + 1

            if t < nt
                tstop = t * tsize
            else
                tstop = nrhs
            end

            trhs = tstop - tstrt + 1

            if SIDE === :L
                Bt = view(B, :, tstrt:tstop)
            else
                Bt = view(B, tstrt:tstop, :)
            end

            Wt = DivisionWorkspace{T}(S, trhs)
            poolt = spool_mt(s, T, 1)
            sgetrs_mt!(s, side, trans, L, U, Bptr, Fptr, nBptr, Nptr, Ntgt, Nval, Bt, Wt, poolt, 1)
        end
    else
        W = DivisionWorkspace{T}(S, nrhs)
        sgetrs_mt!(s, side, trans, L, U, Bptr, Fptr, nBptr, Nptr, Ntgt, Nval, B, W, pool, nt)
    end

    return B
end

function sgetrs_mt!(
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
