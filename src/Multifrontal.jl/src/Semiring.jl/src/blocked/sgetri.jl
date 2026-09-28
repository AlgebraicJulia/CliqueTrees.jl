# ===== sgetri! =====

function sgetri!(
        s::AbstractSemiring,
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        Bptr::AbstractVector{I},
        Fptr::AbstractVector{I},
        nBptr::I,
        Nptr::AbstractVector{I},
        Ntgt::AbstractVector{I},
        Nval::AbstractVector{T},
        C::AbstractMatrix;
        nt::Integer = nthreads(),
    ) where {T, I}
    n = convert(I, size(C, 2))

    W = DivisionWorkspace{T}(L.S, n)
    pool = spool_mt(T, nt)

    sgetri_mt!(s, L, U, C, W, pool, nt)

    for c in reverse(oneto(nBptr))
        fstrt = Fptr[c]
        fstop = Fptr[c + one(I)] - one(I)

        jstrt = Bptr[c]
        jstop = Bptr[c + one(I)] - one(I)

        Cc = view(C, one(I):n, jstrt:n)
        Cd = view(C, one(I):n, jstop + one(I):n)

        if jstop < n
            sgetrs_mt!(s, Val(:L), Val(:N), L, U, Cd, W, pool, nt, fstrt, fstop)
        end

        sgemx_sparse!(s, Val(:N), Val(:N), Cc, Nptr, Ntgt, Nval, jstrt, jstop, Cc, nt)
    end

    return C
end
