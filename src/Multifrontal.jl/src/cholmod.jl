using SparseArrays.CHOLMOD: Factor
using SparseArrays.LibSuiteSparse: cholmod_factor_struct

function ChordalCholesky(F::Factor)
    return ChordalCholesky{:L}(F)
end

function ChordalCholesky{:L}(F::Factor{T, I}) where {T, I}
    fstruct = unsafe_load(F.ptr)
    ncol = convert(I, fstruct.n)
    perm0 = unsafe_wrap(Array, Ptr{I}(fstruct.Perm), ncol)

    if iszero(fstruct.is_super)
        xsize = convert(I, fstruct.nzmax)
        blkptr0 = unsafe_wrap(Array, Ptr{I}(fstruct.p), ncol + one(I))
        bagtgt0 = unsafe_wrap(Array, Ptr{I}(fstruct.i), xsize)
        blkval0 = unsafe_wrap(Array, Ptr{T}(fstruct.x), xsize)

        P, S = cholmod_symb_node(ncol, perm0, blkptr0, bagtgt0)
    else
        nfrt = convert(I, fstruct.nsuper)
        ssize = convert(I, fstruct.ssize)
        xsize = convert(I, fstruct.xsize)

        resptr0 = unsafe_wrap(Array, Ptr{I}(fstruct.super), nfrt + one(I))
        bagptr0 = unsafe_wrap(Array, Ptr{I}(fstruct.pi), nfrt + one(I))
        bagtgt0 = unsafe_wrap(Array, Ptr{I}(fstruct.s), ssize)
        blkptr0 = unsafe_wrap(Array, Ptr{I}(fstruct.px), nfrt + one(I))
        blkval0 = unsafe_wrap(Array, Ptr{T}(fstruct.x), xsize)

        P, S = cholmod_symb_supn(ncol, nfrt, perm0, resptr0, bagptr0, bagtgt0)
    end

    CF = ChordalCholesky{:L, T}(P, S)

    return cholmod_copy!(CF, blkptr0, blkval0)
end

function cholmod_symb_supn(
    ncol::I,
    nfrt::I,
    perm0::AbstractVector{I},
    resptr0::AbstractVector{I},
    bagptr0::AbstractVector{I},
    bagtgt0::AbstractVector{I},
) where {I}

    nsep = zero(I)

    for j in oneto(nfrt)
        nn = resptr0[j + one(I)] - resptr0[j]
        nj = bagptr0[j + one(I)] - bagptr0[j]
        nsep += nj - nn
    end

    res = BipartiteGraph{I, I, FVector{I},   OneTo{I}}(ncol, nfrt, ncol)
    sep = BipartiteGraph{I, I, FVector{I}, FVector{I}}(ncol, nfrt, nsep)
    perm = FVector{I}(undef, ncol)

    pointers(res)[one(I)] = one(I)
    pointers(sep)[one(I)] = one(I)
    q = one(I)

    for j in oneto(nfrt)
        vstrt = resptr0[j] + one(I)
        vstop = resptr0[j + one(I)]
        nn = vstop - vstrt + one(I)

        pstrt = bagptr0[j] + nn + one(I)
        pstop = bagptr0[j + one(I)]

        pointers(res)[j + one(I)] = pointers(res)[j] + nn

        for p in pstrt:pstop
            targets(sep)[q] = bagtgt0[p] + one(I)
            q += one(I)
        end

        pointers(sep)[j + one(I)] = q
    end

    S = ChordalSymbolic(res, sep)

    for j in oneto(ncol)
        perm[j] = perm0[j] + one(I)
    end

    return Permutation(perm), S
end

function cholmod_symb_node(
        ncol::I,
        perm0::AbstractVector{I},
        bagptr0::AbstractVector{I},
        bagtgt0::AbstractVector{I},
    ) where {I}

    nsep = zero(I)

    for j in oneto(ncol)
        pstrt = bagptr0[j] + one(I)
        pstop = bagptr0[j + one(I)]

        for p in pstrt:pstop
            v = bagtgt0[p] + one(I)

            if v > j
                nsep += pstop - p + one(I)
                break
            end
        end
    end

    res = BipartiteGraph{I, I, FVector{I},   OneTo{I}}(ncol, ncol, ncol)
    sep = BipartiteGraph{I, I, FVector{I}, FVector{I}}(ncol, ncol, nsep)
    perm = FVector{I}(undef, ncol)

    for j in oneto(ncol + one(I))
        pointers(res)[j] = j
    end

    pointers(sep)[one(I)] = one(I)
    q = one(I)

    for j in oneto(ncol)
        pstrt = bagptr0[j] + one(I)
        pstop = bagptr0[j + one(I)]

        for p in pstrt:pstop
            v = bagtgt0[p] + one(I)

            if v > j
                targets(sep)[q] = v
                q += one(I)
            end
        end

        pointers(sep)[j + one(I)] = q
    end

    S = ChordalSymbolic(res, sep)

    for j in oneto(ncol)
        perm[j] = perm0[j] + one(I)
    end

    return Permutation(perm), S
end

function cholmod_copy!(
        CF::ChordalCholesky{:L, T, I},
        blkptr0::AbstractVector{I},
        blkval0::AbstractVector{T},
    ) where {T, I}
    nfrt = nfr(CF)

    for j in oneto(nfrt)
        nn = eltypedegree(CF.S.res, j)
        na = eltypedegree(CF.S.sep, j)

        bp = blkptr0[j] + one(I)
        Dp = CF.S.Dptr[j]
        Lp = CF.S.Lptr[j]

        for c in oneto(nn)
            for r in oneto(nn)
                CF.Dval[Dp] = blkval0[bp]
                Dp += one(I)
                bp += one(I)
            end

            for r in oneto(na)
                CF.Lval[Lp] = blkval0[bp]
                Lp += one(I)
                bp += one(I)
            end
        end
    end

    return CF
end

