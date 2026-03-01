"""
    ChordalTriangular{DIAG, UPLO, T, I, Val} <: AbstractMatrix{T}

A triangular matrix with chordal sparsity pattern. The type parameters
`DIAG` and `UPLO` specify its precise structure:

  - `DIAG`: `:N` or `:U` (non-unit / unit triangular)
  - `UPLO`: `:L` or `:U` (lower / upper triangular)

### Example

```julia

```

### Fields

   - `A.S`: symbolic factorization

"""
struct ChordalTriangular{DIAG, UPLO, T, I, Val <: AbstractVector{T}} <: AbstractMatrix{T}
    S::ChordalSymbolic{I}
    Dval::Val
    Lval::Val
end

const FChordalTriangular{DIAG, UPLO, T, I} = ChordalTriangular{DIAG, UPLO, T, I, FVector{T}}

const AdjOrTransTri{DIAG, UPLO, T, I, Val} = Union{
      Adjoint{T, ChordalTriangular{DIAG, UPLO, T, I, Val}},
    Transpose{T, ChordalTriangular{DIAG, UPLO, T, I, Val}},
}

const MaybeAdjOrTransTri{DIAG, UPLO, T, I, Val} = Union{
        AdjOrTransTri{DIAG, UPLO, T, I, Val},
    ChordalTriangular{DIAG, UPLO, T, I, Val},
}

function ChordalTriangular{DIAG, UPLO}(S::ChordalSymbolic{I}, Dval::Val, Lval::Val) where {DIAG, UPLO, I <: Integer, T, Val <: AbstractVector{T}}
    return ChordalTriangular{DIAG, UPLO, T, I, Val}(S, Dval, Lval)
end

function ChordalTriangular{UPLO}(S::ChordalSymbolic{I}) where {UPLO, I <: Integer}
    n = nv(S.res)
    Dval = Ones{Bool}(S.Dptr[n + one(I)] - one(I))
    Lval = Ones{Bool}(S.Lptr[n + one(I)] - one(I))
    return ChordalTriangular{:N, UPLO}(S, Dval, Lval)
end

function ChordalTriangular(F::ChordalFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    return ChordalTriangular{DIAG, UPLO}(F.S, F.Dval, F.Lval)
end

function Base.show(io::IO, A::T) where {T <: ChordalTriangular}
    n = size(A, 1)
    print(io, "$n×$n $T with $(nnz(A)) stored entries")
    return
end

function Base.show(io::IO, ::MIME"text/plain", A::T) where {DIAG, UPLO, T <: ChordalTriangular{DIAG, UPLO}}
    n = size(A, 1)
    println(io, "$n×$n $T with $(nnz(A)) stored entries:")

    if n < 16
        print_matrix(io, A)
    else
        showsymbolic(io, A.S, Val(UPLO))
    end

    return
end

function Base.replace_in_print_matrix(A::ChordalTriangular{DIAG, UPLO}, i::Integer, j::Integer, str::AbstractString) where {DIAG, UPLO}
    if UPLO === :L && i < j || UPLO === :U && i > j
        str = replace_with_centered_mark(str)
    end

    return str
end

function Base.copy(A::ChordalTriangular{DIAG, UPLO, T, I, Val}) where {DIAG, UPLO, T, I, Val}
    return ChordalTriangular{DIAG, UPLO, T, I, Val}(A.S, copy(A.Dval), copy(A.Lval))
end

function Base.copy(A::AdjOrTransTri{DIAG, UPLO, T, I}) where {DIAG, UPLO, T, I}
    P = parent(A)

    if UPLO === :L
        B = ChordalTriangular{DIAG, :U}(P.S, similar(P.Dval), similar(P.Lval))
    else
        B = ChordalTriangular{DIAG, :L}(P.S, similar(P.Dval), similar(P.Lval))
    end

    for j in vertices(P.S.res)
        nn = eltypedegree(P.S.res, j)
        na = eltypedegree(P.S.sep, j)

        Dp = P.S.Dptr[j]
        Lp = P.S.Lptr[j]

        DA = reshape(view(P.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)
        DB = reshape(view(B.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

        if UPLO === :L
            LA = reshape(view(P.Lval, Lp:Lp + nn * na - one(I)), na, nn)
            LB = reshape(view(B.Lval, Lp:Lp + nn * na - one(I)), nn, na)
        else
            LA = reshape(view(P.Lval, Lp:Lp + nn * na - one(I)), nn, na)
            LB = reshape(view(B.Lval, Lp:Lp + nn * na - one(I)), na, nn)
        end

        if A isa Adjoint
            adjoint!(DB, DA)
            adjoint!(LB, LA)
        else
            transpose!(DB, DA)
            transpose!(LB, LA)
        end
    end

    return B
end

# ===== Abstract Matrix Interface =====

function SparseArrays.nnz(A::ChordalTriangular)
    return nnz(A.S)
end

function Base.size(L::ChordalTriangular)
    n = convert(Int, nov(L.S.res))
    return (n, n)
end

function LinearAlgebra.istriu(::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    return UPLO === :U
end

function LinearAlgebra.istril(::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    return UPLO === :L
end

function LinearAlgebra.isposdef(A::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    posdiag(D) = all(ispositive, view(D, diagind(D)))

    if DIAG === :N
        return mapreducefront((D, L) -> posdiag(D), &, A; init=true)
    else
        return true
    end
end

function LinearAlgebra.det(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    if DIAG === :N
        return mapreducefront((D, L) -> det(D), *, A; init=one(T))
    else
        return one(T)
    end
end

function LinearAlgebra.logdet(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    if DIAG === :N
        return mapreducefront((D, L) -> logdet(D), +, A; init=zero(T))
    else
        return zero(T)
    end
end

function LinearAlgebra.logabsdet(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    addmul((a, b), (c, d)) = (a + c, b * d)

    if DIAG === :N
        return mapreducefront((D, L) -> logabsdet(D), addmul, A; init=(zero(real(T)), one(T)))
    else
        return (zero(real(T)), one(T))
    end
end

function LinearAlgebra.tr(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    if DIAG === :N
        return mapreducefront((D, L) -> tr(D), +, A; init=zero(T))
    else
        return convert(T, size(A, 1))
    end
end

function LinearAlgebra.rank(A::ChordalTriangular{DIAG, UPLO, T, I}; kw...) where {DIAG, UPLO, T, I <: Integer}
    blockrank(D) = rank(D; kw...)

    if DIAG === :N
        return mapreducefront((D, L) -> blockrank(D), +, A; init=0)
    else
        return size(A, 1)
    end
end

function LinearAlgebra.diag(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    out = Vector{T}(undef, size(A, 1))

    if DIAG === :N
        foreachfront(A) do D, L, res, sep
            out[res] .= view(D, diagind(D))
        end
    else
        out .= one(T)
    end

    return out
end

function SparseArrays.sparse(A::MaybeAdjOrTransTri{DIAG, UPLO, T, I}) where {DIAG, UPLO, T, I <: Integer}
    colptr = Vector{I}(undef, 1 + size(A, 1))
    rowval = Vector{I}(undef, nnz(parent(A)))
    nzval = Vector{T}(undef, nnz(parent(A)))

    p = zeros(I)

    foreachfront(parent(A)) do D, L, res, sep
        for w in eachindex(res)
            colptr[res[w]] = p[] + one(I)

            for v in w:length(res)
                if UPLO === :L
                    x = D[v, w]
                else
                    x = D[w, v]
                end

                if A isa Adjoint
                    x = conj(x)
                end

                p[] += one(I); rowval[p[]] = res[v]; nzval[p[]] = x
            end

            for v in eachindex(sep)
                if UPLO === :L
                    x = L[v, w]
                else
                    x = L[w, v]
                end

                if A isa Adjoint
                    x = conj(x)
                end

                p[] += one(I); rowval[p[]] = sep[v]; nzval[p[]] = x
            end
        end
    end

    colptr[end] = p[] + one(I)

    S = SparseMatrixCSC{T, I}(size(A)..., colptr, rowval, nzval)

    if (UPLO === :L && A isa AdjOrTransTri) || (UPLO === :U && A isa ChordalTriangular)
        S = copy(transpose(S))
    end

    return S
end

function Base.getindex(A::ChordalTriangular{DIAG, UPLO, T}, v::Integer, w::Integer) where {DIAG, UPLO, T}
    if DIAG !== :U || v != w
        return getflatindex(A, flatindex(A, v, w))
    else
        return one(T)
    end
end

function Base.isstored(A::ChordalTriangular{DIAG, UPLO}, v::Integer, w::Integer) where {DIAG, UPLO}
    return isstored(A.S, v, w, Val(UPLO))
end

function Base.setindex!(A::ChordalTriangular{DIAG, UPLO}, x, v::Integer, w::Integer) where {DIAG, UPLO}
    if DIAG !== :U || v != w
        return setflatindex!(A, x, flatindex(A, v, w))
    else
        error()
    end
end

function flatindex(A::ChordalTriangular{DIAG, UPLO}, v::Integer, w::Integer) where {DIAG, UPLO}
    @boundscheck checkbounds(A, v, w)
    return flatindex(A.S, v, w, Val(UPLO))
end

function flatindices(A::ChordalTriangular{DIAG, UPLO}, B::SparseMatrixCSC) where {DIAG, UPLO}
    return flatindices(A.S, B, Val(UPLO))
end

function getflatindex(A::ChordalTriangular{DIAG, UPLO, T}, p::Integer) where {DIAG, UPLO, T}
    if ispositive(p)
        return A.Dval[p]
    elseif isnegative(p)
        return A.Lval[-p]
    else
        return zero(T)
    end
end

function setflatindex!(A::ChordalTriangular, x, p::Integer)
    if ispositive(p)
        A.Dval[p] = x
    elseif isnegative(p)
        A.Lval[-p] = x
    else
        error()
    end

    return A
end

function Base.copy!(A::ChordalTriangular{DIAG, :L, T, I}, B::SparseMatrixCSC) where {DIAG, T, I}
    zerorec!(A.Dval)
    zerorec!(A.Lval)

    foreachfront(A) do D, L, res, sep
        rlo = first(res)
        rhi = last(res) + one(I)

        if !isempty(sep)
            slo = first(sep)
            shi = last(sep) + one(I)
        end

        @inbounds for j in eachindex(res)
            k = one(I)

            for p in nzrange(B, res[j])
                row = rowvals(B)[p]

                if row < rhi
                    row < rlo && continue
                    parent(D)[row - rlo + one(I), j] = nonzeros(B)[p]
                elseif !isempty(sep)
                    row < slo && continue
                    row >= shi && break

                    while sep[k] < row
                        k += one(I)
                    end

                    L[k, j] = nonzeros(B)[p]
                    k += one(I)
                end
            end
        end
    end

    return A
end

function Base.copy!(A::ChordalTriangular{DIAG, :U, T, I}, B::SparseMatrixCSC) where {DIAG, T, I}
    zerorec!(A.Dval)
    zerorec!(A.Lval)

    foreachfront(A) do D, L, res, sep
        rlo = first(res)
        rhi = last(res) + one(I)

        @inbounds for j in eachindex(res)
            for p in nzrange(B, res[j])
                row = rowvals(B)[p]
                row < rlo && continue
                row >= rhi && break
                parent(D)[row - rlo + one(I), j] = nonzeros(B)[p]
            end
        end

        @inbounds for j in eachindex(sep)
            for p in nzrange(B, sep[j])
                row = rowvals(B)[p]
                row < rlo && continue
                row >= rhi && break
                L[row - rlo + one(I), j] = nonzeros(B)[p]
            end
        end
    end

    return A
end

function Base.fill!(A::ChordalTriangular, x)
    fill!(A.Dval, x)
    fill!(A.Lval, x)
    return A
end

function LinearAlgebra.cond(F::MaybeAdjOrTransTri, p::Real)
    if p == 1
        condest1(F)
    elseif p == 2
        condest2(F)
    elseif p == Inf
        condest1(adjoint(F))
    else
        error()
    end
end

function LinearAlgebra.opnorm(A::MaybeAdjOrTransTri, p::Real)
    if p == 1
        opnorm1(A)
    elseif p == 2
        opnormest2(A)
    elseif p == Inf
        opnorminf(A)
    else
        error()
    end
end

function foreachfront(f, A::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    return foreachfront_impl(f, A.S.res, A.S.sep, A.S.Dptr, A.S.Lptr, A.Dval, A.Lval, Val(DIAG), Val(UPLO))
end

function foreachfront_impl(
        f,
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        diag::Val{DIAG},
        uplo::Val{UPLO},
    ) where {DIAG, UPLO, T, I}
    @inbounds for j in vertices(res)
        #
        # nn is the length of the residual at node j
        #
        #   nn = | res(j) |
        #
        nn = eltypedegree(res, j)
        #
        # na is the length of the separator at node j
        #
        #   na = | sep(j) |
        #
        na = eltypedegree(sep, j)
        #
        # D and L are part of A:
        #
        #        res(j)
        #   A = [ D    ] res(j)
        #       [ L    ] sep(j)
        #
        Dp = Dptr[j]
        Lp = Lptr[j]
        D = tri(uplo, diag, reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn))

        if UPLO === :L
            L = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
        else
            L = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
        end

        f(D, L, neighbors(res, j), neighbors(sep, j))
    end

    return
end

function mapreducefront(f, op, A::ChordalTriangular{DIAG, UPLO}; init) where {DIAG, UPLO}
    return mapreducefront_impl(f, op, A.S.res, A.S.sep, A.S.Dptr, A.S.Lptr, A.Dval, A.Lval, Val(DIAG), Val(UPLO), init)
end

function mapreducefront_impl(
        f,
        op,
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        diag::Val{DIAG},
        uplo::Val{UPLO},
        init,
    ) where {DIAG, UPLO, T, I}
    out = init

    @inbounds for j in vertices(res)
        #
        # nn is the length of the residual at node j
        #
        #   nn = | res(j) |
        #
        nn = eltypedegree(res, j)
        #
        # na is the length of the separator at node j
        #
        #   na = | sep(j) |
        #
        na = eltypedegree(sep, j)
        #
        # D and L are part of A:
        #
        #        res(j)
        #   A = [ D    ] res(j)
        #       [ L    ] sep(j)
        #
        Dp = Dptr[j]
        Lp = Lptr[j]
        D = tri(uplo, diag, reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn))

        if UPLO === :L
            L = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
        else
            L = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
        end

        out = op(out, f(D, L))
    end

    return out
end
