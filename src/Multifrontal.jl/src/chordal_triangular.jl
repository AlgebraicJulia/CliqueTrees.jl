"""
    ChordalTriangular{UPLO, DIAG, T, I, Val} <: AbstractMatrix{T}

A triangular matrix with chordal sparsity pattern. The type parameters
`UPLO` and `DIAG` specify its precise structure:

  - `UPLO`: `:L` or `:U` (lower / upper triangular)
  - `DIAG`: `:N` or `:U` (non-unit / unit triangular)

### Basic Usage

```julia

```

### Fields

   - `A.S`: symbolic factorization

"""
struct ChordalTriangular{UPLO, DIAG, T, I, Val <: AbstractVector{T}} <: AbstractMatrix{T}
    S::ChordalSymbolic{I}
    Dval::Val
    Lval::Val
end

const FChordalTriangular{UPLO, DIAG, T, I} = ChordalTriangular{UPLO, T, I, FVector{T}}

const MaybeAdjOrTransTri{UPLO, DIAG, T, I, Val} = Union{
                 ChordalTriangular{UPLO, DIAG, T, I, Val},
      Adjoint{T, ChordalTriangular{UPLO, DIAG, T, I, Val}},
    Transpose{T, ChordalTriangular{UPLO, DIAG, T, I, Val}},
}

function ChordalTriangular{UPLO, DIAG}(S::ChordalSymbolic{I}, Dval::Val, Lval::Val) where {UPLO, DIAG, I <: Integer, T, Val <: AbstractVector{T}}
    return ChordalTriangular{UPLO, DIAG, T, I, Val}(S, Dval, Lval)
end

function ChordalTriangular{UPLO}(S::ChordalSymbolic{I}) where {UPLO, I <: Integer}
    n = nv(S.res)
    Dval = Ones{Bool}(S.Dptr[n + one(I)] - one(I))
    Lval = Ones{Bool}(S.Lptr[n + one(I)] - one(I))
    return ChordalTriangular{UPLO, :N}(S, Dval, Lval)
end

function ChordalTriangular(F::ChordalCholesky{UPLO}) where {UPLO}
    return ChordalTriangular{UPLO, :N}(F.S, F.Dval, F.Lval)
end

function ChordalTriangular(F::ChordalLDLt{UPLO}) where {UPLO}
    return ChordalTriangular{UPLO, :U}(F.S, F.Dval, F.Lval)
end

function Base.show(io::IO, A::T) where {T <: ChordalTriangular}
    n = size(A, 1)
    print(io, "$n×$n $T with $(nnz(A)) stored entries")
    return
end

function Base.show(io::IO, ::MIME"text/plain", A::T) where {UPLO, T <: ChordalTriangular{UPLO}}
    n = size(A, 1)
    println(io, "$n×$n $T with $(nnz(A)) stored entries:")

    if n < 16
        print_matrix(io, A)
    else
        showsymbolic(io, A.S, Val(UPLO))
    end

    return
end

function Base.replace_in_print_matrix(A::ChordalTriangular{UPLO}, i::Integer, j::Integer, str::AbstractString) where {UPLO}
    if UPLO === :L && i < j || UPLO === :U && i > j
        str = replace_with_centered_mark(str)
    end

    return str
end

function Base.copy(A::ChordalTriangular{UPLO, DIAG, T, I, Val}) where {UPLO, DIAG, T, I, Val}
    return ChordalTriangular{UPLO, DIAG, T, I, Val}(A.S, copy(A.Dval), copy(A.Lval))
end

# ===== Abstract Matrix Interface =====

function SparseArrays.nnz(A::ChordalTriangular)
    return nnz(A.S)
end

function Base.size(L::ChordalTriangular)
    n = convert(Int, nov(L.S.res))
    return (n, n)
end

function LinearAlgebra.istriu(::ChordalTriangular{:U})
    return true
end

function LinearAlgebra.istriu(::ChordalTriangular{:L})
    return false
end

function LinearAlgebra.istril(::ChordalTriangular{:L})
    return true
end

function LinearAlgebra.istril(::ChordalTriangular{:U})
    return false
end

function LinearAlgebra.isposdef(A::ChordalTriangular{UPLO, :N, T, I}) where {UPLO, T, I <: Integer}
    for j in vertices(A.S.res)
        nn = eltypedegree(A.S.res, j)
        Dp = A.S.Dptr[j]
        D = reshape(view(A.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

        for i in oneto(nn)
            ispositive(D[i, i]) || return false
        end
    end

    return true
end

function LinearAlgebra.isposdef(::ChordalTriangular{UPLO, :U}) where {UPLO}
    return true
end

function LinearAlgebra.det(A::ChordalTriangular{UPLO, :N, T, I}) where {UPLO, T, I <: Integer}
    out = one(T)

    for j in vertices(A.S.res)
        nn = eltypedegree(A.S.res, j)
        Dp = A.S.Dptr[j]
        D = reshape(view(A.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

        for i in oneto(nn)
            out *= D[i, i]
        end
    end

    return out
end

function LinearAlgebra.det(A::ChordalTriangular{UPLO, :U, T, I}) where {UPLO, T, I <: Integer}
    return one(T)
end

function LinearAlgebra.logdet(A::ChordalTriangular{UPLO, :N, T, I}) where {UPLO, T, I <: Integer}
    out = zero(T)

    for j in vertices(A.S.res)
        nn = eltypedegree(A.S.res, j)
        Dp = A.S.Dptr[j]
        D = reshape(view(A.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

        for i in oneto(nn)
            out += log(D[i, i])
        end
    end

    return out
end

function LinearAlgebra.logdet(A::ChordalTriangular{UPLO, :U, T, I}) where {UPLO, T, I <: Integer}
    return zero(T)
end

function LinearAlgebra.logabsdet(A::ChordalTriangular{UPLO, :N, T, I}) where {UPLO, T, I <: Integer}
    out = zero(real(T))
    sgn = one(T)

    for j in vertices(A.S.res)
        nn = eltypedegree(A.S.res, j)
        Dp = A.S.Dptr[j]
        D = reshape(view(A.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

        for i in oneto(nn)
            d = D[i, i]
            out += log(abs(d))
            sgn *= sign(d)
        end
    end

    return (out, sgn)
end

function LinearAlgebra.logabsdet(A::ChordalTriangular{UPLO, :U, T, I}) where {UPLO, T, I <: Integer}
    return (zero(real(T)), one(T))
end

function LinearAlgebra.diag(A::ChordalTriangular{UPLO, :N, T, I}) where {UPLO, T, I <: Integer}
    out = Vector{T}(undef, size(A, 1))

    for j in vertices(A.S.res)
        res = neighbors(A.S.res, j)
        nn = eltypedegree(A.S.res, j)
        Dp = A.S.Dptr[j]
        D = reshape(view(A.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

        for i in oneto(nn)
            out[res[i]] = D[i, i]
        end
    end

    return out
end

function LinearAlgebra.diag(A::ChordalTriangular{UPLO, :U, T, I}) where {UPLO, T, I <: Integer}
    return ones(T, size(A, 1))
end

function SparseArrays.sparse(A::ChordalTriangular)
    return csc(A, Val(:N))
end

function SparseArrays.sparse(A::Adjoint{T, <:ChordalTriangular{UPLO, DIAG, T}}) where {UPLO, DIAG, T}
    return csc(parent(A), Val(:C))
end

function SparseArrays.sparse(A::Transpose{T, <:ChordalTriangular{UPLO, DIAG, T}}) where {UPLO, DIAG, T}
    return csc(parent(A), Val(:T))
end

function csc(A::ChordalTriangular{UPLO, DIAG, T, I}, tA::Val{TRANS}) where {UPLO, DIAG, T, I <: Integer, TRANS}
    colptr = Vector{I}(undef, 1 + size(A, 1))
    rowval = Vector{I}(undef, nnz(A))
    nzval = Vector{T}(undef, nnz(A))

    p = zero(I)

    for j in vertices(A.S.res)
        nn = eltypedegree(A.S.res, j)
        na = eltypedegree(A.S.sep, j)

        res = neighbors(A.S.res, j)
        sep = neighbors(A.S.sep, j)

        Dp = A.S.Dptr[j]
        Lp = A.S.Lptr[j]

        D = reshape(view(A.Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

        if UPLO === :L
            L = reshape(view(A.Lval, Lp:Lp + nn * na - one(I)), na, nn)
        else
            L = reshape(view(A.Lval, Lp:Lp + nn * na - one(I)), nn, na)
        end

        for w in oneto(nn)
            colptr[res[w]] = p + one(I)

            for v in w:nn
                if DIAG === :U && v == w
                    x = one(T)
                elseif UPLO === :L
                    x = D[v, w]
                else
                    x = D[w, v]
                end

                if TRANS === :C
                    x = conj(x)
                end

                p += one(I); rowval[p] = res[v]; nzval[p] = x
            end

            for v in oneto(na)
                if UPLO === :L
                    x = L[v, w]
                else
                    x = L[w, v]
                end

                if TRANS === :C
                    x = conj(x)
                end

                p += one(I); rowval[p] = sep[v]; nzval[p] = x
            end
        end
    end

    colptr[end] = p + one(I)

    S = SparseMatrixCSC(size(A)..., colptr, rowval, nzval)

    if (UPLO === :L && TRANS === :N) || (UPLO === :U && TRANS !== :N)
        return S
    else
        return copy(transpose(S))
    end
end

function Base.getindex(A::ChordalTriangular{UPLO, DIAG, T}, v::Integer, w::Integer) where {UPLO, DIAG, T}
    if DIAG !== :U || v != w
        triple = loc(A, v, w)

        if !isnothing(triple)
            flag, i, j = triple

            if flag
                x = A.Dval[i]
            else
                x = A.Lval[i]
            end

            return x
        else
            return zero(T)
        end
    else
        return one(T)
    end
end

function Base.isstored(A::ChordalTriangular{UPLO}, v::Integer, w::Integer) where {UPLO}
    return isstored(A.S, v, w, Val(UPLO))
end

function Base.setindex!(A::ChordalTriangular{UPLO, DIAG}, x, v::Integer, w::Integer) where {UPLO, DIAG}
    if DIAG !== :U || v != w
        triple = loc(A, v, w)

        if !isnothing(triple)
            flag, i, j = triple

            if flag
                A.Dval[i] = x
            else
                A.Lval[i] = x
            end

            return A
        else
            error()
        end
    else
        error()
    end
end

function loc(A::ChordalTriangular{UPLO}, v::Integer, w::Integer) where {UPLO}
    @boundscheck checkbounds(A, v, w)
    return loc(A.S, v, w, Val(UPLO))
end
