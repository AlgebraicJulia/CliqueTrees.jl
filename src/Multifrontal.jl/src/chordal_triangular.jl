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
struct ChordalTriangular{DIAG, UPLO, T, I, Dvl <: AbstractVector{T}, Lvl <: AbstractVector{T}} <: AbstractMatrix{T}
    S::ChordalSymbolic{I}
    Dval::Dvl
    Lval::Lvl
end

const FChordalTriangular{DIAG, UPLO, T, I} = ChordalTriangular{
    DIAG,
    UPLO,
    T,
    I,
    FVector{T},
    FVector{T},
}

const DChordalTriangular{DIAG, UPLO, T, I} = ChordalTriangular{
    DIAG,
    UPLO,
    T,
    I,
    Vector{T},
    Vector{T},
}

const SymbolicChordalTriangular{UPLO, I} = ChordalTriangular{
    :N,
    UPLO,
    Bool,
    I,
    IOnes{Bool},
    IOnes{Bool},
}

const AdjOrTransTri{DIAG, UPLO, T, I, Dvl, Lvl} = Union{
      Adjoint{T, ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl}},
    Transpose{T, ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl}},
}

const MaybeAdjOrTransTri{DIAG, UPLO, T, I, Dvl, Lvl} = Union{
        AdjOrTransTri{DIAG, UPLO, T, I, Dvl, Lvl},
    ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl},
}

function ChordalTriangular{DIAG, UPLO}(S::ChordalSymbolic{I}, Dval::Dvl, Lval::Lvl) where {DIAG, UPLO, I <: Integer, T, Dvl <: AbstractVector{T}, Lvl <: AbstractVector{T}}
    return ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl}(S, Dval, Lval)
end

function ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl}(S::ChordalSymbolic{I}) where {DIAG, UPLO, T, I, Dvl, Lvl}
    Dval = allocate(Dvl, ndz(S))
    Lval = allocate(Lvl, nlz(S))
    return ChordalTriangular{DIAG, UPLO, T, I, Dvl, Lvl}(S, Dval, Lval)
end

function SymbolicChordalTriangular{UPLO}(S::ChordalSymbolic{I}) where {UPLO, I}
    return SymbolicChordalTriangular{UPLO, I}(S)
end

function ChordalTriangular{DIAG, UPLO}(F::ChordalFactorization) where {DIAG, UPLO}
    return ChordalTriangular{DIAG, UPLO}(getfield(F, :S), getfield(F, :Dval), getfield(F, :Lval))
end

function ChordalTriangular{DIAG}(F::ChordalFactorization{<:Any, UPLO}) where {DIAG, UPLO}
    return ChordalTriangular{DIAG, UPLO}(F)
end

function ChordalTriangular(F::ChordalFactorization{DIAG, UPLO}) where {DIAG, UPLO}
    return ChordalTriangular{DIAG, UPLO}(F)
end

function Base.getproperty(A::ChordalTriangular{DIAG, UPLO}, s::Symbol) where {DIAG, UPLO}
    if s === :uplo
        return Val(UPLO)
    elseif s === :diag
        return Val(DIAG)
    else
        return getfield(A, s)
    end
end

function Base.show(io::IO, A::T) where {T <: ChordalTriangular}
    n = ncl(A)
    print(io, "$n×$n $T with $(nnz(A)) stored entries")
    return
end

for Tri in (:FChordalTriangular, :DChordalTriangular)
    @eval function Base.show(io::IO, ::Type{$Tri{DIAG, UPLO, T, I}}) where {DIAG, UPLO, T, I}
        print(io, $("$Tri{"), repr(DIAG), ", ", repr(UPLO), ", ", T, ", ", I, "}")
    end
end

function Base.show(io::IO, ::MIME"text/plain", A::T) where {DIAG, UPLO, T <: ChordalTriangular{DIAG, UPLO}}
    n = ncl(A)
    println(io, "$n×$n $T with $(nnz(A)) stored entries:")

    if n < 16
        print_matrix(io, A)
    else
        showsymbolic(io, A.S, A.uplo)
    end

    return
end

function Base.replace_in_print_matrix(A::ChordalTriangular{DIAG, UPLO}, i::Integer, j::Integer, str::AbstractString) where {DIAG, UPLO}
    if UPLO === :L && i < j || UPLO === :U && i > j
        str = replace_with_centered_mark(str)
    end

    return str
end

function Base.similar(A::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    return ChordalTriangular{DIAG, UPLO}(A.S, similar(A.Dval), similar(A.Lval))
end

function Base.similar(A::AdjOrTransTri{DIAG, UPLO}) where {DIAG, UPLO}
    P = parent(A)

    if UPLO === :L
        B = ChordalTriangular{DIAG, :U}(P.S, similar(P.Dval), similar(P.Lval))
    else
        B = ChordalTriangular{DIAG, :L}(P.S, similar(P.Dval), similar(P.Lval))
    end

    return B
end

function Base.copy(A::MaybeAdjOrTransTri)
    return copyto!(similar(A), A)
end

function Base.copyto!(A::MaybeAdjOrTransTri, B::MaybeAdjOrTransTri)
    AP, TA = unwrap(A)
    BP, TB = unwrap(B)
    copy_impl!(AP, BP, TA, TB)
    return A
end

function copy_impl!(A::ChordalTriangular, B::ChordalTriangular, ::Val{TA}, ::Val{TB}) where {TA, TB}
    if TA === TB
        copyto!(A.Dval, B.Dval)
        copyto!(A.Lval, B.Lval)
    elseif TA === :N
        copy_impl!(A, B, TB)
    elseif TB === :N
        copy_impl!(A, B, TA)
    else
        conj!(copyto!(A.Dval, B.Dval))
        conj!(copyto!(A.Lval, B.Lval))
    end

    return A
end

function copy_impl!(A::ChordalTriangular, B::ChordalTriangular, ::Val{TRANS}) where {TRANS}
    @inbounds for j in fronts(B)
        DB, _ = diagblock(B, j)
        LB, _ = offdblock(B, j)
        DA, _ = diagblock(A, j)
        LA, _ = offdblock(A, j)

        if TRANS === :C
            adjoint!(parent(DA), parent(DB))
            adjoint!(LA, LB)
        else
            transpose!(parent(DA), parent(DB))
            transpose!(LA, LB)
        end
    end

    return A
end

# ===== Abstract Matrix Interface =====

function SparseArrays.nnz(A::ChordalTriangular)
    return nnz(A.S)
end

function SparseArrays.findnz(A::ChordalTriangular{DIAG, UPLO, T, I}) where {DIAG, UPLO, T, I}
    m = half(ndz(A) + ncl(A)) + nlz(A)
    rows = Vector{I}(undef, m)
    cols = Vector{I}(undef, m)
    vals = Vector{T}(undef, m)

    p = zero(I)

    @inbounds for j in fronts(A)
        D, res = diagblock(A, j)
        L, sep = offdblock(A, j)

        if UPLO === :L
            for w in eachindex(res)
                for v in w:length(res)
                    p += one(I)
                    rows[p] = res[v]
                    cols[p] = res[w]
                    vals[p] = D[v, w]
                end

                for v in eachindex(sep)
                    p += one(I)
                    rows[p] = sep[v]
                    cols[p] = res[w]
                    vals[p] = L[v, w]
                end
            end
        else
            for v in eachindex(res)
                for w in 1:v
                    p += one(I)
                    rows[p] = res[w]
                    cols[p] = res[v]
                    vals[p] = D[w, v]
                end
            end

            for v in eachindex(sep)
                for w in eachindex(res)
                    p += one(I)
                    rows[p] = res[w]
                    cols[p] = sep[v]
                    vals[p] = L[w, v]
                end
            end
        end
    end

    return (rows, cols, vals)
end

function ndz(A::ChordalTriangular)
    return ndz(A.S)
end

function nlz(A::ChordalTriangular)
    return nlz(A.S)
end

function ncl(A::MaybeAdjOrTransTri)
    return ncl(parent(A).S)
end

function nfr(A::MaybeAdjOrTransTri)
    return nfr(parent(A).S)
end

function Base.size(L::ChordalTriangular)
    return size(L.S)
end

function LinearAlgebra.istriu(::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    return UPLO === :U
end

function LinearAlgebra.istril(::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    return UPLO === :L
end

function LinearAlgebra.isposdef(A::ChordalTriangular{DIAG, UPLO}) where {DIAG, UPLO}
    if DIAG === :N
        @inbounds for j in fronts(A)
            D, res = diagblock(A, j)
            all(ispositive, view(D, diagind(D))) || return false
        end
    end

    return true
end

function LinearAlgebra.det(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    out = one(T)

    if DIAG === :N
        @inbounds for j in fronts(A)
            D, res = diagblock(A, j)
            out *= det(D)
        end
    end

    return out
end

function LinearAlgebra.logdet(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    out = zero(T)

    if DIAG === :N
        @inbounds for j in fronts(A)
            D, res = diagblock(A, j)
            out += logdet(D)
        end
    end

    return out
end

function LinearAlgebra.logabsdet(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    out = (zero(real(T)), one(T))

    if DIAG === :N
        @inbounds for j in fronts(A)
            D, res = diagblock(A, j)
            a, b = out
            c, d = logabsdet(D)
            out = (a + c, b * d)
        end
    end

    return out
end

function LinearAlgebra.tr(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    if DIAG === :N
        out = zero(T)

        @inbounds for j in fronts(A)
            D, res = diagblock(A, j)
            out += tr(D)
        end

        return out
    else
        return convert(T, ncl(A))
    end
end

function LinearAlgebra.rank(A::ChordalTriangular{DIAG, UPLO, T, I}; kw...) where {DIAG, UPLO, T, I <: Integer}
    if DIAG === :N
        out = 0

        @inbounds for j in fronts(A)
            D, res = diagblock(A, j)
            out += rank(D; kw...)
        end

        return out
    else
        return ncl(A)
    end
end

function LinearAlgebra.diag(A::ChordalTriangular{DIAG, UPLO, T}) where {DIAG, UPLO, T}
    out = Vector{T}(undef, ncl(A))

    if DIAG === :N
        @inbounds for j in fronts(A)
            D, res = diagblock(A, j)
            out[res] .= view(D, diagind(D))
        end
    else
        out .= one(T)
    end

    return out
end

function SparseArrays.sparse(A::ChordalTriangular{DIAG, UPLO, T, I}) where {DIAG, UPLO, T, I <: Integer}
    return sparse_impl(A, Val(:N))
end

function SparseArrays.sparse(A::Adjoint{T, ChordalTriangular{DIAG, UPLO, T, I}}) where {DIAG, UPLO, T, I <: Integer}
    return sparse_impl(parent(A), Val(:C))
end

function SparseArrays.sparse(A::Transpose{T, ChordalTriangular{DIAG, UPLO, T, I}}) where {DIAG, UPLO, T, I <: Integer}
    return sparse_impl(parent(A), Val(:T))
end

function sparse_impl(A::ChordalTriangular{DIAG, UPLO, T, I}, ::Val{TRANS}) where {DIAG, UPLO, T, I <: Integer, TRANS}
    m = half(ndz(A) + ncl(A)) + nlz(A)

    colptr = Vector{I}(undef, 1 + ncl(A))
    rowval = Vector{I}(undef, m)
    nzval = Vector{T}(undef, m)

    p = zero(I)

    @inbounds for j in fronts(A)
        D, res = diagblock(A, j)
        L, sep = offdblock(A, j)

        for w in eachindex(res)
            colptr[res[w]] = p + one(I)

            for v in w:length(res)
                if UPLO === :L
                    x = D[v, w]
                else
                    x = D[w, v]
                end

                if TRANS === :C
                    x = conj(x)
                end

                p += one(I); rowval[p] = res[v]; nzval[p] = x
            end

            for v in eachindex(sep)
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

    S = SparseMatrixCSC{T, I}(size(A)..., colptr, rowval, nzval)

    if isforward(UPLO, TRANS, :R)
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

function Base.isstored(A::ChordalTriangular, v::Integer, w::Integer)
    return isstored(A.S, v, w, A.uplo)
end

function Base.setindex!(A::ChordalTriangular{DIAG, UPLO}, x, v::Integer, w::Integer) where {DIAG, UPLO}
    if DIAG !== :U || v != w
        return setflatindex!(A, x, flatindex(A, v, w))
    else
        error()
    end
end

function flatindex(A::ChordalTriangular, v::Integer, w::Integer)
    @boundscheck checkbounds(A, v, w)
    return flatindex(A.S, v, w, A.uplo)
end

function flatindices(A::ChordalTriangular, B::SparseMatrixCSC)
    return flatindices(A.S, B, A.uplo)
end

function getflatindex(A::ChordalTriangular{DIAG, UPLO, T}, p::Integer) where {DIAG, UPLO, T}
    poff = ndz(A)

    if iszero(p)
        return zero(T)
    elseif p <= poff
        return A.Dval[p]
    else
        return A.Lval[p - poff]
    end
end

function setflatindex!(A::ChordalTriangular, x, p::Integer)
    poff = ndz(A)

    if iszero(p)
        error()
    elseif p <= poff
        A.Dval[p] = x
    else
        A.Lval[p - poff] = x
    end

    return A
end


function Base.copyto!(A::ChordalTriangular{DIAG, :L, T, I}, B::SparseMatrixCSC) where {DIAG, T, I}
    zerorec!(A.Dval)
    zerorec!(A.Lval)

    @inbounds for i in fronts(A)
        D, res = diagblock(A, i)
        L, sep = offdblock(A, i)

        rlo = first(res)
        rhi = last(res) + one(I)

        if !isempty(sep)
            slo = first(sep)
            shi = last(sep) + one(I)
        end

        for j in eachindex(res)
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

function Base.copyto!(A::ChordalTriangular{DIAG, :U, T, I}, B::SparseMatrixCSC) where {DIAG, T, I}
    zerorec!(A.Dval)
    zerorec!(A.Lval)

    @inbounds for i in fronts(A)
        D, res = diagblock(A, i)
        L, sep = offdblock(A, i)

        rlo = first(res)
        rhi = last(res) + one(I)

        for j in eachindex(res)
            for p in nzrange(B, res[j])
                row = rowvals(B)[p]
                row < rlo && continue
                row >= rhi && break
                parent(D)[row - rlo + one(I), j] = nonzeros(B)[p]
            end
        end

        for j in eachindex(sep)
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

function Base.copy!(A::ChordalTriangular, B)
    return copyto!(A, B)
end

function Base.fill!(A::ChordalTriangular, x)
    fill!(A.Dval, x)
    fill!(A.Lval, x)
    return A
end

function LinearAlgebra.cond(F::MaybeAdjOrTransTri, p::Real=2)
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

function LinearAlgebra.opnorm(A::MaybeAdjOrTransTri, p::Real=2)
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

function fronts(A::MaybeAdjOrTransTri)
    return fronts(parent(A).S)
end

function diagblock(A::ChordalTriangular, j::Integer)
    return diagblock_impl(A.S.res, A.S.Dptr, A.Dval, A.diag, A.uplo, j)
end

function diagblock_impl(
        res::AbstractGraph{I},
        Dptr::AbstractVector{I},
        Dval::AbstractVector{T},
        diag::Val{DIAG},
        uplo::Val{UPLO},
        j::Integer,
    ) where {DIAG, UPLO, T, I}
    @boundscheck checkbounds(vertices(res), j)
    #
    # nn is the length of the residual at node j
    #
    #   nn = | res(j) |
    #
    nn = eltypedegree(res, j)
    #
    # D is the diagonal block of A at front j:
    #
    #        res(j)
    #   A = [ D    ] res(j)
    #
    Dp = Dptr[j]
    D = tri(uplo, diag, reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn))

    return D, neighbors(res, j)
end

function offdblock(A::ChordalTriangular, j::Integer)
    return offdblock_impl(A.S.res, A.S.sep, A.S.Lptr, A.Lval, A.uplo, j)
end

function offdblock_impl(
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        Lptr::AbstractVector{I},
        Lval::AbstractVector{T},
        uplo::Val{UPLO},
        j::Integer,
    ) where {UPLO, T, I}
    @boundscheck checkbounds(vertices(res), j)
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
    # L is the off-diagonal block of A at front j:
    #
    #        res(j)
    #   A = [ L    ] sep(j)
    #
    Lp = Lptr[j]

    if UPLO === :L
        L = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end

    return L, neighbors(sep, j)
end
