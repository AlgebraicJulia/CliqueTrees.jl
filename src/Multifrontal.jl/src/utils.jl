function permuteto(::Type{T}, v::AbstractVector, perm::AbstractVector) where {T}
    out = FVector{T}(undef, length(v))

    @inbounds for i in eachindex(perm)
        out[i] = v[perm[i]]
    end

    return out
end

function permuteto(::Type{T}, v::AbstractFill, ::AbstractVector) where {T}
    return convert(AbstractFill{T}, v)
end

function char(::Val{:N})
    return 'N'
end

function char(::Val{:C})
    return 'C'
end

function char(::Val{:T})
    return 'T'
end

function char(::Val{:L})
    return 'L'
end

function char(::Val{:U})
    return 'U'
end

function char(::Val{:R})
    return 'R'
end

function checksymtri(A::HermOrSymTri{UPLO, T}) where {UPLO, T}
    return (!(T <: Complex) || A isa Hermitian) && A.uplo == char(parent(A).uplo)
end

function checksymbolic(A)
    return true
end

function checksymbolic(A, B, C...)
    return symbolic(A) === symbolic(B) && checksymbolic(B, C...)
end

function symmetric(graph, uplo::Char)
    return symmetric(BipartiteGraph(graph), uplo)
end

function symmetric(graph::AbstractGraph{V}, uplo::Char) where {V}
    E = etype(graph)

    fwd = graph; bwd = reverse(fwd)

    n = nv(fwd)
    m = ne(fwd) + ne(bwd)

    mrk = FVector{V}(undef, n)
    ptr = FVector{E}(undef, n + one(V))
    tgt = FVector{V}(undef, m)

    for v in vertices(fwd)
        mrk[v] = zero(V)
    end

    p = zero(E)

    for v in vertices(fwd)
        mrk[v] = v
        ptr[v] = p + one(E)

        for w in neighbors(fwd, v)
            uplo === 'L' && w < v && continue
            uplo === 'U' && w > v && continue

            if mrk[w] < v
                mrk[w] = v
                p += one(E); tgt[p] = w
            end
        end

        for w in neighbors(bwd, v)
            if mrk[w] < v
                mrk[w] = v
                p += one(E); tgt[p] = w
            end
        end
    end

    ptr[n + one(V)] = p + one(E)
    return BipartiteGraph{V, E}(n, n, m, ptr, tgt)
end

function sympermute(A::AbstractMatrix, invp::AbstractVector, src::Char, tgt::Char)
    return sympermute!(Matrix(A), A, invp, src, tgt)
end

function sympermute(A::ChordalTriangular, invp::AbstractVector, src::Char, tgt::Char)
    return sympermute(sparse(A), invp, src, tgt)
end

function sympermute!(C::AbstractMatrix, A::SparseMatrixCSC, invp::AbstractVector, src::Char, tgt::Char)
    return copyto!(C, sympermute(A, invp, src, tgt))
end

function sympermute!(C::AbstractMatrix, A::ChordalTriangular, invp::AbstractVector, src::Char, tgt::Char)
    return copyto!(C, sympermute(A, invp, src, tgt))
end

function sympermute!(C::AbstractMatrix{T}, A::AbstractMatrix{T}, invp::AbstractVector, src::Char, tgt::Char) where {T}
    n = size(A, 1)

    for j in 1:n
        pj = invp[j]

        if src === 'L'
            rng = j:n
        else
            rng = 1:j
        end

        for i in rng
            pi = invp[i]

            if tgt === 'L'
                lo, hi = minmax(pi, pj)
            else
                hi, lo = minmax(pi, pj)
            end

            if (i > j) == (pi > pj)
                C[hi, lo] = A[i, j]
            else
                C[hi, lo] = conj(A[i, j])
            end
        end
    end

    return C
end

function sympermute(A::SparseMatrixCSC{T, I}, invp::AbstractVector, src::Char, tgt::Char) where {T, I}
    n = size(A, 1)
    m = zero(I)
    colptr = zeros(I, n + 1)

    @inbounds for j in axes(A, 2)
        pj = invp[j]

        for p in nzrange(A, j)
            i = rowvals(A)[p]

            src === 'L' && i < j && continue
            src === 'U' && i > j && continue

            pi = invp[i]

            if tgt === 'L'
                hi = max(pi, pj)
            else
                hi = min(pi, pj)
            end

            if hi < n
                colptr[hi + two(I)] += one(I)
            end

            m += one(I)
        end
    end

    @inbounds colptr[1] = p = one(I)

    @inbounds for i in axes(A, 1)
        colptr[i + 1] = p += colptr[i + 1]
    end

    rowval = Vector{I}(undef, m)
    nzval = Vector{T}(undef, m)

    @inbounds for j in axes(A, 2)
        pj = invp[j]

        for p in nzrange(A, j)
            i = rowvals(A)[p]

            src === 'L' && i < j && continue
            src === 'U' && i > j && continue

            pi = invp[i]

            if tgt === 'L'
                lo, hi = minmax(pi, pj)
            else
                hi, lo = minmax(pi, pj)
            end

            q = colptr[hi + one(I)]
            rowval[q] = lo
            v = nonzeros(A)[p]

            if (i > j) == (pi > pj)
                nzval[q] = conj(v)
            else
                nzval[q] = v
            end

            colptr[hi + one(I)] += one(I)
        end
    end

    B = SparseMatrixCSC{T, I}(n, n, colptr, rowval, nzval)
    return copy(adjoint(B))
end

function colpermute(A::SparseMatrixCSC{T, I}, perm::AbstractVector) where {T, I}
    n = size(A, 2)
    m = nnz(A)
    colptr = Vector{I}(undef, n + 1)
    rowval = Vector{I}(undef, m)
    nzval = Vector{T}(undef, m)

    p = 1

    for j in axes(A, 2)
        colptr[j] = p

        for q in nzrange(A, perm[j])
            rowval[p] = rowvals(A)[q]
            nzval[p] = nonzeros(A)[q]
            p += 1
        end
    end

    colptr[end] = p
    return SparseMatrixCSC{T, I}(size(A)..., colptr, rowval, nzval)
end

function rowpermute(A::SparseMatrixCSC, perm::AbstractVector)
    return permute(A, perm, axes(A, 2))
end

function tri(uplo::Val{:L}, diag::Val{:N}, A::AbstractMatrix)
    return LowerTriangular(A)
end

function tri(uplo::Val{:L}, diag::Val{:U}, A::AbstractMatrix)
    return UnitLowerTriangular(A)
end

function tri(uplo::Val{:U}, diag::Val{:N}, A::AbstractMatrix)
    return UpperTriangular(A)
end

function tri(uplo::Val{:U}, diag::Val{:U}, A::AbstractMatrix)
    return UnitUpperTriangular(A)
end

function sym(::Val{UPLO}, A::AbstractMatrix) where {UPLO}
    return Hermitian(A, UPLO)
end

function adj(tA::Val{:N}, A::AbstractMatrix)
    return A
end

function adj(tA::Val{:T}, A::AbstractMatrix)
    return transpose(A)
end

function adj(tA::Val{:C}, A::AbstractMatrix)
    return adjoint(A)
end

function zerotri!(A::AbstractMatrix{T}, ind::AbstractVector, ::Val{UPLO}) where {UPLO, T}
    @assert size(A, 1) == size(A, 2)
    n = length(ind)

    @inbounds for j in eachindex(ind)
        indj = ind[j]

        if UPLO === :L
            rng = j:n
        else
            rng = 1:j
        end

        for i in rng
            A[ind[i], indj] = zero(T)
        end
    end

    return A
end

function zerotri!(A::AbstractMatrix, uplo::Val)
    return zerotri!(A, axes(A, 1), uplo)
end

function zerorec!(A::AbstractVector{T}, ind::AbstractVector, ::Val{:L}) where {T}
    @inbounds for i in eachindex(ind)
        A[ind[i]] = zero(T)
    end

    return A
end

function zerorec!(A::AbstractVector)
    return zerorec!(A, axes(A, 1), Val(:L))
end

function zerorec!(A::AbstractMatrix{T}, ind::AbstractVector, ::Val{:L}) where {T}
    @inbounds for j in axes(A, 2)
        for i in eachindex(ind)
            A[ind[i], j] = zero(T)
        end
    end

    return A
end

function zerorec!(A::AbstractMatrix{T}, ind::AbstractVector, ::Val{:R}) where {T}
    @inbounds for j in eachindex(ind)
        indj = ind[j]

        for i in axes(A, 1)
            A[i, indj] = zero(T)
        end
    end

    return A
end

function zerorec!(A::AbstractMatrix)
    return zerorec!(A, axes(A, 1), Val(:L))
end

function copyrec!(A::AbstractVecOrMat, B::AbstractVecOrMat)
    return copyscatterrec!(A, B, axes(B, 1), Val(:L))
end

function copygatherrec!(A::AbstractVector, B::AbstractVector, ind::AbstractVector)
    return copygatherrec!(A, B, ind, Val(:L))
end

function copyscatterrec!(A::AbstractVector, B::AbstractVector, ind::AbstractVector)
    return copyscatterrec!(A, B, ind, Val(:L))
end

function addgatherrec!(A::AbstractVector, B::AbstractVector, ind::AbstractVector)
    return addgatherrec!(A, B, ind, Val(:L))
end

function addscatterrec!(A::AbstractVector, B::AbstractVector, ind::AbstractVector)
    return addscatterrec!(A, B, ind, Val(:L))
end

function addrec!(A::AbstractVecOrMat, B::AbstractVecOrMat)
    return addscatterrec!(A, B, axes(B, 1), Val(:L))
end

function copytri!(A::AbstractMatrix, B::AbstractMatrix, uplo::Val)
    return copyscattertri!(A, B, axes(B, 1), uplo)
end

function addtri!(A::AbstractMatrix, B::AbstractMatrix, uplo::Val)
    return addscattertri!(A, B, axes(B, 1), uplo)
end

function symmtri!(A::AbstractMatrix{T}, ::Val{:L}) where {T}
    n = size(A, 1)

    @inbounds for j in axes(A, 2)
        for i in 1:j-1
            A[i, j] = conj(A[j, i])
        end
    end

    return A
end

function symmtri!(A::AbstractMatrix{T}, ::Val{:U}) where {T}
    n = size(A, 1)

    @inbounds for j in axes(A, 2)
        for i in j+1:n
            A[i, j] = conj(A[j, i])
        end
    end

    return A
end

function rdivtri!(A::AbstractMatrix{T}, α, ::Val{:L}) where {T}
    n = size(A, 1)

    @inbounds for j in axes(A, 2)
        for i in j:n
            A[i, j] /= α
        end
    end

    return A
end

function rdivtri!(A::AbstractMatrix{T}, α, ::Val{:U}) where {T}
    @inbounds for j in axes(A, 2)
        for i in 1:j
            A[i, j] /= α
        end
    end

    return A
end

for (f, op) in [(:copy, :(=)), (:add, :(+=))]
    @eval function $(Symbol(f, :gatherrec!))(
            A::AbstractVecOrMat,
            B::AbstractVecOrMat,
            ind::AbstractVector,
            ::Val{:L},
        )
        @assert size(A, 1) <= length(ind)
        @assert size(A, 2) == size(B, 2)

        @inbounds for j in axes(A, 2)
            for i in axes(A, 1)
                $(Expr(op, :(A[i, j]), :(B[ind[i], j])))
            end
        end

        return A
    end

    @eval function $(Symbol(f, :gatherrec!))(
            A::AbstractVecOrMat,
            B::AbstractVecOrMat,
            ind::AbstractVector,
            ::Val{:R},
        )
        @assert size(A, 1) == size(B, 1)
        @assert size(A, 2) <= length(ind)

        @inbounds for j in axes(A, 2)
            indj = ind[j]

            for i in axes(A, 1)
                $(Expr(op, :(A[i, j]), :(B[i, indj])))
            end
        end

        return A
    end

    @eval function $(Symbol(f, :scatterrec!))(
            A::AbstractVecOrMat,
            B::AbstractVecOrMat,
            ind::AbstractVector,
            ::Val{:L},
        )
        @assert size(B, 1) <= length(ind)
        @assert size(A, 2) == size(B, 2)

        @inbounds for j in axes(B, 2)
            for i in axes(B, 1)
                $(Expr(op, :(A[ind[i], j]), :(B[i, j])))
            end
        end

        return A
    end

    @eval function $(Symbol(f, :scatterrec!))(
            A::AbstractVecOrMat,
            B::AbstractVecOrMat,
            ind::AbstractVector,
            ::Val{:R},
        )
        @assert size(A, 1) == size(B, 1)
        @assert size(B, 2) <= length(ind)

        @inbounds for j in axes(B, 2)
            indj = ind[j]

            for i in axes(B, 1)
                $(Expr(op, :(A[i, indj]), :(B[i, j])))
            end
        end

        return A
    end

    @eval function $(Symbol(f, :gathertri!))(
            A::AbstractMatrix,
            B::AbstractMatrix,
            ind::AbstractVector,
            ::Val{UPLO},
        ) where {UPLO}
        @assert size(A, 1) == size(A, 2) <= length(ind)
        n = size(A, 1)

        @inbounds for j in axes(A, 2)
            indj = ind[j]

            if UPLO === :L
                rng = j:n
            else
                rng = 1:j
            end

            for i in rng
                $(Expr(op, :(A[i, j]), :(B[ind[i], indj])))
            end
        end

        return A
    end

    @eval function $(Symbol(f, :scattertri!))(
            A::AbstractMatrix,
            B::AbstractMatrix,
            ind::AbstractVector,
            ::Val{UPLO},
        ) where {UPLO}
        @assert size(B, 1) == size(B, 2) <= length(ind)
        n = size(B, 1)

        @inbounds for j in axes(B, 2)
            indj = ind[j]

            if UPLO === :L
                rng = j:n
            else
                rng = 1:j
            end

            for i in rng
                $(Expr(op, :(A[ind[i], indj]), :(B[i, j])))
            end
        end

        return A
    end
end

function unwrap(A::AbstractVecOrMat)
    return (A, Val(:N))
end

function unwrap(A::Adjoint)
    return (parent(A), Val(:C))
end

function unwrap(A::Transpose)
    return (parent(A), Val(:T))
end

function isforward(UPLO, TRANS, SIDE)
    return UPLO === :L && (TRANS === :N && SIDE === :L || TRANS !== :N && SIDE === :R) ||
           UPLO === :U && (TRANS !== :N && SIDE === :L || TRANS === :N && SIDE === :R)
end

function swaprec!(v::AbstractVector, j::Integer, k::Integer)
    @inbounds v[j], v[k] = v[k], v[j]

    return
end

function swaprec!(::AbstractFill, ::Integer, ::Integer)
    return
end


# Hermitian swap: exchange rows/cols j and k for Hermitian matrix stored in lower triangle
function swaptri!(A::AbstractMatrix, j::Integer, k::Integer, ::Val{:L})
    n = size(A, 1)

    @inbounds begin
        A[j, j], A[k, k] = A[k, k], A[j, j]

        for i in k+1:n
            A[i, j], A[i, k] = A[i, k], A[i, j]
        end

        for i in j+1:k-1
            A[i, j], A[k, i] = conj(A[k, i]), conj(A[i, j])
        end

        for i in 1:j-1
            A[j, i], A[k, i] = A[k, i], A[j, i]
        end

        A[k, j] = conj(A[k, j])
    end

    return
end

# Hermitian swap: exchange rows/cols j and k for Hermitian matrix stored in upper triangle
function swaptri!(A::AbstractMatrix, j::Integer, k::Integer, ::Val{:U})
    n = size(A, 1)

    @inbounds begin
        A[j, j], A[k, k] = A[k, k], A[j, j]

        for i in 1:j-1
            A[i, j], A[i, k] = A[i, k], A[i, j]
        end

        for i in j+1:k-1
            A[j, i], A[i, k] = conj(A[i, k]), conj(A[j, i])
        end

        for i in k+1:n
            A[j, i], A[k, i] = A[k, i], A[j, i]
        end

        A[j, k] = conj(A[j, k])
    end

    return
end

function swapcol!(A::AbstractMatrix, j::Integer, k::Integer)
    @inbounds for i in axes(A, 1)
        A[i, j], A[i, k] = A[i, k], A[i, j]
    end

    return
end

function swaprow!(A::AbstractMatrix, j::Integer, k::Integer)
    @inbounds for i in axes(A, 2)
        A[j, i], A[k, i] = A[k, i], A[j, i]
    end

    return
end


function cdiv!(::Val, ::Val{:N}, ::AbstractMatrix, ::AbstractVector)
    return
end

function cdiv!(::Val{:R}, ::Val{:U}, A::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    @inbounds for j in axes(A, 2)
        idj = inv(d[j])

        for i in axes(A, 1)
            A[i, j] *= idj
        end
    end

    return
end

function cdiv!(::Val{:L}, ::Val{:U}, A::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    @inbounds for j in axes(A, 2)
        for i in axes(A, 1)
            A[i, j] *= inv(d[i])
        end
    end

    return
end

function cmul!(::Val, ::Val{:N}, ::AbstractMatrix, ::AbstractVector)
    return
end

function cmul!(::Val{:R}, ::Val{:U}, A::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    @inbounds for j in axes(A, 2)
        dj = d[j]

        for i in axes(A, 1)
            A[i, j] *= dj
        end
    end

    return
end

function cmul!(::Val{:L}, ::Val{:U}, A::AbstractMatrix{T}, d::AbstractVector{T}) where {T}
    @inbounds for j in axes(A, 2)
        for i in axes(A, 1)
            A[i, j] *= d[i]
        end
    end

    return
end

function revtri!(A::AbstractMatrix, ::Val{UPLO}) where UPLO
    n = size(A, 1)

    @inbounds for i in 1:n ÷ 2
        ri = n - i + 1
        A[i, i], A[ri, ri] = A[ri, ri], A[i, i]
    end

    @inbounds for j in 1:n
        rj = n - j + 1

        if UPLO === :L
            rng = j + 1:n
        else
            rng = 1:j - 1
        end

        for i in rng
            ri = n - i + 1
            A[ri, rj] = A[i, j]
        end
    end
end

# ===== allocate =====

function allocate(::Type{Arr}, dims::Integer...) where {Arr}
    return allocate(Arr, dims)
end

function allocate(::Type{Arr}, dims::Tuple) where {Arr}
    return Arr(undef, dims)
end

function allocate(::Type{Zeros{T, N, NTuple{N, OneTo{Int}}}}, dims::NTuple{N}) where {T, N}
    return Zeros{T, N}(map(OneTo{Int}, dims))
end

function allocate(::Type{Ones{T, N, NTuple{N, OneTo{Int}}}}, dims::NTuple{N}) where {T, N}
    return Ones{T, N}(map(OneTo{Int}, dims))
end

function allocate(::Type{Arr}, (n,)::Tuple) where {Arr <: OneTo}
    return Arr(n)
end

# Unwrap adjoint/transpose for selupd, returning (parent, transpose_flag, conjugate_flag)
function unwrap2(A)
    if A isa Adjoint
        B = parent(A)

        if B isa Transpose
            return (parent(B), Val(:N), Val(:C))
        else
            return (B,         Val(:T), Val(:C))
        end
    elseif A isa Transpose
        B = parent(A)

        if B isa Adjoint
            return (parent(B), Val(:N), Val(:C))
        else
            return (B,         Val(:T), Val(:N))
        end
    else
        return (A, Val(:N), Val(:N))
    end
end

# SELected UPDate: C ← α A Bᴴ + conj(α) B Aᴴ + β C for sparse Hermitian
function selupd!(C::Hermitian{T, SparseMatrixCSC{T, I}}, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β) where {T, I}
    selupd!(parent(C), C.uplo, A, adjoint(B),      α,  β)
    selupd!(parent(C), C.uplo, B, adjoint(A), conj(α), 1)
    return C
end

# SELected UPDate: C ← α A Bᴴ + α conj(B) Aᵀ + β C for sparse Symmetric
function selupd!(C::Symmetric{T, SparseMatrixCSC{T, I}}, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β) where {T, I}
    selupd!(parent(C), C.uplo, A,                     adjoint(B),   α, β)
    selupd!(parent(C), C.uplo, adjoint(transpose(B)), transpose(A), α, 1)
    return C
end

# SELected UPDate: C ← α A B + β C for SparseMatrixCSC
function selupd!(C::SparseMatrixCSC, uplo::Char, A::AbstractVecOrMat, B::AbstractVecOrMat, α, β)
    AP, tA, cA = unwrap2(A)
    BP, tB, cB = unwrap2(B)
    return selupd_impl!(C, uplo, AP, BP, α, β, tA, cA, tB, cB)
end

function selupd_impl!(C::SparseMatrixCSC, uplo::Char, A::AbstractVector, B::AbstractVector, α, β, ::Val{tA}, ::Val{cA}, ::Val{tB}, ::Val{cB}) where {tA, cA, tB, cB}
    @assert size(C, 1) == size(C, 2) == length(A) == length(B)

    @inbounds for j in axes(C, 2)
        if cB === :C
            Bj = conj(B[j])
        else
            Bj = B[j]
        end

        for p in nzrange(C, j)
            i = rowvals(C)[p]

            if (uplo == 'L' && i >= j) || (uplo == 'U' && i <= j)
                if cA === :C
                    Ai = conj(A[i])
                else
                    Ai = A[i]
                end

                if iszero(β)
                    nonzeros(C)[p] = α * Ai * Bj
                else
                    nonzeros(C)[p] = β * nonzeros(C)[p] + α * Ai * Bj
                end
            end
        end
    end

    return C
end

function selupd_impl!(C::SparseMatrixCSC, uplo::Char, A::AbstractMatrix, B::AbstractMatrix, α, β, tA::Val{TA}, cA::Val{CA}, tB::Val{TB}, cB::Val{CB}) where {TA, CA, TB, CB}
    @assert size(C, 1) == size(C, 2)

    if TA === :N && TB === :N
        @assert size(A, 1) == size(C, 1)
        @assert size(B, 2) == size(C, 1)
        @assert size(A, 2) == size(B, 1)
    elseif TA === :N && TB !== :N
        @assert size(A, 1) == size(C, 1)
        @assert size(B, 1) == size(C, 1)
        @assert size(A, 2) == size(B, 2)
    elseif TA !== :N && TB === :N
        @assert size(A, 2) == size(C, 1)
        @assert size(B, 2) == size(C, 1)
        @assert size(A, 1) == size(B, 1)
    else
        @assert size(A, 2) == size(C, 1)
        @assert size(B, 1) == size(C, 1)
        @assert size(A, 1) == size(B, 2)
    end

    if TA === :N
        rng = axes(A, 2)
    else
        rng = axes(A, 1)
    end

    if iszero(β)
        fill!(nonzeros(C), β)
    else
        rmul!(nonzeros(C), β)
    end

    for k in rng
        if TA === :N
            Ak = view(A, :, k)
        else
            Ak = view(A, k, :)
        end

        if TB === :N
            Bk = view(B, k, :)
        else
            Bk = view(B, :, k)
        end

        selupd_impl!(C, uplo, Ak, Bk, α, 1, tA, cA, tB, cB)
    end

    return C
end
