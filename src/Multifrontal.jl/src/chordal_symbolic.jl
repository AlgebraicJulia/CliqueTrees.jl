"""
    ChordalSymbolic{I}

A symbolic factorization of a sparse symmetric matrix. It can
be used to construct a [`ChordalCholesky`](@ref) or [`ChordalLDLt`](@ref) matrix
factorization, determining the sparsity pattern of its lower
and upper triangular factors.

### Fields

  - `S.L`: lower triangular factor (Boolean)
  - `S.U`: upper triangular factor (Boolean)

"""
struct ChordalSymbolic{I}
    res::BipartiteGraph{I, I, FVector{I}, OneTo{I}}
    sep::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    rel::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    chd::BipartiteGraph{I, I, FVector{I}, FVector{I}}
    pnt::FVector{I}
    idx::FVector{I}
    Dptr::FVector{I}
    Lptr::FVector{I}
    nMptr::I
    nMval::I
    nNval::I
    nFval::I
end

"""
    symbolic(A::AbstractMatrix; kw...)

Construct a symbolic factorization. The keyword arguments
`kw` are passed to the function [`cliquetree`](@ref).

### Example

```julia-repl
julia> using CliqueTrees.Multifrontal, LinearAlgebra

julia> A = [
           4  2  0  0  2
           2  5  0  0  3
           0  0  4  2  0
           0  0  2  5  2
           2  3  0  2  7
       ]

julia> perm, S = symbolic(A);

julia> S
5×5 ChordalSymbolic{Int64} with 10 stored entries:
  true    ⋅      ⋅      ⋅      ⋅  
 false   true    ⋅      ⋅      ⋅  
 false   true   true    ⋅      ⋅  
  true  false  false   true    ⋅  
 false   true   true   true   true

julia> F = cholesky!(ChordalCholesky(A, perm, S))
5×5 FChordalCholesky{:L, Float64, Int64} with 10 stored entries:
 2.0   ⋅    ⋅    ⋅    ⋅ 
 0.0  2.0   ⋅    ⋅    ⋅ 
 0.0  1.0  2.0   ⋅    ⋅ 
 1.0  0.0  0.0  2.0   ⋅ 
 0.0  1.0  1.0  1.0  2.0
```
"""
function symbolic(A::AbstractMatrix; kw...)
    return symbolic(sparse(A); kw...)
end

function symbolic(A::SparseMatrixCSC; check::Bool=true, kw...)
    if !check || ishermitian(A)
        return symbolic(BipartiteGraph(A); kw...)
    elseif istril(A)
        return symbolic(Hermitian(A, :L); kw...)
    elseif istriu(A)
        return symbolic(Hermitian(A, :U); kw...)
    end

    error()
end

function symbolic(A::HermOrSym; check::Bool=true, kw...)
    return symbolic(symmetric(BipartiteGraph(parent(A)), A.uplo); kw...)
end

function symbolic(graph::AbstractGraph{I}; kw...) where {I <: Integer}
    perm, tree = cliquetree(graph; kw...)
    return FVector{I}(perm), ChordalSymbolic(tree)
end

function ChordalSymbolic(tree::CliqueTree{I, I}) where {I <: Integer}
    res = residuals(tree)
    sep = separators(tree)
    return ChordalSymbolic(res, sep, Tree(tree))
end

function ChordalSymbolic(res::BipartiteGraph{I, I}, sep::BipartiteGraph{I, I}) where {I}
    pnt = FVector{I}(undef, nv(res))
    idx = FVector{I}(undef, nov(res))
    pstop = ne(sep) + one(I)

    for i in reverse(vertices(res))
        pstrt = pointers(sep)[i]

        if pstrt < pstop
            w = targets(sep)[pstrt]
            j = idx[w]
        else
            j = zero(I)
        end

        pnt[i] = j

        for v in neighbors(res, i)
            idx[v] = i
        end

        pstop = pstrt
    end

    return ChordalSymbolic(res, sep, Tree(nv(res), pnt))
end

function ChordalSymbolic(res::BipartiteGraph{I, I}, sep::BipartiteGraph{I, I}, tree::Tree{I}) where {I}
    chd = tree.graph
    pnt = tree.tree.prnt

    reltgt = FVector{I}(undef, ne(sep))
    rel = BipartiteGraph(nov(sep), nv(sep), ne(sep), pointers(sep), reltgt)

    nMptr = jMptr = one(I)
    nMval = jMval = zero(I)
    nNval = jNval = zero(I)
    nFval = zero(I)
    Dp = zero(I)
    Lp = zero(I)

    idx = FVector{I}(undef, nov(res))
    Dptr = FVector{I}(undef, nv(res) + one(I))
    Lptr = FVector{I}(undef, nv(res) + one(I))

    for j in vertices(res)
        Dptr[j] = Dp + one(I)
        Lptr[j] = Lp + one(I)

        nn = eltypedegree(res, j)
        na = eltypedegree(sep, j)
        nj = nn + na

        jres = neighbors(res, j)
        jsep = neighbors(sep, j)

        for w in jres
            idx[w] = j
        end

        for i in neighbors(chd, j)
            q = one(I); w = jres[q]

            for p in incident(sep, i)
                v = targets(sep)[p]

                while w < v && q < nn
                    q += one(I); w = jres[q]
                end

                while w < v
                    q += one(I); w = jsep[q - nn]
                end

                targets(rel)[p] = q
            end

            ma = eltypedegree(sep, i)

            jMptr -= one(I)
            jMval -= ma * ma
            jNval -= ma
        end

        if ispositive(na)
            jMptr += one(I)
            jMval += na * na
            jNval += na

            nMptr = max(nMptr, jMptr)
            nMval = max(nMval, jMval)
            nNval = max(nNval, jNval)
        end

        nFval = max(nFval, nj)
        Dp += nn * nn
        Lp += nn * na
    end

    Dptr[nv(res) + one(I)] = Dp + one(I)
    Lptr[nv(res) + one(I)] = Lp + one(I)

    return ChordalSymbolic(res, sep, rel, chd, pnt, idx, Dptr, Lptr, nMptr, nMval, nNval, nFval)
end

function Base.getproperty(S::ChordalSymbolic, d::Symbol)
    if d === :L
        return SymbolicChordalTriangular{:L}(S)
    elseif d === :U
        return SymbolicChordalTriangular{:U}(S)
    else
        return getfield(S, d)
    end
end

function Base.show(io::IO, S::T) where {T <: ChordalSymbolic}
    n = ncl(S)
    print(io, "$n×$n $T with $(nnz(S)) stored entries")
    return
end

function Base.show(io::IO, ::MIME"text/plain", S::T) where {T <: ChordalSymbolic}
    n = ncl(S)
    println(io, "$n×$n $T with $(nnz(S)) stored entries:")

    if n < 16
        print_matrix(io, S.L)
    else
        showsymbolic(io, S, Val(:L))
    end

    return
end

# ===== Braille Patterns =====

const brailleblocks = UInt16['⠁', '⠂', '⠄', '⡀', '⠈', '⠐', '⠠', '⢀']

function showsymbolic(io::IO, S::ChordalSymbolic, uplo::Val{Q}) where {Q}
    n = ncl(S)

    maxheight, maxwidth = displaysize(io)
    maxheight -= 4
    maxwidth ÷= 2

    if get(io, :limit, true)
        scaleheight = max(8, min(n, 2maxwidth, 4maxheight))
        scalewidth = max(4, min(n, 2maxwidth, 4maxheight))
    else
        scaleheight = max(8, n)
        scalewidth = max(4, n)
    end

    rowscale = max(1, scaleheight - 1) / max(1, n - 1)
    colscale = max(1, scalewidth - 1) / max(1, n - 1)

    braillegrid = Matrix{UInt16}(undef, (scalewidth - 1) ÷ 2 + 4, (scaleheight - 1) ÷ 4 + 1)
    braillegrid               .= '⠀'
    braillegrid[1,       :]   .= '⎢'
    braillegrid[end - 1, :]   .= '⎥'
    braillegrid[1,       1]    = '⎡'
    braillegrid[1,     end]    = '⎣'
    braillegrid[end - 1, 1]    = '⎤'
    braillegrid[end - 1, end]  = '⎦'
    braillegrid[end,     :]   .= '\n'

    for j in vertices(S.res)
        res = neighbors(S.res, j)
        sep = neighbors(S.sep, j)

        nn = eltypedegree(S.res, j)
        na = eltypedegree(S.sep, j)

        for w in oneto(nn)
            rw = res[w]

            if Q == :L
                rng = w:nn
            else
                rng = oneto(w)
            end

            for v in rng
                rv = res[v]
                setbraille!(braillegrid, rv, rw, rowscale, colscale)
            end

            for v in oneto(na)
                sv = sep[v]

                if Q == :L
                    setbraille!(braillegrid, sv, rw, rowscale, colscale)
                else
                    setbraille!(braillegrid, rw, sv, rowscale, colscale)
                end
            end
        end
    end

    for c in @view braillegrid[1:end - 1]
        print(io, Char(c))
    end
end

function setbraille!(braillegrid, row, col, rowscale, colscale)
    si = round(Int, (row - 1) * rowscale + 1)
    sj = round(Int, (col - 1) * colscale + 1)

    i =  (sj - 1) ÷ 2  + 2
    j =  (si - 1) ÷ 4  + 1
    b = ((sj - 1) % 2) * 4 + ((si - 1) % 4 + 1)

    braillegrid[i, j] |= brailleblocks[b]
end

# ===== Abstract Matrix Interface =====

function SparseArrays.nnz(S::ChordalSymbolic)
    return ndz(S) + nlz(S)
end

function ndz(S::ChordalSymbolic)
    n = nfr(S)
    return convert(Int, S.Dptr[n + 1]) - 1
end

function nlz(S::ChordalSymbolic)
    n = nfr(S)
    return convert(Int, S.Lptr[n + 1]) - 1
end

function ncl(S::ChordalSymbolic)
    return convert(Int, nov(S.res))
end

function nfr(S::ChordalSymbolic)
    return convert(Int, nv(S.res))
end

function fronts(S::ChordalSymbolic)
    return oneto(nfr(S))
end

function Base.size(S::ChordalSymbolic)
    n = ncl(S)
    return (n, n)
end

function Base.size(S::ChordalSymbolic, i::Integer)
    return ncl(S)
end

function Base.isstored(S::ChordalSymbolic, v::Integer, w::Integer, uplo::Val{UPLO}) where {UPLO}
    return !iszero(flatindex(S, v, w, uplo))
end

function flatindex(S::ChordalSymbolic{I}, v::Integer, w::Integer, uplo::Val) where {I <: Integer}
    return flatindex(S, convert(I, v), convert(I, w), uplo)
end

function flatindex(S::ChordalSymbolic{I}, v::I, w::I, ::Val{:L}) where {I <: Integer}
    poff = convert(I, ndz(S))

    if v ≥ w
        j = S.idx[w]

        nn = eltypedegree(S.res, j)
        na = eltypedegree(S.sep, j)

        res = neighbors(S.res, j)
        sep = neighbors(S.sep, j)

        w = w - first(res)

        if v in res
            v = v - first(res)
            return S.Dptr[j] + v + w * nn
        else
            i = searchsortedfirst(sep, v)

            if i ≤ na && sep[i] == v
                v = convert(I, i) - one(I)
                return S.Lptr[j] + poff + v + w * na
            end
        end
    end

    return zero(I)
end

function flatindex(S::ChordalSymbolic{I}, v::I, w::I, ::Val{:U}) where {I <: Integer}
    poff = convert(I, ndz(S))

    if v ≤ w
        j = S.idx[v]

        nn = eltypedegree(S.res, j)
        na = eltypedegree(S.sep, j)

        res = neighbors(S.res, j)
        sep = neighbors(S.sep, j)

        v = v - first(res)

        if w in res
            w = w - first(res)
            return S.Dptr[j] + v + w * nn
        else
            i = searchsortedfirst(sep, w)

            if i ≤ na && sep[i] == w
                w = convert(I, i) - one(I)
                return S.Lptr[j] + poff + v + w * nn
            end
        end
    end

    return zero(I)
end

function flatindices(S::ChordalSymbolic{I}, B::SparseMatrixCSC, ::Val{:L}) where {I}
    P = zeros(I, nnz(B)); poff = convert(I, ndz(S))
    res = S.res
    sep = S.sep

    rhi = one(I)

    @inbounds for j in vertices(res)
        pj = S.Dptr[j] - one(I)

        rlo = rhi
        rhi = pointers(res)[j + one(I)]

        for col in rlo:rhi - one(I)
            row = rlo

            for p in nzrange(B, col)
                wa = rowvals(B)[p]
                wa < rlo && continue
                wa >= rhi && break

                P[p] = pj += wa - row + one(I)
                row = wa + one(I)
            end

            pj += rhi - row
        end
    end

    shi = one(I)

    @inbounds for j in vertices(res)
        pj = S.Lptr[j] + poff - one(I)

        slo = shi
        shi = pointers(sep)[j + one(I)]
        slo >= shi && continue

        swr = targets(sep)[slo]
        nwr = targets(sep)[shi - one(I)] + one(I)

        for col in neighbors(res, j)
            k = slo

            for p in nzrange(B, col)
                row = targets(sep)[k]
                wa = rowvals(B)[p]
                wa < swr && continue
                wa >= nwr && break

                while row < wa
                    pj += one(I)
                    k += one(I)
                    row = targets(sep)[k]
                end

                P[p] = pj += one(I)
                k += one(I)
            end

            pj += shi - k
        end
    end

    return P
end

function flatindices(S::ChordalSymbolic{I}, B::SparseMatrixCSC, ::Val{:U}) where {I}
    P = zeros(I, nnz(B)); poff = convert(I, ndz(S))
    res = S.res
    sep = S.sep

    rhi = one(I)

    @inbounds for j in vertices(res)
        pj = S.Dptr[j] - one(I)

        rlo = rhi
        rhi = pointers(res)[j + one(I)]

        for col in rlo:rhi - one(I)
            row = rlo

            for p in nzrange(B, col)
                wa = rowvals(B)[p]
                wa < rlo && continue
                wa >= rhi && break

                P[p] = pj += wa - row + one(I)
                row = wa + one(I)
            end

            pj += rhi - row
        end

        pj = S.Lptr[j] + poff - one(I)

        for col in neighbors(sep, j)
            row = rlo

            for p in nzrange(B, col)
                wa = rowvals(B)[p]
                wa < rlo && continue
                wa >= rhi && break

                P[p] = pj += wa - row + one(I)
                row = wa + one(I)
            end

            pj += rhi - row
        end
    end

    return P
end

function flatindices(invp::AbstractVector{I}, S::ChordalSymbolic{I}, A::SparseMatrixCSC, uplo::Val{UPLO}; check::Bool=true) where {I, UPLO}
    if !check || ishermitian(A)
        return flatindices(invp, S, Hermitian(A, UPLO), uplo)
    elseif istril(A)
        return flatindices(invp, S, Hermitian(A, :L), uplo)
    elseif istriu(A)
        return flatindices(invp, S, Hermitian(A, :U), uplo)
    end

    error()
end

function flatindices(invp::AbstractVector{I}, S::ChordalSymbolic{I}, A::HermOrSym, uplo::Val{UPLO}) where {I, UPLO}
    m = convert(I, nnz(parent(A)))
    colptr = parent(A).colptr
    rowval = parent(A).rowval
    nzval = collect(oneto(m))
    B = SparseMatrixCSC(size(A)..., colptr, rowval, nzval)

    if UPLO === :L
        C = sympermute(B, invp, A.uplo, 'L')
    else
        C = sympermute(B, invp, A.uplo, 'U')
    end

    P = flatindices(S, C, uplo)
    fill!(nzval, zero(I))

    for (i, j) in zip(nonzeros(C), P)
        nzval[i] = j
    end

    return nzval
end
