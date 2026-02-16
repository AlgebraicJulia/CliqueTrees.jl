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

### Basic Usage

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

function symbolic(A::SparseMatrixCSC{<:Any, I}; kw...) where {I <: Integer}
    return symbolic(BipartiteGraph(A); kw...)
end

function symbolic(graph::AbstractGraph{I}; kw...) where {I <: Integer}
    perm, tree = cliquetree(graph; kw...)
    return FVector{I}(perm), ChordalSymbolic(tree)
end

function symbolic(A::AbstractMatrix, clique::AbstractVector; kw...)
    return symbolic(sparse(A), clique; kw...)
end

function symbolic(A::SparseMatrixCSC, clique::AbstractVector; kw...)
    return symbolic(BipartiteGraph(A), clique; kw...)
end

function ChordalSymbolic(tree::CliqueTree{I, I}) where {I <: Integer}
    res = residuals(tree)
    sep = separators(tree)

    reltgt = FVector{I}(undef, ne(sep))
    rel = BipartiteGraph(nov(sep), nv(sep), ne(sep), pointers(sep), reltgt)

    # compute workspace sizes
    nMptr = jMptr = one(I)
    nMval = jMval = zero(I)
    nNval = jNval = zero(I)
    nFval = zero(I)
    Dp = zero(I)
    Lp = zero(I)

    # compute column to supernode mapping
    idx = FVector{I}(undef, nov(res))
    Dptr = FVector{I}(undef, nv(res) + one(I))
    Lptr = FVector{I}(undef, nv(res) + one(I))

    for j in vertices(res)
        Dptr[j] = Dp + one(I)
        Lptr[j] = Lp + one(I)

        # nn = |res(j)|
        nn = eltypedegree(res, j)

        # na = |sep(j)|
        na = eltypedegree(sep, j)

        # nj = |bag(j)|
        nj = nn + na

        bag = tree[j]

        for w in residual(bag)
            idx[w] = j
        end

        for i in childindices(tree, j)
            q = one(I); w = bag[q]

            for p in incident(sep, i)
                v = targets(sep)[p]

                while w < v
                    q += one(I); w = bag[q]
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

    chd = tree.tree.tree.graph
    pnt = tree.tree.tree.tree.prnt

    return ChordalSymbolic(res, sep, rel, chd, pnt, idx, Dptr, Lptr, nMptr, nMval, nNval, nFval)
end

function Base.getproperty(S::ChordalSymbolic, d::Symbol)
    if d === :L
        return ChordalTriangular{:L}(S)
    elseif d === :U
        return ChordalTriangular{:U}(S)
    else
        return getfield(S, d)
    end
end

function Base.show(io::IO, S::T) where {T <: ChordalSymbolic}
    n = size(S, 1)
    print(io, "$n×$n $T with $(nnz(S)) stored entries")
    return
end

function Base.show(io::IO, ::MIME"text/plain", S::T) where {T <: ChordalSymbolic}
    n = size(S, 1)
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
    n = size(S, 1)

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

function SparseArrays.nnz(S::ChordalSymbolic{I}) where {I <: Integer}
    n = nv(S.res)
    nRval = S.Dptr[n + one(I)] - one(I)
    nLval = S.Lptr[n + one(I)] - one(I)
    nnz = half(nRval + nov(S.res)) + nLval
    return convert(Int, nnz)
end

function Base.size(S::ChordalSymbolic)
    n = convert(Int, nov(S.res))
    return (n, n)
end

function Base.size(S::ChordalSymbolic, i::Integer)
    return size(S)[i]
end
