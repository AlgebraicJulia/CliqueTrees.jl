function lacpy!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    LAPACK.lacpy!(A, B, 'L')
    return
end

function potrf!(A::AbstractMatrix{T}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    LAPACK.potrf!('L', A)
    return
end

function trsm!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    BLAS.trsm!('R', 'L', 'T', 'N', one(T), A, B)
    return
end

function syrk!(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    BLAS.syrk!('L', 'N', -one(T), A, one(T), B)
    return
end

function trsv!(A::AbstractMatrix{T}, b::AbstractVector{T}, tA::Val{false}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    BLAS.trsv!('L', 'N', 'N', A, b)
    return
end

function trsv!(A::AbstractMatrix{T}, b::AbstractVector{T}, tA::Val{true}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    BLAS.trsv!('L', 'T', 'N', A, b)
    return
end

function gemv!(A::AbstractMatrix{T}, x::AbstractVector{T}, y::AbstractVector{T}, tA::Val{false}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    BLAS.gemv!('N', -one(T), A, x, one(T), y)
    return
end

function gemv!(A::AbstractMatrix{T}, x::AbstractVector{T}, y::AbstractVector{T}, tA::Val{true}) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    BLAS.gemv!('T', -one(T), A, x, one(T), y)
    return
end

struct CholFact{T, I}
    tree::CliqueTree{I, I}
    perm::Vector{I}
    blkptr::FVector{I}
    blkval::FVector{T}
    frtval::FVector{T}
end

function cholesky(matrix::AbstractMatrix; alg::EliminationAlgorithm=AMF())
    fact = cholesky(matrix, alg)
    return fact
end

function cholesky(matrix::AbstractMatrix, alg::EliminationAlgorithm)
    return cholesky!(sparse(matrix), alg)
end

function cholesky!(matrix::SparseMatrixCSC; alg::EliminationAlgorithm=AMF())
    fact = cholesky!(matrix, alg)
    return fact
end

function cholesky!(matrix::SparseMatrixCSC{T, I}, alg::EliminationAlgorithm) where {T, I}
    perm, tree = cliquetree(matrix; alg)
    tril!(permute!(matrix, perm, perm))

    residual = residuals(tree)
    separator = separators(tree)

    ns = nsmax = namax = njmax = blkln = zero(I)

    for j in vertices(separator)
        for i in childindices(tree, j)
            ns -= one(I)
        end

        if !isnothing(parentindex(tree, j))
            ns += one(I)
        end

        nn = eltypedegree(residual, j)
        na = eltypedegree(separator, j)
        nj = nn + na

        nsmax = max(nsmax, ns)
        namax = max(namax, na)
        njmax = max(njmax, nj)

        blkln = blkln + nn * nj
    end

    treln = nv(separator)
    relln = ne(separator)
    updln = nsmax * namax * namax 
    frtln = njmax * njmax

    blkptr = FVector{I}(undef, treln + one(I))
    relidx = FVector{I}(undef, relln)

    updval = FVector{T}(undef, updln)
    blkval = FVector{T}(undef, blkln)
    frtval = FVector{T}(undef, frtln)

    relptr = pointers(separator)
    mapping = BipartiteGraph(njmax, treln, relln, relptr, relidx)

    cholesky!_impl!(namax, mapping, blkptr,
        updval, blkval, frtval, tree, matrix)

    return CholFact(tree, perm, blkptr, blkval, frtval)
end 

function cholesky!_impl!(
        namax::I,
        mapping::BipartiteGraph{I, I},        
        blkptr::AbstractVector{I},
        updval::AbstractVector{T},
        blkval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I},
        matrix::SparseMatrixCSC{T, I},
    ) where {T, I}
    separator = separators(tree)
    relidx = targets(mapping)
    ns = zero(I)

    cholesky!_init!(relidx, blkptr,
        blkval, tree, matrix)

    for j in vertices(separator)
        ns = cholesky!_loop!(namax, mapping, blkptr,
            updval, blkval, frtval, tree, ns, j)
    end

    return
end

function cholesky!_init!(
        relidx::AbstractVector{I},
        blkptr::AbstractVector{I},
        blkval::AbstractVector{T},
        tree::CliqueTree{I, I},
        matrix::SparseMatrixCSC{T, I},
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)
    treln = nv(separator)
    p = q = one(I)

    for j in vertices(separator)
        blkptr[j] = p

        res = neighbors(residual, j)
        sep = neighbors(separator, j)
        bag = Clique(res, sep)

        nn = eltypedegree(residual, j)
        na = eltypedegree(separator, j)
        nj = nn + na

        for v in res
            k = one(I)

            for q in nzrange(matrix, v)
                w = rowvals(matrix)[q]

                while bag[k] < w
                    blkval[p] = zero(T)
                    k += one(I)
                    p += one(I)
                end

                blkval[p] = nonzeros(matrix)[q]
                k += one(I)
                p += one(I) 
            end

            while k <= nj
                blkval[p] = zero(T)
                k += one(I)
                p += one(I)
            end
        end

        pj = parentindex(tree, j)

        if !isnothing(pj)
            pbag = tree[pj]    

            k = one(I)

            for w in sep
                while pbag[k] < w
                    k += one(I)
                end

                relidx[q] = k
                q += one(I)
                k += one(I)
            end
        end
    end

    blkptr[treln + one(I)] = p
    return
end

function cholesky!_loop!(
        namax::I,
        mapping::BipartiteGraph{I, I},        
        blkptr::AbstractVector{I},
        updval::AbstractVector{T},
        blkval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I},
        ns::I,
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = eltypedegree(residual, j)

    # na is the size of the separator at node j
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # F is the frontal matrix at node j
    #
    #     F = [ F₁  F₂  ]
    #
    #       = [ F₁₁ F₁₂ ]
    #         [ F₂₁ F₂₂ ]
    #
    # only the lower triangular part is used
    F = reshape(view(frtval, oneto(nj * nj)), nj, nj)
    F₁ =  view(F, oneto(nj),      oneto(nn))
    F₁₁ = view(F, oneto(nn),      oneto(nn))
    F₂₁ = view(F, nn + one(I):nj, oneto(nn))
    F₂₂ = view(F, nn + one(I):nj, nn + one(I):nj)

    # B is part of the Cholesky factor
    #
    #          res(j)
    #     B = [ B₁₁  ] res(j)
    #         [ B₂₁  ] sep(j)
    #
    pstrt = blkptr[j]
    pstop = blkptr[j + one(I)]
    B = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)

    # copy B into F₁
    #
    #     F₁₁ ← B₁₁
    #     F₂₁ ← B₂₁
    #
    lacpy!(F, B); fill!(F₂₂, zero(T))

    for i in Iterators.reverse(childindices(tree, j))
        cholesky!_add_update!(F, namax, ns, i, mapping, updval)
        ns -= one(I)
    end

    # factorize F₁₁ as
    #
    #     F₁₁ = L₁₁ L₁₁ᵀ
    #
    # and store F₁₁ ← L₁₁
    potrf!(F₁₁)

    if ispositive(na)
        # solve for L₂₁ in
        #
        #     L₂₁ L₁₁ᵀ = F₂₁
        #
        # and store F₂₁ ← L₂₁
        trsm!(F₁₁, F₂₁)
    
        # compute
        #
        #    U₂₂ = F₂₂ - L₂₁ L₂₁ᵀ
        #
        # and store F₂₂ ← U₂₂
        syrk!(F₂₁, F₂₂)

        ns += one(I)
        pstrt = namax * namax * (ns - one(I)) + one(I)
        pstop = pstrt + na * na
        U₂₂ = reshape(view(updval, pstrt:pstop - one(I)), na, na)
        lacpy!(U₂₂, F₂₂)
    end
 
    # copy F₁  into B
    #
    #     B₁₁ ← F₁₁
    #     B₂₁ ← F₂₁
    #
    lacpy!(B, F₁)
    return ns
end

function cholesky!_add_update!(
        F::AbstractMatrix{T},
        namax::I,
        ns::I,
        i::I,
        mapping::BipartiteGraph{I, I},
        updval::AbstractVector{T},
    ) where {T, I}
    # na is the size of the separator at node i.
    #
    #     na = | sep(i) |
    #
    na = eltypedegree(mapping, i)

    # ind is the subset inclusion
    #
    #     ind: sep(i) → sep(parent(i))
    #
    ind = neighbors(mapping, i)

    # U is the na × na update matrix for node i.
    pstrt = namax * namax * (ns - one(I)) + one(I)
    pstop = pstrt + na * na
    U = reshape(view(updval, pstrt:pstop - one(I)), na, na)

    # for all uj in sep(i) ...
    for uj in oneto(na)
        # let fj = ind(uj)
        fj = ind[uj]

        # for all ui in sep(i) ...
        for ui in oneto(na)
            # let fi = ind(ui)
            fi = ind[ui]

            # compute the sum
            #
            #     F[fi, fj] + U[ui, uj]
            #
            # and assign it to F[fi, fj]
            F[fi, fj] += U[ui, uj]
        end
    end

    return
end

function LinearAlgebra.ldiv(F::CholFact{T}, b::AbstractVector) where {T}
    n = length(b)
    x = Vector{T}(undef, n)

    for i in oneto(n)
        x[i] = b[i]
    end

    return ldiv!(F, x)
end

function LinearAlgebra.ldiv!(F::CholFact{T}, b::AbstractVector{T}) where {T}
    tree = F.tree
    perm = F.perm
    blkptr = F.blkptr
    blkval = F.blkval
    frtval = F.frtval

    residual = residuals(tree)
    separator = separators(tree)

    n = nov(separator)
    x = FVector{T}(undef, n)

    for v in outvertices(separator) 
        x[v] = b[perm[v]]
    end
    
    for j in vertices(separator)
        ldiv!_loop_fwd!(blkptr, blkval, frtval, tree, x, j)
    end
    
    for j in reverse(vertices(separator))
        ldiv!_loop_bwd!(blkptr, blkval, frtval, tree, x, j)
    end

    for v in outvertices(separator)
        b[perm[v]] = x[v]
    end
    
    return b
end

function ldiv!_loop_fwd!(
        blkptr::AbstractVector{I},
        blkval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        x::AbstractVector{T},
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    bag = tree[j] 
    
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    pstrt = blkptr[j]
    pstop = blkptr[j + one(I)]
    L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    L₁₁ = view(L, oneto(nn),      oneto(nn))
    L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of the RHS
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    X = view(frtval, oneto(nj))
    X₁ = view(X, oneto(nn))
    X₂ = view(X, nn + one(I):nj)

    for k in oneto(nj)
        X[k] = x[bag[k]]
    end

    trsv!(L₁₁, X₁, Val(false))
    
    if ispositive(na)
        gemv!(L₂₁, X₁, X₂, Val(false))
    end

    for k in oneto(nj)
        x[bag[k]] = X[k]
    end

    return
end

function ldiv!_loop_bwd!(
        blkptr::AbstractVector{I},
        blkval::AbstractVector{T},
        frtval::AbstractVector{T},
        tree::CliqueTree{I, I}, 
        x::AbstractVector{T},
        j::I,
    ) where {T, I}
    residual = residuals(tree)
    separator = separators(tree)

    # nn is the size of the residual at node j
    #
    #     nn = | res(j) |
    #
    nn = eltypedegree(residual, j)

    # na is the size of the separator at node j.
    #
    #     na = | sep(j) |
    #
    na = eltypedegree(separator, j)

    # nj is the size of the bag at node j
    #
    #     nj = | bag(j) |
    #
    nj = nn + na

    # bag is the bag at node j
    bag = tree[j]        
 
    # L is part of the Cholesky factor
    #
    #          res(j)
    #     L = [ L₁₁  ] res(j)
    #         [ L₂₁  ] sep(j)
    #
    pstrt = blkptr[j]
    pstop = blkptr[j + one(I)]
    L = reshape(view(blkval, pstrt:pstop - one(I)), nj, nn)
    
    L₁₁ = view(L, oneto(nn),      oneto(nn))
    L₂₁ = view(L, nn + one(I):nj, oneto(nn))

    # X is part of the RHS
    #
    #     X = [ X₁ ] res(j)
    #         [ X₂ ] sep(j)
    #
    X = view(frtval, oneto(nj))
    X₁ = view(X, oneto(nn))
    X₂ = view(X, nn + one(I):nj)

    for k in oneto(nj)
        X[k] = x[bag[k]]
    end
    
    if ispositive(na)
        gemv!(L₂₁, X₂, X₁, Val(true))
    end

    trsv!(L₁₁, X₁, Val(true))

    for k in oneto(nj)
        x[bag[k]] = X[k]
    end

    return
end
