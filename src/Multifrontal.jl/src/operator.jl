function opnorm1(A::ChordalTriangular{DIAG, :L, T}) where {DIAG, T}
    mapreducefront(max, A; init=zero(real(T))) do D, L
        maxsum = zero(real(T))

        for w in axes(D, 2)
            colsum = zero(real(T))

            for v in w:size(D, 1)
                colsum += abs(D[v, w])
            end

            for v in axes(L, 1)
                colsum += abs(L[v, w])
            end

            maxsum = max(maxsum, colsum)
        end

        return maxsum
    end
end

function opnorm1(A::ChordalTriangular{DIAG, :U, T}) where {DIAG, T}
    colsums = zeros(real(T), size(A, 1))

    foreachfront(A) do D, L, res, sep
        for w in eachindex(res)
            for v in w:length(res)
                col = res[v]
                colsums[col] += abs(D[w, v])
            end

            for v in eachindex(sep)
                col = sep[v]
                colsums[col] += abs(L[w, v])
            end
        end
    end

    return maximum(colsums)
end

function opnorminf(A::ChordalTriangular{DIAG, :L, T}) where {DIAG, T}
    rowsums = zeros(real(T), size(A, 1))

    foreachfront(A) do D, L, res, sep
        for w in eachindex(res)
            for v in w:length(res)
                row = res[v]
                rowsums[row] += abs(D[v, w])
            end

            for v in eachindex(sep)
                row = sep[v]
                rowsums[row] += abs(L[v, w])
            end
        end
    end

    return maximum(rowsums)
end

function opnorminf(A::ChordalTriangular{DIAG, :U, T}) where {DIAG, T}
    mapreducefront(max, A; init=zero(real(T))) do D, L
        maxsum = zero(real(T))

        for w in axes(D, 1)
            rowsum = zero(real(T))

            for v in w:size(D, 2)
                rowsum += abs(D[w, v])
            end

            for v in axes(L, 2)
                rowsum += abs(L[w, v])
            end

            maxsum = max(maxsum, rowsum)
        end

        return maxsum
    end
end

function opnorm1(A::AdjOrTransTri)
    opnorminf(parent(A))
end

function opnorminf(A::AdjOrTransTri)
    opnorm1(parent(A))
end

function isparallel!(c, A, b)
    mul!(c, adjoint(A), b)

    for v in c
        if abs(v) == length(b)
            return true
        end
    end

    return false
end

# A Block Algorithm for Matrix 1-Norm Estimation,
# with an Application to 1-Norm Psuedospectra
#
# Higham and Tisseur
#
# Algorithm 2.4: practical block 1-norm estimator
function opnormest1(F, inv::Bool=false; numcols::Int=min(2, size(F, 1)), maxiter::Int=5)
    @assert 0 < numcols <= size(F, 1)

    n = size(F, 1); T = eltype(F)

    # integer vector recording indices of used unit vectors ej
    hst = Vector{Int}(undef, maxiter * numcols)

    row = Vector{Int}(undef, n)
    sgncur = Matrix{T}(undef, n, numcols)
    sgnold = Matrix{T}(undef, n, numcols)
    sgnaux = Vector{T}(undef, numcols)
    B = Matrix{T}(undef, n, numcols)
    C = Matrix{T}(undef, n, numcols)
    h = Vector{real(T)}(undef, n)

    # choose starting matrix B with columns of unit 1-norm
    fill!(view(B, :, 1), one(T))
    fill!(sgncur, zero(T))

    for j in 2:numcols
        while true
            rand!(view(B, :, j), (-1, 1))

            if !isparallel!(view(sgnaux, 1:j - 1), view(B, :, 1:j - 1), view(B, :, j))
                break
            end
        end
    end

    rdiv!(B, n)

    estcur = zero(real(T))
    estold = zero(real(T))
    estcol = 0

    for k in 1:maxiter
        if inv
            ldiv!(C, F, B)
        else
            mul!(C, F, B)
        end

        estcur = zero(real(T))
        estcol = 0

        for i in 1:numcols
            y = norm(view(C, :, i), 1)

            if y > estcur
                estcur = y
                estcol = i
            end
        end

        if k >= 2
            if k == 2 || estcur > estold
                maxrow = hst[(k - 2) * numcols + estcol]
            end

            if estcur <= estold
                estcur = estold
                break
            end
        end

        estold  = estcur
        sgnold .= sgncur

        k == maxiter && break

        for j in 1:numcols
            for i in 1:n
                if iszero(C[i, j])
                    sgncur[i, j] = one(C[i, j])
                else
                    sgncur[i, j] = sign(C[i, j])
                end
            end
        end

        if T <: Real
            # if every column of sgncur is parallel to a column of sgnold, break. 
            allparallel = true

            for j in 1:numcols
                allparallel || break
                allparallel = isparallel!(sgnaux, sgnold, view(sgncur, :, j))
            end

            allparallel && break

            # ensure no column of sgncur is parallel to another column of sgncur or to sgnold
            for j in 1:numcols
                repeated = false

                if j > 1
                    repeated = isparallel!(view(sgnaux, 1:j - 1), view(sgncur, :, 1:j - 1), view(sgncur, :, j))
                end

                if !repeated
                    repeated = isparallel!(sgnaux, sgnold, view(sgncur, :, j))
                end

                while repeated
                    rand!(view(sgncur, :, j), (-1, 1))

                    if j > 1
                        repeated = isparallel!(view(sgnaux, 1:j - 1), view(sgncur, :, 1:j - 1), view(sgncur, :, j))
                    end

                    if !repeated
                        repeated = isparallel!(sgnaux, sgnold, view(sgncur, :, j))
                    end
                end
            end
        end

        if inv
            ldiv!(C, adjoint(F), sgncur)
        else
            mul!(C, adjoint(F), sgncur)
        end

        maxh = zero(real(T))

        for i in 1:n
            h[i] = norm(view(C, i, 1:numcols), Inf)
            maxh = max(maxh, h[i])
        end

        k >= 2 && h[maxrow] == maxh && break

        sortperm!(row, h; rev=true, initialized=false)

        if numcols > 1
            addctr = numcols
            elmctr = 0

            while addctr > 0 && elmctr < n
                elmctr += 1
                elmcur = row[elmctr]
                found = false

                for i in 1:numcols * (k - 1)
                    found && break
                    found = elmcur == hst[i]
                end

                if !found
                    addctr -= 1

                    for i in 1:elmcur - 1
                        B[i, numcols - addctr] = 0
                    end

                    B[elmcur, numcols - addctr] = 1

                    for i in elmcur + 1:n
                        B[i, numcols - addctr] = 0
                    end

                    hst[k * numcols - addctr] = elmcur
                else
                    if elmctr == numcols && addctr == numcols
                        break
                    end
                end
            end
        else
            hst[k] = row[1]
            fill!(B, zero(T))
            B[row[1], 1] = one(T)
        end
    end

    return estcur
end

# Finding Structure with Randomness:
# Probabilistic Algorithms for Constructing
# Approximate Matrix Decompositions
#
# Halko, Martinsson, and Tropp
#
# Algorithm 4.4: Randomized Subspace Iteration
function opnormest2(F, inv::Bool=false; numcols::Int=6, maxiter::Int=2)
    B = randn(eltype(F), size(F, 2), numcols)
    C = similar(B)

    if inv
        ldiv!(C, F, B)
    else
        mul!(C, F, B)
    end

    copyto!(B, qr!(C).Q)

    for _ in 1:maxiter
        if inv
            ldiv!(C, F', B)
        else
            mul!(C, F', B)
        end

        copyto!(B, qr!(C).Q)

        if inv
            ldiv!(C, F, B)
        else
            mul!(C, F, B)
        end

        copyto!(B, qr!(C).Q)
    end

    if inv
        ldiv!(C, F, B)
    else
        mul!(C, F, B)
    end

    return first(svdvals!(C))
end

function condest1(F; kw...)
    opnormest1(F; kw...) * opnormest1(F, true; kw...)
end

function condest1(F::MaybeAdjOrTransTri; kw...)
    opnorm1(F) * opnormest1(F, true; kw...)
end

function condest2(F; kw...)
    opnormest2(F; kw...) * opnormest2(F, true; kw...)
end
