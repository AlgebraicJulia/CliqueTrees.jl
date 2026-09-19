# ===== slu2! =====

function slu2!(s::S, A::AbstractMatrix) where {S <: AbstractSemiring}
    n = size(A, 1)
    @assert size(A, 2) == n

    @inbounds for i in 1:n
        #
        #   A = [ Aii Ain ]
        #       [ Ani Ann ]
        #
        if !(S <: IntegralQuantale)
            #
            #   Ani ← Ani Aii*
            #
            sAii = sstar(s, A[i, i])

            for k in i + 1:n
                A[k, i] = sprod(s, A[k, i], sAii)
            end
        end
        #
        #   Ann ← Ani Ain + Ann
        #
        for j in i + 1:n
            Aij = A[i, j]

            for k in i + 1:n
                A[k, j] = smuladd(s, A[k, i], Aij, A[k, j])
            end
        end
    end

    return A
end

# ===== slu! =====

function slu!(s::AbstractSemiring, A::AbstractMatrix)
    n = size(A, 1)
    @assert size(A, 2) == n

    if n <= THRESHOLD
        slu2!(s, A)
    else
        m = prevpow(2, n) >> 1

        A₁₁ = view(A,     1:m,     1:m)
        A₂₂ = view(A, m + 1:n, m + 1:n)
        A₁₂ = view(A,     1:m, m + 1:n)
        A₂₁ = view(A, m + 1:n,     1:m)
        #
        #   A₁₁ ← L₁₁ + U₁₁
        #
        slu!(s, A₁₁)
        #
        #   A₁₂ ← L₁₁* A₁₂
        #   A₂₁ ← A₂₁ U₁₁*
        #
        strsx!(s, Val(:L), Val(:L), A₁₁, A₁₂)
        strsx!(s, Val(:R), Val(:U), A₁₁, A₂₁)
        #
        #   A₂₂ ← A₂₁ A₁₂ + A₂₂
        #
        sgemx!(s, A₂₂, A₂₁, A₁₂)
        #
        #   A₂₂ ← L₂₂ + U₂₂
        #
        slu!(s, A₂₂)
    end

    return A
end
