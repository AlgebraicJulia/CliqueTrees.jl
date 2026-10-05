const HYTPQRT_BASE = 8

#
# Update A₁ and A₂
#
#   [ A₁ A₂ ] ← [ A₁ A₂ ] [ Q₁₁ Q₁₂ ]
#                         [ Q₂₁ Q₂₂ ]
#
# where Q is the hyperbolic matrix
#
#   Q = [ Q₁₁ Q₁₂ ] = [ I     ] - [ I     ] T₁₁ [ I  V₁₂ ]
#       [ Q₂₁ Q₂₂ ]   [     I ]   [ -V₁₂ᴴ ]
#
function hytprfb!(A1::AbstractMatrix, A2::AbstractMatrix, V12::AbstractMatrix, T11::AbstractMatrix, Z::AbstractVector, ::Val{UPLO}, ::Val{UPDN}) where {UPLO, UPDN}
    if UPLO === :L
        m = size(A1, 1)
        n = size(V12, 1)
    else
        m = size(V12, 2)
        n = size(A1, 2)
    end

    if UPDN === :D
        σ = -1
    else
        σ = 1
    end

    if 0 < m
        Z1 = reshape(view(Z, 1:m * n), m, n)
        #
        #   Z₁ ← (A₁ - A₂ V₁₂ᴴ) T₁₁
        #
        copyto!(Z1, A1)

        if UPLO === :L
            gemm!(Val(:N), Val(:C), σ, A2, V12, 1, Z1)
            trmm!(Val(:R), Val(:U), Val(:N), Val(:N), 1, T11, Z1)
        else
            gemm!(Val(:C), Val(:N), σ, V12, A2, 1, Z1)
            trmm!(Val(:L), Val(:U), Val(:C), Val(:N), 1, T11, Z1)
        end
        #
        #   A₁ ← A₁ - Z₁
        #
        axpy!(-1, Z1, A1)
        #
        #   A₂ ← A₂ - Z₁ V₁₂
        #
        if UPLO === :L
            gemm!(Val(:N), Val(:N), -1, Z1, V12, 1, A2)
        else
            gemm!(Val(:N), Val(:N), -1, V12, Z1, 1, A2)
        end
    end

    return
end

@generated function hytpqxt2_refl!(A::AbstractMatrix, B::AbstractMatrix, τ, j::Int, ::Val{P}, ::Val{:L}, ::Val{UPDN}) where {P, UPDN}
    if UPDN === :D
        s = :(-cBjk)
    else
        s = :(cBjk)
    end

    quote
        @inbounds begin
            @nexprs $P i -> A_i_j = A[j + i, j]

            @simd for k in axes(B, 2)
                cBjk = conj(B[j, k])
                @nexprs $P i -> A_i_j = muladd($s, B[j + i, k], A_i_j)
            end

            @nexprs $P i -> A[j + i, j] -= A_i_j *= τ

            @simd for k in axes(B, 2)
                Bjk = B[j, k]
                @nexprs $P i -> B[j + i, k] = muladd(-A_i_j, Bjk, B[j + i, k])
            end
        end
    end
end

@generated function hytpqxt2_refl!(A::AbstractMatrix, B::AbstractMatrix, τ, j::Int, ::Val{P}, ::Val{:U}, ::Val{UPDN}) where {P, UPDN}
    if UPDN === :D
        s = :(-cBkj)
    else
        s = :(cBkj)
    end

    quote
        @inbounds begin
            @nexprs $P i -> A_j_i = A[j, j + i]

            @simd for k in axes(B, 1)
                cBkj = conj(B[k, j])
                @nexprs $P i -> A_j_i = muladd($s, B[k, j + i], A_j_i)
            end

            @nexprs $P i -> A[j, j + i] -= A_j_i *= τ

            @simd for k in axes(B, 1)
                Bkj = B[k, j]
                @nexprs $P i -> B[k, j + i] = muladd(-A_j_i, Bkj, B[k, j + i])
            end
        end
    end
end

function hytpqxt2_refl!(A::AbstractMatrix, B::AbstractMatrix, τ, j::Int, uplo::Val, updn::Val)
    m = size(A, 1); P = m - j
    P == 0 && return
    P == 1 && return hytpqxt2_refl!(A, B, τ, j, Val(1), uplo, updn)
    P == 2 && return hytpqxt2_refl!(A, B, τ, j, Val(2), uplo, updn)
    P == 3 && return hytpqxt2_refl!(A, B, τ, j, Val(3), uplo, updn)
    P == 4 && return hytpqxt2_refl!(A, B, τ, j, Val(4), uplo, updn)
    P == 5 && return hytpqxt2_refl!(A, B, τ, j, Val(5), uplo, updn)
    P == 6 && return hytpqxt2_refl!(A, B, τ, j, Val(6), uplo, updn)
    P == 7 && return hytpqxt2_refl!(A, B, τ, j, Val(7), uplo, updn)
    error()
end

@generated function hytpqxt2_tcol!(B::AbstractMatrix, W::AbstractMatrix, τ, j::Int, ::Val{P}, ::Val{UPLO}, ::Val{UPDN}) where {P, UPLO, UPDN}
    if UPDN === :D
        s = :τ
    else
        s = :(-τ)
    end

    quote
        T = promote_eltype(B, W)

        @inbounds begin
            @nexprs $P i -> t_i = zero(T)

            if UPLO === :L
                @simd for k in axes(B, 2)
                    cBjk = conj(B[j, k])
                    @nexprs $P i -> t_i = muladd(B[i, k], cBjk, t_i)
                end
            else
                @simd for k in axes(B, 1)
                    Bkj = B[k, j]
                    @nexprs $P i -> t_i = muladd(conj(B[k, i]), Bkj, t_i)
                end
            end

            @nexprs $P i -> t_i *= $s

            t = @ntuple $P i -> t_i

            $(ntuple(P) do i
                Wij = :(W[$i, $i] * t[$i])

                for k in i + 1:P
                    Wij = :(muladd(W[$i, $k], t[$k], $Wij))
                end

                :(W[$i, j] = $Wij)
            end...)
        end
    end
end

function hytpqxt2_tcol!(B::AbstractMatrix, W::AbstractMatrix, τ, j::Int, uplo::Val, updn::Val)
    P = j - 1
    P == 1 && return hytpqxt2_tcol!(B, W, τ, j, Val(1), uplo, updn)
    P == 2 && return hytpqxt2_tcol!(B, W, τ, j, Val(2), uplo, updn)
    P == 3 && return hytpqxt2_tcol!(B, W, τ, j, Val(3), uplo, updn)
    P == 4 && return hytpqxt2_tcol!(B, W, τ, j, Val(4), uplo, updn)
    P == 5 && return hytpqxt2_tcol!(B, W, τ, j, Val(5), uplo, updn)
    P == 6 && return hytpqxt2_tcol!(B, W, τ, j, Val(6), uplo, updn)
    P == 7 && return hytpqxt2_tcol!(B, W, τ, j, Val(7), uplo, updn)
    error()
end

function hytpqxt2!(A11::AbstractMatrix, A12::AbstractMatrix, W::AbstractMatrix, uplo::Val{:L}, updn::Val{UPDN}) where {UPDN}
    T = promote_eltype(A11, A12, W)

    m = size(A11, 1)
    n = size(A12, 2)

    info = 0; j = 1

    @inbounds while info == 0 && j ≤ m
        #
        #             j
        #         [   ⋮   ]   [  ⋮  ]
        #   A = j [ ⋯ α ⋯ ]   [  cᵀ ]
        #         [ ⋯ b ⋯ ]   [  D  ]
        #
        # construct the hyperbolic matrix
        #
        #   H := [ 1    ] - τ [  1 ] [ 1 vᴴ ]
        #        [    I ]     [ -v ]
        #
        # such that [ α cᵀ ] H = [ β 0 ⋯ 0 ].
        #
        ν = zero(real(T))

        @simd for k in 1:n
            ν += abs2(A12[j, k])
        end

        α = real(A11[j, j])

        if UPDN === :D
            β² = α^2 - ν
        else
            β² = α^2 + ν
        end

        if iszero(ν)
            for i in 1:j
                W[i, j] = 0
            end
        elseif ispositive(β²)
            #
            # compute the τ and v, and apply H to
            # the pivot row:
            #
            #   [ α cᵀ ] ← [ α cᵀ ] H
            #
            if UPDN === :D
                A11[j, j] = β = copysign(sqrt(β²), α)
                δ = (α + β) / ν
            else
                A11[j, j] = β = -copysign(sqrt(β²), α)
                δ = inv(α - β)
            end

            τ = -inv(β * δ)

            for k in 1:n
                A12[j, k] *= δ
            end
            #
            # apply H to the trailing rows
            #
            #   [ b D ] ← [ b D ] H
            #
            hytpqxt2_refl!(A11, A12, τ, j, uplo, updn)
            #
            # compute the compact WY column
            #
            #   τ T₁₁ V vᴴ
            #
            if j > 1
                hytpqxt2_tcol!(A12, W, τ, j, uplo, updn)
            end

            W[j, j] = τ
        else
            info = j
        end

        j += 1
    end

    return info
end

function hytpqxt2!(A11::AbstractMatrix, A12::AbstractMatrix, W::AbstractMatrix, uplo::Val{:U}, updn::Val{UPDN}) where {UPDN}
    T = promote_eltype(A11, A12, W)

    m = size(A11, 1)
    n = size(A12, 1)

    info = 0; j = 1

    @inbounds while info == 0 && j ≤ m
        #
        #             j
        #         [   ⋮ ⋮  ]
        #       j [ ⋯ α bᵀ ]
        #    A =  [   ⋮ ⋮  ]
        #
        #         [ ⋯ c D  ]
        #
        # construct the hyperbolic matrix
        #
        #   H := [ 1    ] - τ [  1 ] [ 1 vᴴ ]
        #        [    I ]     [ -v ]
        #
        # such that
        #
        #   H [ α ] = [ β ]
        #     [ c ]   [ 0 ]
        #             [ ⋮ ]
        #
        ν = zero(real(T))

        @simd for k in 1:n
            ν += abs2(A12[k, j])
        end

        α = real(A11[j, j])

        if UPDN === :D
            β² = α^2 - ν
        else
            β² = α^2 + ν
        end

        if iszero(ν)
            for i in 1:j
                W[i, j] = 0
            end
        elseif ispositive(β²)
            #
            # compute the τ and v, and apply H to
            # the pivot column:
            #
            #   [ α ] ← H [ α ]
            #   [ c ]     [ c ]
            #
            if UPDN === :D
                A11[j, j] = β = copysign(sqrt(β²), α)
                δ = (α + β) / ν
            else
                A11[j, j] = β = -copysign(sqrt(β²), α)
                δ = inv(α - β)
            end

            τ = -inv(β * δ)

            for k in 1:n
                A12[k, j] *= δ
            end
            #
            # apply H to the trailing columns
            #
            #   [ bᵀ ] ← H [ bᵀ ]
            #   [ D  ]     [ D  ]
            #
            hytpqxt2_refl!(A11, A12, τ, j, uplo, updn)
            #
            # compute the compact WY column
            #
            #   τ T₁₁ Vᴴ v
            #
            if j > 1
                hytpqxt2_tcol!(A12, W, τ, j, uplo, updn)
            end

            W[j, j] = τ
        else
            info = j
        end

        j += 1
    end

    return info
end

# Compute a hyperbolic matrix
#
#   Q = [ Q₁₁ Q₁₂ ] = [ I     ] - [ I     ] T₁₁ [ I  V₁₂ ]
#       [ Q₂₁ Q₂₂ ]   [     I ]   [ -V₁₂ᴴ ]
#
# such that
#
#   [ A₁₁ A₁₂ ] [ Q₁₁ Q₁₂ ] = [ L₁₁   ]
#               [ Q₂₁ Q₂₂ ]
#
# where A₁ and L₁ are lower triangular. Updates A₁₁, A₁₂, and W₁₁
#
#   A₁₁ ← L₁₁
#   A₁₂ ← V₁₂
#   W₁₁ ← T₁₁
#
function hytpqxt3!(A11::AbstractMatrix, A12::AbstractMatrix, W::AbstractMatrix, Z::AbstractVector, uplo::Val{UPLO}, updn::Val{UPDN}) where {UPLO, UPDN}
    m = size(A11, 1)

    if UPLO === :L
        n = size(A12, 2)
    else
        n = size(A12, 1)
    end

    if m ≤ HYTPQRT_BASE
        info = hytpqxt2!(A11, A12, W, uplo, updn)
    else
        k = m >> 1
        #
        #   [ A₁₁ A₁₂ ] = [ B₁₁     V₁₃ ]
        #                 [ B₂₁ B₂₂ V₂₃ ]
        #
        B11 = view(A11,     1:k,     1:k)
        B22 = view(A11, k + 1:m, k + 1:m)

        if UPLO === :L
            B21 = view(A11, k + 1:m,     1:k)
            V13 = view(A12,     1:k,     1:n)
            V23 = view(A12, k + 1:m,     1:n)
        else
            B21 = view(A11,     1:k, k + 1:m)
            V13 = view(A12,     1:n,     1:k)
            V23 = view(A12,     1:n, k + 1:m)
        end
        #
        #   W₁₁ = [ T₁₁ T₁₂ ]
        #         [     T₂₂ ]
        #
        T11 = view(W,       1:k,     1:k)
        T22 = view(W,   k + 1:m, k + 1:m)
        T12 = view(W,       1:k, k + 1:m)
        #
        # compute a hyperbolic matrix
        #
        #   R = [ R₁₁ R₁₃ ] = [ I     ] - [ I     ] T₁₁ [ I  V₁₃ ]
        #       [ R₃₁ R₃₃ ]   [     I ]   [ -V₁₃ᴴ ]
        #
        # such that
        #
        #   [ B₁₁ V₁₃ ] [ R₁₁ R₁₃ ] = [ L₁₁    ]
        #               [ R₃₁ R₃₃ ]
        #
        # and update
        #
        #   B₁₁ ← L₁₁
        #
        info = hytpqxt3!(B11, V13, T11, Z, uplo, updn)

        if info == 0
            #
            # update the second row
            #
            #   [ B₂₁ V₂₃ ] ← [ B₂₁ V₂₃ ] [ R₁₁ R₁₃ ]
            #                             [ R₃₁ R₃₃ ]
            #
            hytprfb!(B21, V23, V13, T11, Z, uplo, updn)
            #
            # compute a hyperbolic matrix
            #
            #   S = [ S₂₂ S₂₃ ] = [ I     ] - [ I     ] T₂₂ [ I  V₂₃ ]
            #       [ S₃₂ S₃₃ ]   [     I ]   [ -V₂₃ᴴ ]
            #
            # such that
            #
            #   [ B₂₂ V₂₃ ] [ S₂₂ S₂₃ ] = [ L₂₂    ]
            #               [ S₃₂ S₃₃ ]
            #
            # and update
            #
            #   B₂₂ ← L₂₂
            #
            info = hytpqxt3!(B22, V23, T22, Z, uplo, updn)

            if info == 0
                #
                #   T₁₂ ← T₁₁ V₁₃ V₂₃ᴴ T₂₂
                #
                if UPDN === :D
                    σ = 1
                else
                    σ = -1
                end

                if UPLO === :L
                    gemm!(Val(:N), Val(:C), σ, V13, V23, 0, T12)
                else
                    gemm!(Val(:C), Val(:N), σ, V13, V23, 0, T12)
                end

                trmm!(Val(:L), Val(:U), Val(:N), Val(:N), 1, T11, T12)
                trmm!(Val(:R), Val(:U), Val(:N), Val(:N), 1, T22, T12)
            else
                info += k
            end
        end
    end

    return info
end

#
# Compute a hyperbolic matrix
#
#   Q = [ Q₁₁ Q₁₂ ] = [ I     ] - [ I     ] T₁₁ [ I  V₁₂ ]
#       [ Q₂₁ Q₂₂ ]   [     I ]   [ -V₁₂ᴴ ]
#
# such that
#
#   [ A₁₁ A₁₂ ] [ Q₁₁ Q₁₂ ] = [ L₁₁    ]
#               [ Q₂₁ Q₂₂ ]
#
# where A₁₁ and L₁₁ are lower triangular. Updates
#
#   A₁₁ ← L₁₁
#   A₁₂ ← V₁₂
#   W   ← T₁₁
#
function hytpqxt!(A11::AbstractMatrix, A12::AbstractMatrix, W::AbstractMatrix, Z::AbstractVector, uplo::Val{UPLO}, updn::Val{UPDN}) where {UPLO, UPDN}
    m1 = size(A11, 1)

    if UPLO === :L
        n2 = size(A12, 2)
    else
        n2 = size(A12, 1)
    end

    info = 0

    if 0 < n2
        wsize = size(W, 1)
        jstrt = 1

        @inbounds while info == 0 && jstrt ≤ m1
            jstop = min(jstrt + wsize - 1, m1)
            jsize = jstop - jstrt + 1
            #
            #              j
            #         [    ⋮    ]   [  ⋮  ]
            #   A = j [ ⋯ Ajj ⋯ ]   [ Aj2 ]
            #         [ ⋯ Anj ⋯ ]   [ An2 ]
            #
            Ajj = view(A11, jstrt:jstop, jstrt:jstop)

            if UPLO === :L
                Aj2 = view(A12, jstrt:jstop,         1:n2)
                Anj = view(A11, jstop + 1:m1, jstrt:jstop)
                An2 = view(A12, jstop + 1:m1,        1:n2)
            else
                Aj2 = view(A12,        1:n2,  jstrt:jstop)
                Anj = view(A11, jstrt:jstop, jstop + 1:m1)
                An2 = view(A12,        1:n2, jstop + 1:m1)
            end

            Wjj = view(W, 1:jsize, jstrt:jstop)
            #
            # compute a hyperbolic matrix
            #
            #   Q = [ Qjj Qj2 ] = [ I     ] - [ I     ] Tjj [ I  Vj2 ]
            #       [ Q2j Q22 ]   [     I ]   [ -Vj2ᴴ ]
            #
            # such that
            #
            #   [ Ajj Aj2 ] [ Qjj Qj2 ] = [ Ljj    ]
            #               [ Q2j Q22 ]
            #
            # and update
            #
            #   Ajj ← Ljj
            #   Aj2 ← Vj2
            #   Wjj ← Tjj
            #
            info = hytpqxt3!(Ajj, Aj2, Wjj, Z, uplo, updn)

            if info == 0
                #
                # update the remaining rows
                #
                #   [ Anj An2 ] ← [ Anj An2 ] [ Qjj Qj2 ]
                #                             [ Q2j Q22 ]
                #
                hytprfb!(Anj, An2, Aj2, Wjj, Z, uplo, updn)
                jstrt = jstop + 1
            else
                info += jstrt - 1
            end
        end
    end

    return info
end

#
# Given a hyperbolic matrix
#
#   Q = [ Q₁₁ Q₁₂ ]
#       [ Q₂₁ Q₂₂ ]
#
# update A₂₁ and A₂₂
#
#   [ A₂₁ A₂₂ ] ← [ A₂₁ A₂₂ ] [ Q₂₁ Q₂₂ ] = [ L₂₁ L₂₂ ]
#                             [ Q₂₁ Q₂₂ ]
#
function hytpmqxt!(A12::AbstractMatrix, W::AbstractMatrix, A21::AbstractMatrix, A22::AbstractMatrix, Z::AbstractVector, uplo::Val{UPLO}, updn::Val{UPDN}) where {UPLO, UPDN}
    if UPLO === :L
        n2 = size(A12, 2)
        m1 = size(A12, 1)
        m2 = size(A21, 1)
    else
        n2 = size(A12, 1)
        m1 = size(A12, 2)
        m2 = size(A21, 2)
    end

    if 0 < n2
        wsize = size(W, 1)
        jstrt = 1

        @inbounds while jstrt ≤ m1
            jstop = min(jstrt + wsize - 1, m1)
            jsize = jstop - jstrt + 1
            #
            #   A = [ ⋯ A2j ⋯ ]   [  ⋯  ]
            #
            if UPLO === :L
                Aj2 = view(A12, jstrt:jstop,         1:n2)
                A2j = view(A21,         1:m2, jstrt:jstop)
            else
                Aj2 = view(A12,        1:n2,  jstrt:jstop)
                A2j = view(A21, jstrt:jstop,         1:m2)
            end

            Wjj = view(W, 1:jsize, jstrt:jstop)
            #
            #   [ A2j A22 ] ← [ A2j A22 ] [ Q2j Q22 ]
            #                             [ Q2j Q22 ]
            #
            hytprfb!(A2j, A22, Aj2, Wjj, Z, uplo, updn)
            jstrt = jstop + 1
        end
    end

    return
end
