const STPQXT_NB = 64

function stpqxt!(s::AbstractSemiring, A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}; nt::Integer = nthreads()) where {T}
    @assert size(A, 1) == size(A, 2) == size(X, 1) == size(Y, 2)
    @assert size(X, 2) == size(Y, 1)

    n = size(A, 1)
    r = size(X, 2)
    nb = min(STPQXT_NB, n)

    Yc = FMatrix{T}(undef, n, r)
    β = FVector{T}(undef, r)
    Γ = FMatrix{T}(undef, nb, r)
    Δ = FMatrix{T}(undef, nb, r)
    D = FMatrix{T}(undef, nb, nb)

    @inbounds for c in 1:r
        for i in 1:n
            Yc[i, c] = Y[c, i]
        end
    end

    fill!(β, sone(s, T, Val(:N)))

    for k in 1:nb:n
        b = min(nb, n - k + 1)
        kk = k:k + b - 1
        Γk = view(Γ, 1:b, :)
        Δk = view(Δ, 1:b, :)
        Dk = view(D, 1:b, 1:b)
        Xk = view(X, kk, :)
        Σk = view(Yc, kk, :)
        Akk = view(A, kk, kk)
        copyto!(Dk, Akk)
        stpqxt2!(s, Dk, Xk, Σk, β, Γk, Δk)
        copyto!(Akk, Dk)

        if k + b <= n
            nn = k + b:n
            m = length(nn)
            L21 = view(A, nn, kk); X2 = view(X, nn, :)
            U12 = view(A, kk, nn); Y2 = view(Yc, nn, :)

            if nt <= 1
                stpmqxt!(s, Val(:L), Xk, Γk, L21, X2)
                stpmqxt!(s, Val(:R), Δk, Σk, U12, Y2)
            else
                W = vecwidth(T)
                ch = W * cld(clamp(cld(m, nt), 64, 512), W)

                @sync for c in 1:ch:m
                    R = c:min(c + ch - 1, m)
                    L21R = view(L21, R, :)
                    U12R = view(U12, :, R)
                    X2R = view(X2, R, :)
                    Y2R = view(Y2, R, :)
                    @spawn stpmqxt!(s, Val(:L), Xk, Γk, L21R, X2R)
                    @spawn stpmqxt!(s, Val(:R), Δk, Σk, U12R, Y2R)
                end
            end
        end
    end

    return A
end

#
# given a factorization
#
#   D* = U* L*
#
# vectors x, y, and a scalar β = q*, computes the factorizations
#
#   [ q  yᵀ ]* = [ q yᵀ ]* [         ]*
#   [ x  D  ]    [   U' ]  [ x β  L' ]
#
#   [ D  x ]*   [ U x' ]* [ L     ]*
#   [ yᵀ q ]    [   q' ]  [ y'ᵀ   ]
#
# and the operators G and H such that, for all rows [ B b ] and columns [ C ],
#                                                                       [ c ]
#
#   [ ℓ b ] G = [ ℓ' b' ]      where      [ B b ] [ U x' ]* = [ ℓ  b' β' ]
#                                                 [   q' ]
#
#                                         [ b B ] [ q yᵀ ]* = [ b β  ℓ'  ]
#                                                 [   U' ]
#
#   H [ u ] = [ u' ]           where      [ L       ]* [ C ] = [ u  ]
#     [ c ]   [ c' ]                      [ y'ᵀ     ]  [ c ]   [ c' ]
#
#                                         [         ]* [ c ] = [ c  ]
#                                         [ x β  L' ]  [ C ]   [ u' ]
#
function stpqxt2!(s::AbstractSemiring, A::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}, β::AbstractVector{T},
                  Γ::AbstractMatrix{T}, Δ::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    z = szero(s, T, Val(:N))

    @inbounds for c in axes(X, 2)
        βc = β[c]

        for k in 1:n
            #
            # given
            #             k                  c                  c
            #         [   ⋮    ]         [   ⋮   ]          [   ⋮   ]
            #   A = k [ ⋯ p uᵀ ]   X = k [ ⋯ a ⋯ ]   Yc = k [ ⋯ b ⋯ ] ,
            #         [ ⋯ ℓ ⋱  ]         [ ⋯ x ⋯ ]          [ ⋯ y ⋯ ]
            #
            # let
            #
            #   Nk = [ p a ]      β = q*
            #        [ b q ]
            #
            ak = X[k, c]
            bk = Y[k, c]

            if (ak == z && bk == z) || βc == z
                Γ[k, c] = Δ[k, c] = Y[k, c] = z
            else
                #
                # compute the LU and UL factorizations
                #
                #   Nₖ* = U₁* L₁* = [ p a  ]* [     ]*
                #                   [   q' ]  [ σ   ]
                #
                #       = L₂* U₂* = [ p'   ]* [   δ ]*
                #                   [ b  q ]  [     ]
                #
                # where β' = q'* and γ = (L₂*)₂₁
                #
                p2, γ, δ, σ, βc = srotmg(s, A[k, k], βc, ak, bk)

                if k < n
                    Ank = view(A, k + 1:n, k)
                    Xnc = view(X, k + 1:n, c)
                    Akn = view(A, k, k + 1:n)
                    Ync = view(Y, k + 1:n, c)
                    #
                    #   [ Ank Xnc ] ← [ Ank Xnc ] [ 1 + ak γ  ak ]
                    #                             [ γ         1  ]
                    #
                    srotm!(s, Val(:L), ak, γ, Ank, Xnc)
                    #
                    #   [ Akn ] ← [ 1 δ ] [ Akn ]
                    #   [ Ync ]   [ σ 1 ] [ Ync ]
                    #
                    srotm!(s, Val(:R), δ, σ, Akn, Ync)
                end

                A[k, k] = p2
                Γ[k, c] = γ
                Δ[k, c] = δ
                Y[k, c] = σ
            end
        end

        β[c] = βc
    end

    return A
end
