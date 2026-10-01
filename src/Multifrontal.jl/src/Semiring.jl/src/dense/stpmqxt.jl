const STPQXT_TB = 256
const STPQXT_TK = 4
const STPQXT_TC = 4

#
# given the operator G computed by stpqxt2! and stored in Xk and Γ, computes
#
#   [ L X ] ← [ L X ] G
#
function stpmqxt!(s::AbstractSemiring, ::Val{:L}, Xk::AbstractMatrix{T}, Γ::AbstractMatrix{T},
                  L::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T}
    m = size(L, 1)
    b = size(L, 2)
    r = size(X, 2)

    @inbounds for i in 1:STPQXT_TB:m
        stpmqxt_tile!(s, Xk, Γ, L, X, i:min(i + STPQXT_TB - 1, m), b, r)
    end

    return
end

#
# given the operator H computed by stpqxt2! and stored in Δ and Σ, computes
#
#   [ U  ]     [ U  ]
#   [ Yᵀ ] ← H [ Yᵀ ]
#
function stpmqxt!(s::AbstractSemiring, ::Val{:R}, Δ::AbstractMatrix{T}, Σ::AbstractMatrix{T},
                  U::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    b = size(U, 1)
    m = size(U, 2)
    r = size(Y, 2)
    W = vecwidth(T)
    Z = sizeof(T)
    j = 1

    if stride(Δ, 1) == stride(Σ, 1) == 1
        @preserve U Y Δ Σ begin
            ldU = stride(U, 2); ldY = stride(Y, 2); ldΔ = stride(Δ, 2); ldΣ = stride(Σ, 2)

            @inbounds while j + W - 1 <= m
                pU = pointer(U) + (j - 1) * ldU * Z
                c = 1

                while c + STPQXT_TC - 1 <= r
                    srotm_gather_kern!(s, pU, ldU, pointer(Y) + ((c - 1) * ldY + j - 1) * Z, ldY,
                                       pointer(Δ) + (c - 1) * ldΔ * Z, ldΔ, pointer(Σ) + (c - 1) * ldΣ * Z, ldΣ, b, Val(STPQXT_TC))
                    c += STPQXT_TC
                end

                while c + 1 <= r
                    srotm_gather_kern!(s, pU, ldU, pointer(Y) + ((c - 1) * ldY + j - 1) * Z, ldY,
                                       pointer(Δ) + (c - 1) * ldΔ * Z, ldΔ, pointer(Σ) + (c - 1) * ldΣ * Z, ldΣ, b, Val(2))
                    c += 2
                end

                while c <= r
                    srotm_gather_kern!(s, pU, ldU, pointer(Y) + ((c - 1) * ldY + j - 1) * Z, ldY,
                                       pointer(Δ) + (c - 1) * ldΔ * Z, ldΔ, pointer(Σ) + (c - 1) * ldΣ * Z, ldΣ, b, Val(1))
                    c += 1
                end

                j += W
            end
        end
    end

    @inbounds while j + 3 <= m
        c = 1

        while c + 1 <= r
            Y[j, c], Y[j + 1, c], Y[j + 2, c], Y[j + 3, c], Y[j, c + 1], Y[j + 1, c + 1], Y[j + 2, c + 1], Y[j + 3, c + 1] =
                srotm42!(s, U, Δ, Σ, c, j, Y[j, c], Y[j + 1, c], Y[j + 2, c], Y[j + 3, c], Y[j, c + 1], Y[j + 1, c + 1], Y[j + 2, c + 1], Y[j + 3, c + 1])
            c += 2
        end

        if c <= r
            Y[j, c], Y[j + 1, c], Y[j + 2, c], Y[j + 3, c] =
                srotm4!(s, U, Δ, Σ, c, j, Y[j, c], Y[j + 1, c], Y[j + 2, c], Y[j + 3, c])
        end

        j += 4
    end

    @inbounds while j <= m
        for c in axes(Y, 2)
            bⱼ = Y[j, c]

            for k in 1:b
                uₖ = U[k, j]
                U[k, j] = smuladd(s, Δ[k, c], bⱼ, uₖ, Val(:N), Val(:N))
                bⱼ = smuladd(s, Σ[k, c], uₖ, bⱼ, Val(:N), Val(:N))
            end

            Y[j, c] = bⱼ
        end

        j += 1
    end

    return
end

@generated function srotm_tile_kern!(s::AbstractSemiring, pL::Ptr{T}, sL::Integer, pX::Ptr{T}, sX::Integer,
                                     a::NTuple{KC, T}, g::NTuple{KC, T}, n::Integer, ::Val{K}, ::Val{C}) where {T, KC, K, C}
    @assert KC == K * C
    W = vecwidth(T)
    Z = sizeof(T)

    l(k) = Symbol(:l_, k)
    x(c) = Symbol(:x_, c)

    function pass(vec)
        ex = Expr(:block)

        for k in 1:K
            ptr = :(pL + $(k - 1) * sL * $Z + o)
            push!(ex.args, vec ? :($(l(k)) = vload(Vec{$W, $T}, $ptr)) : :($(l(k)) = unsafe_load($ptr)))
        end

        for c in 1:C
            ptr = :(pX + $(c - 1) * sX * $Z + o)
            push!(ex.args, vec ? :($(x(c)) = vload(Vec{$W, $T}, $ptr)) : :($(x(c)) = unsafe_load($ptr)))
        end

        for c in 1:C, k in 1:K
            q = (c - 1) * K + k

            push!(ex.args, :($(x(c)) = smuladd(s, $(l(k)), a[$q], $(x(c)), Val(:N), Val(:N))))
            push!(ex.args, :($(l(k)) = smuladd(s, $(x(c)), g[$q], $(l(k)), Val(:N), Val(:N))))
        end

        for k in 1:K
            ptr = :(pL + $(k - 1) * sL * $Z + o)
            push!(ex.args, vec ? :(vstore($(l(k)), $ptr)) : :(unsafe_store!($ptr, $(l(k)))))
        end

        for c in 1:C
            ptr = :(pX + $(c - 1) * sX * $Z + o)
            push!(ex.args, vec ? :(vstore($(x(c)), $ptr)) : :(unsafe_store!($ptr, $(x(c)))))
        end

        return ex
    end

    return quote
        i = 1

        while i + $(W - 1) <= n
            o = (i - 1) * $Z
            $(pass(true))
            i += $W
        end

        while i <= n
            o = (i - 1) * $Z
            $(pass(false))
            i += 1
        end

        return
    end
end

@inline function srotm_tile!(s::AbstractSemiring, Xk::AbstractMatrix{T}, Γ::AbstractMatrix{T}, L::AbstractMatrix{T}, X::AbstractMatrix{T},
                             rows::UnitRange{Int}, k₀::Int, c₀::Int, ::Val{K}, ::Val{C}) where {T, K, C}
    a = ntuple(q -> @inbounds(Xk[k₀ + (q - 1) % K, c₀ + (q - 1) ÷ K]), Val(K * C))
    g = ntuple(q -> @inbounds(Γ[k₀ + (q - 1) % K, c₀ + (q - 1) ÷ K]), Val(K * C))
    Z = sizeof(T)

    @preserve L X begin
        pL = pointer(L) + ((k₀ - 1) * stride(L, 2) + first(rows) - 1) * Z
        pX = pointer(X) + ((c₀ - 1) * stride(X, 2) + first(rows) - 1) * Z
        srotm_tile_kern!(s, pL, stride(L, 2), pX, stride(X, 2), a, g, length(rows), Val(K), Val(C))
    end

    return
end

@inline function stpmqxt_cols!(s::AbstractSemiring, Xk, Γ, L, X, rows, b, c, ::Val{C}) where {C}
    k = 1

    @inbounds while k + STPQXT_TK - 1 <= b
        srotm_tile!(s, Xk, Γ, L, X, rows, k, c, Val(STPQXT_TK), Val(C))
        k += STPQXT_TK
    end

    @inbounds while k <= b
        srotm_tile!(s, Xk, Γ, L, X, rows, k, c, Val(1), Val(C))
        k += 1
    end

    return
end

@inline function stpmqxt_tile!(s::AbstractSemiring, Xk, Γ, L, X, rows, b, r)
    c = 1

    @inbounds while c + STPQXT_TC - 1 <= r
        stpmqxt_cols!(s, Xk, Γ, L, X, rows, b, c, Val(STPQXT_TC))
        c += STPQXT_TC
    end

    @inbounds while c + 1 <= r
        stpmqxt_cols!(s, Xk, Γ, L, X, rows, b, c, Val(2))
        c += 2
    end

    @inbounds while c <= r
        stpmqxt_cols!(s, Xk, Γ, L, X, rows, b, c, Val(1))
        c += 1
    end

    return
end

@inline function srotm42!(s::AbstractSemiring, U::AbstractMatrix{T}, Δ::AbstractMatrix{T}, Σ::AbstractMatrix{T}, c::Integer, j::Integer,
                          b₁::T, b₂::T, b₃::T, b₄::T, e₁::T, e₂::T, e₃::T, e₄::T) where {T}
    @inbounds for k in axes(U, 1)
        δ = Δ[k, c];     σ = Σ[k, c]
        δ2 = Δ[k, c + 1]; σ2 = Σ[k, c + 1]
        u₁ = U[k, j]; u₂ = U[k, j + 1]; u₃ = U[k, j + 2]; u₄ = U[k, j + 3]
        #
        # pair c
        #
        v₁ = smuladd(s, δ, b₁, u₁, Val(:N), Val(:N)); b₁ = smuladd(s, σ, u₁, b₁, Val(:N), Val(:N))
        v₂ = smuladd(s, δ, b₂, u₂, Val(:N), Val(:N)); b₂ = smuladd(s, σ, u₂, b₂, Val(:N), Val(:N))
        v₃ = smuladd(s, δ, b₃, u₃, Val(:N), Val(:N)); b₃ = smuladd(s, σ, u₃, b₃, Val(:N), Val(:N))
        v₄ = smuladd(s, δ, b₄, u₄, Val(:N), Val(:N)); b₄ = smuladd(s, σ, u₄, b₄, Val(:N), Val(:N))
        #
        # pair c + 1
        #
        U[k, j]     = smuladd(s, δ2, e₁, v₁, Val(:N), Val(:N)); e₁ = smuladd(s, σ2, v₁, e₁, Val(:N), Val(:N))
        U[k, j + 1] = smuladd(s, δ2, e₂, v₂, Val(:N), Val(:N)); e₂ = smuladd(s, σ2, v₂, e₂, Val(:N), Val(:N))
        U[k, j + 2] = smuladd(s, δ2, e₃, v₃, Val(:N), Val(:N)); e₃ = smuladd(s, σ2, v₃, e₃, Val(:N), Val(:N))
        U[k, j + 3] = smuladd(s, δ2, e₄, v₄, Val(:N), Val(:N)); e₄ = smuladd(s, σ2, v₄, e₄, Val(:N), Val(:N))
    end

    return b₁, b₂, b₃, b₄, e₁, e₂, e₃, e₄
end

@inline function srotm4!(s::AbstractSemiring, U::AbstractMatrix{T}, Δ::AbstractMatrix{T}, Σ::AbstractMatrix{T},
                         c::Integer, j::Integer, b₁::T, b₂::T, b₃::T, b₄::T) where {T}
    @inbounds for k in axes(U, 1)
        δ = Δ[k, c]
        σ = Σ[k, c]
        u₁ = U[k, j]; u₂ = U[k, j + 1]; u₃ = U[k, j + 2]; u₄ = U[k, j + 3]
        U[k, j]     = smuladd(s, δ, b₁, u₁, Val(:N), Val(:N))
        U[k, j + 1] = smuladd(s, δ, b₂, u₂, Val(:N), Val(:N))
        U[k, j + 2] = smuladd(s, δ, b₃, u₃, Val(:N), Val(:N))
        U[k, j + 3] = smuladd(s, δ, b₄, u₄, Val(:N), Val(:N))
        b₁ = smuladd(s, σ, u₁, b₁, Val(:N), Val(:N))
        b₂ = smuladd(s, σ, u₂, b₂, Val(:N), Val(:N))
        b₃ = smuladd(s, σ, u₃, b₃, Val(:N), Val(:N))
        b₄ = smuladd(s, σ, u₄, b₄, Val(:N), Val(:N))
    end

    return b₁, b₂, b₃, b₄
end

@generated function srotm_gather_kern!(s::AbstractSemiring, pU::Ptr{T}, ldU::Integer, pY::Ptr{T}, ldY::Integer,
                                       pΔ::Ptr{T}, ldΔ::Integer, pΣ::Ptr{T}, ldΣ::Integer, nk::Integer, ::Val{C}) where {T, C}
    W = vecwidth(T)
    Z = sizeof(T)

    y(c) = Symbol(:y_, c)

    init = Expr(:block)
    term = Expr(:block)

    for c in 1:C
        push!(init.args, :($(y(c)) = vload(Vec{$W, $T}, pY + $(c - 1) * ldY * $Z)))
        push!(term.args, :(vstore($(y(c)), pY + $(c - 1) * ldY * $Z)))
    end

    body = Expr(:block)
    loads = [:(unsafe_load(pU + ($(t - 1) * ldU + k - 1) * $Z)) for t in 1:W]
    push!(body.args, :(u = Vec{$W, $T}(($(loads...),))))

    for c in 1:C
        push!(body.args, :(δ = unsafe_load(pΔ + ($(c - 1) * ldΔ + k - 1) * $Z)))
        push!(body.args, :(σ = unsafe_load(pΣ + ($(c - 1) * ldΣ + k - 1) * $Z)))
        push!(body.args, :(t = smuladd(s, δ, $(y(c)), u, Val(:N), Val(:N))))           # u + δ y
        push!(body.args, :($(y(c)) = smuladd(s, σ, u, $(y(c)), Val(:N), Val(:N))))     # y + σ u
        push!(body.args, :(u = t))
    end

    for t in 1:W
        push!(body.args, :(unsafe_store!(pU + ($(t - 1) * ldU + k - 1) * $Z, u[$t])))
    end

    return quote
        $init

        for k in 1:nk
            $body
        end

        $term
        return
    end
end
