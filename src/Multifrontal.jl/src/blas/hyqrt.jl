# ===== hyperbolic triangular-pentagonal kernels (recursive blocked) =====
#
# Blocked, downdate-only (signature σ = -1) analogues of the tpqrt / tprfb
# pair, computing the Cholesky downdate Aᴴ A - Bᴴ B = Rᴴ R (:U) or
# A Aᴴ - B Bᴴ = R Rᴴ (:L) with J-orthogonal transformations. The compact-WY
# T factor is assembled by an Elmroth-Gustavson recursion; the base case runs
# columnwise J-Householders. Generic element type via the repo BLAS wrappers.
#
const HYQRT_BASE = 8

# ===== :U =====
#
# [Ablk; Bblk] ← (I - V Tᴴ Vᴴ J) [Ablk; Bblk]     (sign-flipped tprfb, l = 0)
#
function hyqrt_apply!(Ablk::AbstractMatrix{T}, Bblk::AbstractMatrix{T}, V::AbstractMatrix{T}, Tv::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T}
    w = size(V, 2)
    p = size(Ablk, 2)
    iszero(p) && return
    Zv = view(Z, oneto(w), oneto(p))

    copyto!(Zv, Ablk)
    gemm!(Val(:C), Val(:N), -one(T), V, Bblk, one(T), Zv)
    trmm!(Val(:L), Val(:U), Val(:C), Val(:N), one(T), Tv, Zv)
    axpy!(-one(T), Zv, Ablk)
    gemm!(Val(:N), Val(:N), -one(T), V, Zv, one(T), Bblk)

    return
end

# factor panel-local columns lo:hi of the pentagon [A; Bp]; fill Tw[lo:hi, lo:hi]
function hyqrt_fact!(A::AbstractMatrix{T}, Bp::AbstractMatrix{T}, Tw::AbstractMatrix{T}, Z::AbstractMatrix{T}, j0::Int, lo::Int, hi::Int) where {T}
    bn = size(Bp, 1)
    w = hi - lo + 1

    if w <= HYQRT_BASE
        @inbounds for jl in lo:hi
            jg = j0 + jl - 1
            #
            # J-Householder annihilating Bp[:, jl] into the pivot A[jg, jg]
            #
            ρ = real(A[jg, jg])
            s2 = zero(real(T))

            for r in 1:bn
                s2 += abs2(Bp[r, jl])
            end

            if iszero(s2)
                for i in lo:jl
                    Tw[i, jl] = zero(T)
                end

                continue
            end

            β2 = ρ * ρ - s2
            ispositive(β2) || return jg
            β = copysign(sqrt(β2), ρ)
            δ = s2 / (ρ + β)
            τ = -δ / β
            A[jg, jg] = β
            invδ = inv(δ)

            for r in 1:bn
                Bp[r, jl] *= invδ
            end
            #
            # apply H to the remaining panel columns jl+1:hi
            #
            for ul in jl + 1:hi
                ug = j0 + ul - 1
                z = A[jg, ug]

                for r in 1:bn
                    z = muladd(-conj(Bp[r, jl]), Bp[r, ul], z)
                end

                z *= τ
                A[jg, ug] -= z

                for r in 1:bn
                    Bp[r, ul] = muladd(-z, Bp[r, jl], Bp[r, ul])
                end
            end
            #
            # T column: g = τ T (Vmᴴ v)
            #
            if jl > lo
                g = view(Tw, lo:jl - 1, jl)
                gemv!(Val(:C), convert(T, τ), view(Bp, :, lo:jl - 1), view(Bp, :, jl), zero(T), g)
                trmv!(Val(:U), Val(:N), Val(:N), view(Tw, lo:jl - 1, lo:jl - 1), g)
            end

            Tw[jl, jl] = convert(T, τ)
        end

        return 0
    end

    mid = lo + (w >> 1) - 1

    info = hyqrt_fact!(A, Bp, Tw, Z, j0, lo, mid)
    ispositive(info) && return info

    hyqrt_apply!(
        view(A, j0 + lo - 1:j0 + mid - 1, j0 + mid:j0 + hi - 1),
        view(Bp, :, mid + 1:hi),
        view(Bp, :, lo:mid),
        view(Tw, lo:mid, lo:mid),
        Z,
    )

    info = hyqrt_fact!(A, Bp, Tw, Z, j0, mid + 1, hi)
    ispositive(info) && return info
    #
    # merge: Tw[lo:mid, mid+1:hi] = T₁ (Vm₁ᴴ Vm₂) T₂
    #
    T12 = view(Tw, lo:mid, mid + 1:hi)
    gemm!(Val(:C), Val(:N), one(T), view(Bp, :, lo:mid), view(Bp, :, mid + 1:hi), zero(T), T12)
    trmm!(Val(:L), Val(:U), Val(:N), Val(:N), one(T), view(Tw, lo:mid, lo:mid), T12)
    trmm!(Val(:R), Val(:U), Val(:N), Val(:N), one(T), view(Tw, mid + 1:hi, mid + 1:hi), T12)

    return 0
end

# hyqrt!(A, B, C, D, W, Z) -> info
#     A nn×nn upper, B bn×nn, C nn×na, D bn×na. In place computes R (over A),
#     C′ (over C) and a D-carrier with RᴴR = AᴴA - BᴴB, RᴴC′ = AᴴC - BᴴD.
#     W (nb × nb) is the panel T scratch, Z (nb × max(nn, na)) the apply staging.
#
function hyqrt!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}, W::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    bn = size(B, 1)
    na = size(C, 2)
    iszero(bn) && return 0
    nb = size(W, 1)
    j0 = 1

    while j0 <= n
        j1 = min(j0 + nb - 1, n)
        w = j1 - j0 + 1
        Bp = view(B, :, j0:j1)

        info = hyqrt_fact!(A, Bp, W, Z, j0, 1, w)
        ispositive(info) && return info

        Tv = view(W, oneto(w), oneto(w))

        if j1 < n
            hyqrt_apply!(view(A, j0:j1, j1 + 1:n), view(B, :, j1 + 1:n), Bp, Tv, Z)
        end

        if ispositive(na)
            hyqrt_apply!(view(C, j0:j1, :), D, Bp, Tv, Z)
        end

        j0 = j1 + 1
    end

    return 0
end

# ===== :L =====
#
# [Ablk Bblk] ← [Ablk Bblk] (I - J V Tᴴ Vᴴ)     (sign-flipped tprfb mirror)
#
function hylqt_apply!(Ablk::AbstractMatrix{T}, Bblk::AbstractMatrix{T}, V::AbstractMatrix{T}, Tv::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T}
    w = size(V, 1)
    p = size(Ablk, 1)
    iszero(p) && return
    Zv = view(Z, oneto(p), oneto(w))

    copyto!(Zv, Ablk)
    gemm!(Val(:N), Val(:C), -one(T), Bblk, V, one(T), Zv)
    trmm!(Val(:R), Val(:U), Val(:N), Val(:N), one(T), Tv, Zv)
    axpy!(-one(T), Zv, Ablk)
    gemm!(Val(:N), Val(:N), -one(T), Zv, V, one(T), Bblk)

    return
end

function hylqt_fact!(A::AbstractMatrix{T}, Bp::AbstractMatrix{T}, Tw::AbstractMatrix{T}, Z::AbstractMatrix{T}, j0::Int, lo::Int, hi::Int) where {T}
    bn = size(Bp, 2)
    w = hi - lo + 1

    if w <= HYQRT_BASE
        @inbounds for jl in lo:hi
            jg = j0 + jl - 1
            #
            # J-Householder annihilating the row Bp[jl, :] into A[jg, jg]
            #
            ρ = real(A[jg, jg])
            s2 = zero(real(T))

            for c in 1:bn
                s2 += abs2(Bp[jl, c])
            end

            if iszero(s2)
                for i in lo:jl
                    Tw[i, jl] = zero(T)
                end

                continue
            end

            β2 = ρ * ρ - s2
            ispositive(β2) || return jg
            β = copysign(sqrt(β2), ρ)
            δ = s2 / (ρ + β)
            τ = -δ / β
            A[jg, jg] = β
            invδ = inv(δ)

            for c in 1:bn
                Bp[jl, c] *= invδ
            end
            #
            # apply H to the remaining panel rows jl+1:hi
            #
            for ul in jl + 1:hi
                ug = j0 + ul - 1
                z = A[ug, jg]

                for c in 1:bn
                    z = muladd(-conj(Bp[jl, c]), Bp[ul, c], z)
                end

                z *= τ
                A[ug, jg] -= z

                for c in 1:bn
                    Bp[ul, c] = muladd(-z, Bp[jl, c], Bp[ul, c])
                end
            end
            #
            # T column: g = τ (Vmᴴ v)
            #
            if jl > lo
                g = view(Tw, lo:jl - 1, jl)

                for i in lo:jl - 1
                    acc = zero(T)

                    for c in 1:bn
                        acc = muladd(Bp[i, c], conj(Bp[jl, c]), acc)
                    end

                    g[i - lo + 1] = convert(T, τ) * acc
                end

                trmv!(Val(:U), Val(:N), Val(:N), view(Tw, lo:jl - 1, lo:jl - 1), g)
            end

            Tw[jl, jl] = convert(T, τ)
        end

        return 0
    end

    mid = lo + (w >> 1) - 1

    info = hylqt_fact!(A, Bp, Tw, Z, j0, lo, mid)
    ispositive(info) && return info

    hylqt_apply!(
        view(A, j0 + mid:j0 + hi - 1, j0 + lo - 1:j0 + mid - 1),
        view(Bp, mid + 1:hi, :),
        view(Bp, lo:mid, :),
        view(Tw, lo:mid, lo:mid),
        Z,
    )

    info = hylqt_fact!(A, Bp, Tw, Z, j0, mid + 1, hi)
    ispositive(info) && return info
    #
    # merge: Tw[lo:mid, mid+1:hi] = T₁ (Vm₁ Vm₂ᴴ) T₂
    # (row-stored reflectors: the gemm conjugates the later block, matching the
    # T-column convention)
    #
    T12 = view(Tw, lo:mid, mid + 1:hi)
    gemm!(Val(:N), Val(:C), one(T), view(Bp, lo:mid, :), view(Bp, mid + 1:hi, :), zero(T), T12)
    trmm!(Val(:L), Val(:U), Val(:N), Val(:N), one(T), view(Tw, lo:mid, lo:mid), T12)
    trmm!(Val(:R), Val(:U), Val(:N), Val(:N), one(T), view(Tw, mid + 1:hi, mid + 1:hi), T12)

    return 0
end

# hylqt!(A, B, C, D, W, Z) -> info
#     A nn×nn lower, B nn×bn, C na×nn, D na×bn. In place computes R (over A),
#     C′ (over C) and a D-carrier with RRᴴ = AAᴴ - BBᴴ, C′Rᴴ = CAᴴ - DBᴴ.
#
function hylqt!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, C::AbstractMatrix{T}, D::AbstractMatrix{T}, W::AbstractMatrix{T}, Z::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    bn = size(B, 2)
    na = size(C, 1)
    iszero(bn) && return 0
    nb = size(W, 1)
    j0 = 1

    while j0 <= n
        j1 = min(j0 + nb - 1, n)
        w = j1 - j0 + 1
        Bp = view(B, j0:j1, :)

        info = hylqt_fact!(A, Bp, W, Z, j0, 1, w)
        ispositive(info) && return info

        Tv = view(W, oneto(w), oneto(w))

        if j1 < n
            hylqt_apply!(view(A, j1 + 1:n, j0:j1), view(B, j1 + 1:n, :), Bp, Tv, Z)
        end

        if ispositive(na)
            hylqt_apply!(view(C, :, j0:j1), D, Bp, Tv, Z)
        end

        j0 = j1 + 1
    end

    return 0
end
