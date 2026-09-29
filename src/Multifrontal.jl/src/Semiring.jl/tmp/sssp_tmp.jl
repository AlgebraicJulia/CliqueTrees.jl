#=
================================================================================
 Elementary-vector (sssp / stsp) solves for ChordalSLU  — QUARANTINED

   src/Multifrontal.jl/src/Semiring.jl/tmp/sssp_tmp.jl

 Solve against

     eₖ = [ 0 ⋯ 0 1 0 ⋯ 0 ]ᵀ,

 where 0 = szero(s, T, trans) and 1 = sone(s, T, trans). Note that the
 "zero" depends on the mode: e.g. for MinPlus it is +Inf in modes :N/:T
 and -Inf in the residuated modes :R/:C.

 This is the elementary code from ref/elementary_sgetrs.jl, quarantined: every
 function is _tmp-suffixed, overrides NO library method, and is not integrated
 into sgetrs!/ldiv!. It CAN be included (into the SR module) and called from a
 benchmark.

 The one structural change from the reference: the backward pass no longer uses
 strsx_subtree! (which had its own @spawn recursion). Instead it routes through
 strsx_solve_tmp! from subtree_tmp.jl — the same subtree-parallel machinery used
 by the dense solve — over the subtree fdesc(r):r. A prebuilt DivisionSchedule
 (built over the whole tree 1:N) is reused when the reached root spans it; other
 roots and other components fall back to a per-solve partition.

 DEPENDS ON subtree_tmp.jl (strsx_solve_tmp!, sgetrs_tmp!, DivisionSchedule);
 include this file AFTER it:

     using CliqueTrees
     SR = CliqueTrees.Multifrontal.Semiring
     Base.include(SR, "tmp/subtree_tmp.jl")
     Base.include(SR, "tmp/sssp_tmp.jl")
================================================================================
=#


################################################################################
#   Forward path solve  (from chordal/strsx.jl)
################################################################################

# ===== strsx_path_tmp! =====
#
# Forward triangular solve against an elementary vector eₖ.
# On entry C = eₖ. On exit C[res(f)] holds the forward solution for
# every front f on the path
#
#   idx(k) → pnt(idx(k)) → ⋯ → r,
#
# and every other entry of C is untouched (hence still zero).
#
# The update vector m of each front, i.e. the contribution
#
#   m = L₂₁ C₁ (+ contributions passed up from below)
#
# to its separator, is kept in a small buffer rather than scattered into
# C. At the parent, it is split via the relative indices rel(f) into a
# part landing in res(parent), which is written into C, and a part
# landing in sep(parent), which becomes the start of the parent's update
# vector. This is the same scheme as lowrank_loop! in lowrank/dense.jl.
#
# Requires length(W.Mval) ≥ 2 nFval (two ping-pong buffers).
#
# Returns the root r.
#
function strsx_path_tmp!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        C::AbstractVector{T},
        W::DivisionWorkspace{T},
        k::I,
    ) where {SIDE, TRANS, UPLO, T, I}
    @assert isforward(UPLO, TRANS, SIDE)

    S = A.S
    nF = S.nFval
    @assert length(W.Mval) >= 2nF
    #
    # m and m′ are the update vectors of the current
    # front and of its parent
    #
    # (both views use UnitRange indices, so that swapping
    # them below is type-stable)
    #
    m = view(W.Mval, one(I):nF)
    m′ = view(W.Mval, nF + one(I):nF + nF)
    #
    # first front: trailing solve starting at k
    #
    f = S.idx[k]
    strsx_path_first_tmp!(s, C, m, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, k, f, trans, A.uplo, diag, side)
    #
    # remaining fronts: walk to the root
    #
    p = S.pnt[f]

    while ispositive(p)
        strsx_path_next_tmp!(s, C, m, m′, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, S.rel, f, p, trans, A.uplo, diag, side)
        m, m′ = m′, m
        f = p
        p = S.pnt[f]
    end

    return f
end

#
# The first front f = idx(k). Only the trailing rows j:nn of res(f),
# where j is the local index of k, can be nonzero:
#
#                j:nn
#     L = [ ⋯    ⋯    ] 1:j-1
#         [ ⋯   D₁₁ʲ  ] j:nn
#         [ ⋯   L₂₁ʲ  ] sep(f)
#
function strsx_path_first_tmp!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        m::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        k::I,
        f::I,
        trans::Val,
        uplo::Val{UPLO},
        diag::Val,
        side::Val,
    ) where {T, I, UPLO}
    #
    # nn is the size of the residual at node f
    #
    #     nn = | res(f) |
    #
    nn = eltypedegree(res, f)
    #
    # na is the size of the separator at node f
    #
    #     na = | sep(f) |
    #
    na = eltypedegree(sep, f)
    Rp = pointers(res)[f]
    Dp = Dptr[f]
    Lp = Lptr[f]
    #
    # j is the local index of k in res(f)
    #
    j = k - Rp + one(I)

    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
        L₂₁ʲ = view(L₂₁, oneto(na), j:nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
        L₂₁ʲ = view(L₂₁, j:nn, oneto(na))
    end

    D₁₁ʲ = view(D₁₁, j:nn, j:nn)
    C₁ʲ = view(C, k:Rp + nn - one(I))
    #
    #   M₂ ← 0
    #
    M₂ = view(m, oneto(na))
    szerorec!(s, M₂, trans)
    #
    #   C₁ʲ ← (D₁₁ʲ)* C₁ʲ
    #   M₂  ← M₂ + L₂₁ʲ C₁ʲ
    #
    strsx_path_kern_tmp!(s, C₁ʲ, M₂, D₁₁ʲ, L₂₁ʲ, trans, uplo, diag, side)
    return
end

#
# A front p on the path, with child f on the path. Every entry of C[res(p)]
# is zero on entry, so the only input is the update vector m of f.
#
function strsx_path_next_tmp!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        m::AbstractVector{T},
        m′::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        rel::AbstractGraph{I},
        f::I,
        p::I,
        trans::Val,
        uplo::Val{UPLO},
        diag::Val,
        side::Val,
    ) where {T, I, UPLO}
    nn = eltypedegree(res, p)
    na = eltypedegree(sep, p)
    Rp = pointers(res)[p]
    Dp = Dptr[p]
    Lp = Lptr[p]

    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end
    #
    #   C = [ C₁ ] res(p)
    #       [ M₂ ] sep(p)
    #
    C₁ = view(C, Rp:Rp + nn - one(I))
    M₂ = view(m′, oneto(na))
    szerorec!(s, M₂, trans)
    #
    # add the update vector of f into the front of p
    #
    #   [ C₁ ] ← [ C₁ ] + Rᶠ m
    #   [ M₂ ]   [ M₂ ]
    #
    frel = neighbors(rel, f)

    @inbounds for i in eachindex(frel)
        q = frel[i]

        if q <= nn
            C₁[q] = splus(s, C₁[q], m[i], trans)
        else
            M₂[q - nn] = splus(s, M₂[q - nn], m[i], trans)
        end
    end
    #
    #   C₁ ← D₁₁* C₁
    #   M₂ ← M₂ + L₂₁ C₁
    #
    strsx_path_kern_tmp!(s, C₁, M₂, D₁₁, L₂₁, trans, uplo, diag, side)
    return
end

#
#   C₁ ← D₁₁* C₁
#   M₂ ← M₂ + L₂₁ C₁
#
function strsx_path_kern_tmp!(
        s::AbstractSemiring,
        C₁::AbstractVector{T},
        M₂::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        trans::Val,
        uplo::Val{UPLO},
        diag::Val{DIAG},
        side::Val{SIDE},
    ) where {T, UPLO, DIAG, SIDE}
    if isone(length(C₁))
        #
        # scalar front (the common case in sparse problems)
        #
        v = C₁[1]

        if !isintegral(s) && DIAG === :N
            d = sstar(s, D₁₁[1, 1])

            if SIDE === :L
                v = sprod(s, d, v, trans, Val(:N))
            else
                v = sprod(s, v, d, Val(:N), trans)
            end

            C₁[1] = v
        end

        if UPLO === :L
            l₂₁ = view(L₂₁, :, 1)
        else
            l₂₁ = view(L₂₁, 1, :)
        end

        @inbounds for i in eachindex(M₂)
            if SIDE === :L
                M₂[i] = smuladd(s, l₂₁[i], v, M₂[i], trans, Val(:N))
            else
                M₂[i] = smuladd(s, v, l₂₁[i], M₂[i], Val(:N), trans)
            end
        end
    else
        strsx!(s, side, trans, uplo, diag, D₁₁, C₁)

        if !isempty(M₂)
            if SIDE === :L
                sgemx!(s, trans, Val(:N), M₂, L₂₁, C₁)
            else
                sgemx!(s, Val(:N), trans, M₂, C₁, L₂₁)
            end
        end
    end

    return
end

# ===== fdesc_tmp =====
#
# The first descendant of a front f. Fronts are postordered,
# so the subtree rooted at f is fdesc(f):f.
#
function fdesc_tmp(S::ChordalSymbolic{I}, f::I) where {I}
    chd = S.chd

    while ispositive(eltypedegree(chd, f))
        g = f

        for c in neighbors(chd, f)
            g = min(g, c)
        end

        f = g
    end

    return f
end


################################################################################
#   Elementary solve within one strongly connected component
#   (from chordal/sgetrs.jl)
################################################################################

# ===== sgetrs_elem_tmp! =====
#
# Solve against an elementary vector eₖ within one strongly connected
# component. On entry C = eₖ; W must satisfy length(W.Mval) ≥ 2 nFval.
#
#   1. forward pass on the path from idx(k) to its root r
#   2. backward pass on the subtree fdesc(r):r, routed through the
#      subtree-parallel strsx_solve_tmp!
#
function sgetrs_elem_tmp!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        C::AbstractVector{T},
        W::DivisionWorkspace{T},
        pool,
        k::I,
        nt::Integer,
        sched = nothing,
    ) where {T, I, SIDE, TRANS}
    S = L.S
    N = nv(S.res)

    if isforward(:L, TRANS, SIDE)
        r = strsx_path_tmp!(s, side, trans, Val(:U), L, C, W, k)
        t = fdesc_of(sched, S, r)
        strsx_solve_tmp!(s, side, trans, Val(:N), U, C, W, pool, nt, t, r, tpsched_range(sched, t, r, N))
    else
        r = strsx_path_tmp!(s, side, trans, Val(:N), U, C, W, k)
        t = fdesc_of(sched, S, r)
        strsx_solve_tmp!(s, side, trans, Val(:U), L, C, W, pool, nt, t, r, tpsched_range(sched, t, r, N))
    end

    return C
end

#
# first descendant of the reached root r: O(1) from the schedule's
# precomputed fd (whole tree), or the direct walk if no schedule
#
@inline fdesc_of(sched::DivisionSchedule, S, r) = @inbounds sched.fd[r]
@inline fdesc_of(::Nothing, S::ChordalSymbolic{I}, r::I) where {I} = fdesc_tmp(S, r)

#
# The prebuilt schedule is built over the whole tree 1:N; it applies only
# when the reached subtree fdesc(r):r is exactly that. Otherwise fall back
# to a per-solve partition (sched = nothing).
#
@inline function tpsched_range(sched, fstrt::I, fstop::I, N::I) where {I}
    return (!isnothing(sched) && fstrt == one(I) && fstop == N) ? sched : nothing
end


################################################################################
#   Elementary solve over all strongly connected components  (BTF)
#   (from blocked/sgetrs.jl)
################################################################################

# ===== sgetrs_elem_tmp! =====
#
# Solve against an elementary vector eₖ. On entry C = eₖ.
# Let c be the strongly connected component containing k.
#
#   - Components processed before c are zero and remain zero, so they
#     are skipped, along with the off-diagonal product feeding c.
#   - Component c is solved with the chordal sgetrs_elem_tmp!.
#   - Components processed after c are solved only if they hold a
#     nonzero entry once all of their inputs are known. An unreached
#     component is zero and remains zero.
#
function sgetrs_elem_tmp!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        Bptr::AbstractVector{I},
        Fptr::AbstractVector{I},
        nBptr::I,
        Nptr::AbstractVector{I},
        Ntgt::AbstractVector{I},
        Nval::AbstractVector{T},
        C::AbstractVector{T},
        W::DivisionWorkspace{T},
        pool,
        k::I,
        nt::Integer,
        sched = nothing,
    ) where {T, I, SIDE, TRANS}
    S = L.S
    N = nv(S.res)
    #
    # c is the component containing k
    #
    #   Bptr[c] ≤ k < Bptr[c + 1]
    #
    c = convert(I, searchsortedlast(view(Bptr, oneto(nBptr)), k))

    sgetrs_elem_tmp!(s, side, trans, L, U, C, W, pool, k, nt, sched)

    if isforward(:U, TRANS, SIDE)
        #
        # components c + 1, …, nBptr gather from earlier ones
        #
        for d in c + one(I):nBptr
            fstrt = Fptr[d]
            fstop = Fptr[d + one(I)] - one(I)

            jstrt = Bptr[d]
            jstop = Bptr[d + one(I)] - one(I)

            if SIDE === :L
                sgemx_sparse!(s, trans, Val(:N), C, Nptr, Ntgt, Nval, jstrt, jstop, C, 1)
            else
                sgemx_sparse!(s, Val(:N), trans, C, C, Nptr, Ntgt, Nval, jstrt, jstop, 1)
            end

            if !siszero_tmp(s, trans, C, jstrt, jstop)
                sgetrs_tmp!(s, side, trans, L, U, C, W, pool, nt, fstrt, fstop, tpsched_range(sched, fstrt, fstop, N))
            end
        end
    else
        #
        # component c scatters into earlier ones, then
        # components c - 1, …, 1 are solved and scatter in turn
        #
        for d in reverse(oneto(c))
            fstrt = Fptr[d]
            fstop = Fptr[d + one(I)] - one(I)

            jstrt = Bptr[d]
            jstop = Bptr[d + one(I)] - one(I)

            if d < c
                if siszero_tmp(s, trans, C, jstrt, jstop)
                    continue
                end

                sgetrs_tmp!(s, side, trans, L, U, C, W, pool, nt, fstrt, fstop, tpsched_range(sched, fstrt, fstop, N))
            end

            if SIDE === :L
                sgemx_sparse!(s, trans, Val(:N), C, Nptr, Ntgt, Nval, jstrt, jstop, C, 1)
            else
                sgemx_sparse!(s, Val(:N), trans, C, C, Nptr, Ntgt, Nval, jstrt, jstop, 1)
            end
        end
    end

    return C
end


################################################################################
#   Zero test  (from utils.jl)
################################################################################

#
# Is C[jstrt:jstop] identically zero? A false negative only costs time
# (the component is solved anyway), so this uses isequal rather than ==.
#
function siszero_tmp(s::AbstractSemiring, trans::Val, C::AbstractVector{T}, jstrt::I, jstop::I) where {T, I}
    z = szero(s, T, trans)

    @inbounds for j in jstrt:jstop
        if !isequal(C[j], z)
            return false
        end
    end

    return true
end


################################################################################
#   User entry points  (from chordal_slu.jl)
################################################################################

#
# Solve against an elementary vector eₖ, overwriting b:
#
#   sgetrs_tmp!(F, Val(:L), Val(:N), b, k)   b ← A* eₖ    (column k of A*)
#   sgetrs_tmp!(F, Val(:L), Val(:T), b, k)   b ← A*ᵀ eₖ   (row k of A*)
#   sgetrs_tmp!(F, Val(:L), Val(:C), b, k)   greatest solution of A \ x ∧ eₖ = x
#   sgetrs_tmp!(F, Val(:L), Val(:R), b, k)   same, transposed
#
# Side :R treats b as a row vector.
#
function sgetrs_tmp!(F::ChordalSLU{<:Any, T, I}, side::Val, trans::Val, b::AbstractVector, k::Integer; nt::Integer = Threads.nthreads()) where {T, I}
    W, x, pool, sched = sgetrs_elem_workspace_tmp(F; nt)
    return sgetrs_tmp!(F, side, trans, b, k, W, x, pool, sched; nt)
end

#
# Allocate the workspace for the non-allocating method below. The schedule
# is built once here (over the whole tree 1:N) and reused across solves.
#
function sgetrs_elem_workspace_tmp(F::ChordalSLU{<:Any, T, I}; nt::Integer = Threads.nthreads(), nrhs::Integer = 1) where {T, I}
    W = DivisionWorkspace{T}(F.S.S, two(I))
    x = FVector{T}(undef, size(F, 1))
    pool = spool_mt(T, nt)
    sched = DivisionSchedule{T}(F.S.S; nt, nrhs)
    return W, x, pool, sched
end

#
# Non-allocating version. W, x, pool, sched come from
# sgetrs_elem_workspace_tmp(F); x must not alias b.
#
function sgetrs_tmp!(F::ChordalSLU{<:Any, T, I}, side::Val{SIDE}, trans::Val{TRANS}, b::AbstractVector, k::Integer, W::DivisionWorkspace{T}, x::AbstractVector{T}, pool, sched; nt::Integer = Threads.nthreads()) where {T, I, SIDE, TRANS}
    n = size(F, 1)

    @assert length(b) == length(x) == n
    @assert length(W.Mval) >= 2 * F.S.S.nFval
    @assert b !== x
    @assert 1 <= k <= n

    if isforward(:U, TRANS, SIDE)
        invp = F.cinvp
        perm = F.rperm
    else
        invp = F.rinvp
        perm = F.cperm
    end
    #
    #   x ← P eₖ
    #
    fill!(x, szero(F.s, T, trans))
    j = invp[k]
    x[j] = sone(F.s, T, trans)
    #
    #   x ← U* L* x
    #
    sgetrs_elem_tmp!(F.s, side, trans, F.L, F.U, F.S.Bptr, F.S.Fptr, F.S.nBptr, pointers(F.S.N), targets(F.S.N), F.Nval, x, W, pool, j, nt, sched)
    #
    #   b ← P⁻¹ x
    #
    @inbounds for i in eachindex(x)
        b[perm[i]] = x[i]
    end

    return b
end
