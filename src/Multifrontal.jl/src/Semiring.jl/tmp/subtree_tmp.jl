#=
================================================================================
 Subtree parallelism for the chordal semiring triangular solve  (strsx!)
 Self-contained change set for CliqueTrees.jl
   src/Multifrontal.jl/src/Semiring.jl/src/chordal/
 generated against upstream commit 576f26a
================================================================================

 This file has THREE parts, one per destination:

   PART 1  chordal/chordal.jl   add one include line
   PART 2  chordal/strsx.jl     one new function + 8 replaced functions
   PART 3  chordal/strsx_tp.jl  a NEW file (copy PART 3 verbatim)

 Only forward (scatter) kernels change in strsx.jl; every backward kernel,
 the dense kernels and all other files are untouched. Each forward kernel
 gains a trailing argument `bnd = nothing`. With bnd === nothing (every
 call with nt = 1, and all sequential work) the code is the original code.

 Shortcut: because PART 2 only (re)defines methods with the same signatures
 as the originals, you can also try this without editing any source,
 as an overlay on a stock installation:

     using CliqueTrees
     Base.include(CliqueTrees.Multifrontal.Semiring, "strsx_subtree.jl")

 (verified on stock 576f26a). For a permanent change, move the parts into
 the files below.
================================================================================
=#


# ##############################################################################
# PART 1 — chordal/chordal.jl
# ##############################################################################
#
# Add the include of the new file directly after strsx.jl:
#
#     include("sgetrf.jl")
#     include("strsx.jl")
#     include("strsx_tp.jl")      # <-- NEW
#     include("strtri.jl")
#     include("sgetrs.jl")
#     include("sgetri.jl")
#     include("sgetrp.jl")
#
# (Nothing to paste from this part.)


# ##############################################################################
# PART 2 — chordal/strsx.jl
# ##############################################################################

# ------------------------------------------------------------------------------
# 2a. REPLACE the second `function strsx_mt!(` in strsx.jl — the one whose
#     arguments end in `W::DivisionWorkspace{T}, pool, nt, fstrt::I, fstop::I`
#     (it contains the `for f in fstrt:fstop` / `for f in reverse(...)` loops).
#     It becomes a small dispatcher ...
# ------------------------------------------------------------------------------

function strsx_solve_tmp!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
        fstrt::I,
        fstop::I,
        sched = nothing,
    ) where {SIDE, TRANS, UPLO, T, I}
    if nt > 1 && (isnothing(sched) ? strsx_tp_tmp!(s, side, trans, diag, A, B, W, pool, nt, fstrt, fstop) : strsx_tp_tmp!(s, side, trans, diag, A, B, W, pool, nt, fstrt, fstop, sched))
        return B
    end

    return strsx_rng_tmp!(s, side, trans, diag, A, B, W, pool, nt, fstrt, fstop, nothing)
end

function sgetrs_tmp!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        L::ChordalTriangular{<:Any, :L, T, I},
        U::ChordalTriangular{<:Any, :U, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
        fstrt::I,
        fstop::I,
        sched = nothing,
    ) where {SIDE, TRANS, T, I}
    if isforward(:L, TRANS, SIDE)
        strsx_solve_tmp!(s, side, trans, Val(:U), L, B, W, pool, nt, fstrt, fstop, sched)
        strsx_solve_tmp!(s, side, trans, Val(:N), U, B, W, pool, nt, fstrt, fstop, sched)
    else
        strsx_solve_tmp!(s, side, trans, Val(:N), U, B, W, pool, nt, fstrt, fstop, sched)
        strsx_solve_tmp!(s, side, trans, Val(:U), L, B, W, pool, nt, fstrt, fstop, sched)
    end

    return B
end

# ------------------------------------------------------------------------------
# 2b. ... and ADD this new function right after it. It is the old body of that
#     strsx_mt! (the sequential sweep), with one extra argument `bnd` that is
#     passed to strsx_fwd_1_tmp! and strsx_fwd_tmp!.
# ------------------------------------------------------------------------------

#
# Sequential sweep over the fronts fstrt:fstop.
#
# If bnd is not nothing, then fstrt:fstop is a union of complete
# subtrees, and forward scatters into rows outside of it are
# redirected into the private accumulator bnd.P.
#
function strsx_rng_tmp!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
        fstrt::I,
        fstop::I,
        bnd,
    ) where {SIDE, TRANS, UPLO, T, I}
    S = A.S

    if B isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    if isforward(UPLO, TRANS, SIDE)
        for f in fstrt:fstop
            nn = eltypedegree(S.res, f)

            if isone(nn)
                strsx_fwd_1_tmp!(s, B, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, f, trans, A.uplo, diag, side, bnd)
            else
                strsx_fwd_tmp!(s, B, W.Mval, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, f, trans, A.uplo, diag, side, bnd)
            end
        end
    else
        for f in reverse(fstrt:fstop)
            nn = eltypedegree(S.res, f)

            if isone(nn)
                strsx_bwd_1!(s, B, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, nrhs, f, trans, A.uplo, diag, side)
            else
                strsx_bwd!(s, B, W.Mval, A.Dval, A.Lval, S.Dptr, S.Lptr, S.res, S.sep, pool, nt, nrhs, f, trans, A.uplo, diag, side)
            end
        end
    end

    return B
end

# ------------------------------------------------------------------------------
# 2c. REPLACE `function strsx_fwd_tmp!(` in strsx.jl (there is only one).
#     Change: new trailing argument `bnd = nothing`, passed to both callees.
# ------------------------------------------------------------------------------

function strsx_fwd_tmp!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        pool,
        nt::Integer,
        nrhs::I,
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val,
        side::Val{SIDE},
        bnd = nothing,
    ) where {T, I, TRANS, UPLO, SIDE}
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
    Dp = Dptr[f]
    Lp = Lptr[f]
    #
    #          res(f)
    #     L = [ D₁₁ ] res(f)
    #         [ L₂₁ ] sep(f)
    #
    D₁₁ = reshape(view(Dval, Dp:Dp + nn * nn - one(I)), nn, nn)

    if UPLO === :L
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), na, nn)
    else
        L₂₁ = reshape(view(Lval, Lp:Lp + nn * na - one(I)), nn, na)
    end

    if SIDE === :R && nn < STRSX_NB
        strsx_fwd_upd_small_tmp!(s, C, D₁₁, L₂₁, res, sep, nn, na, nrhs, f, diag, trans, uplo, side, bnd)
    else
        strsx_fwd_upd_tmp!(s, C, Mval, D₁₁, L₂₁, res, sep, na, nrhs, pool, nt, f, trans, uplo, diag, side, bnd)
    end

    return
end

# ------------------------------------------------------------------------------
# 2d. REPLACE `function strsx_fwd_upd_tmp!(` in strsx.jl (there is only one).
#     Change: new trailing argument `bnd`; the final scatter C₂ ← C₂ + M₂ is
#     split at the chunk boundary when bnd !== nothing.
# ------------------------------------------------------------------------------

function strsx_fwd_upd_tmp!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Mval::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        na::I,
        nrhs::I,
        pool,
        nt::Integer,
        f::I,
        trans::Val,
        uplo::Val,
        diag::Val,
        side::Val{SIDE},
        bnd = nothing,
    ) where {T, I, SIDE}
    #
    #   C = [ C₁ ] res(f)
    #       [ C₂ ] sep(f)
    #
    #
    # fres is the residual at node f
    #
    #     fres = res(f)
    #
    fres = neighbors(res, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    if C isa AbstractVector
        C₁ = view(C, fres)
    elseif SIDE === :L
        C₁ = view(C, fres, oneto(nrhs))
    else
        C₁ = view(C, oneto(nrhs), fres)
    end
    #
    #   C₁ ← L₁₁* C₁
    #
    if C isa AbstractVector
        strsx!(s, side, trans, uplo, diag, D₁₁, C₁)
    else
        strsx_mt!(s, side, trans, uplo, diag, D₁₁, C₁, pool, nt)
    end

    if ispositive(na)
        if C isa AbstractVector
            M₂ = view(Mval, oneto(na))
        elseif SIDE === :L
            M₂ = reshape(view(Mval, oneto(na * nrhs)), na, nrhs)
        else
            M₂ = reshape(view(Mval, oneto(na * nrhs)), nrhs, na)
        end
        #
        #   M₂ ← L₂₁ C₁
        #
        szerorec!(s, M₂, trans)

        if C isa AbstractVector
            if SIDE === :L
                sgemx!(s, trans, Val(:N), M₂, L₂₁, C₁)
            else
                sgemx!(s, Val(:N), trans, M₂, C₁, L₂₁)
            end
        elseif SIDE === :L
            sgemx_mt!(s, trans, Val(:N), M₂, L₂₁, C₁, pool, nt)
        else
            sgemx_mt!(s, Val(:N), trans, M₂, C₁, L₂₁, pool, nt)
        end
        #
        #   C₂ ← C₂ + M₂
        #
        if isnothing(bnd) || tpsplit(bnd, fsep, na) == na
            if C isa AbstractVector
                sscatteradd!(s, trans, C, M₂, fsep)
            else
                sscatteradd!(s, trans, C, M₂, fsep, side)
            end
        else
            #
            # split sep(f) into rows inside the subtree (k₁)
            # and rows outside of it (k₂); the latter are
            # accumulated privately in bnd.P
            #
            k = tpsplit(bnd, fsep, na)
            k₁ = oneto(k)
            k₂ = k + one(I):na

            if C isa AbstractVector
                sscatteradd!(s, trans, C, view(M₂, k₁), view(fsep, k₁))
                sscatteradd!(s, trans, bnd.P, view(M₂, k₂), tpslots!(bnd, fsep, k, na))
            elseif SIDE === :L
                sscatteradd!(s, trans, C, view(M₂, k₁, oneto(nrhs)), view(fsep, k₁), side)
                sscatteradd!(s, trans, bnd.P, view(M₂, k₂, oneto(nrhs)), tpslots!(bnd, fsep, k, na), side)
            else
                sscatteradd!(s, trans, C, view(M₂, oneto(nrhs), k₁), view(fsep, k₁), side)
                sscatteradd!(s, trans, bnd.P, view(M₂, oneto(nrhs), k₂), tpslots!(bnd, fsep, k, na), side)
            end
        end
    end

    return
end

# ------------------------------------------------------------------------------
# 2e. REPLACE `function strsx_fwd_upd_small_tmp!(` #1 of 4 in strsx.jl — the method with
#     `C::AbstractVector{T}`, `trans::N_OR_R`, `::Val{:U}`, `::Val{:R}`.
#     Change: new trailing argument `bnd`; the separator scatter loop moved into
#     sfwd_scatter_small! (PART 3), which is the same loop when bnd === nothing.
# ------------------------------------------------------------------------------

function strsx_fwd_upd_small_tmp!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::N_OR_R,
        ::Val{:U},
        ::Val{:R},
        bnd = nothing,
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    @inbounds for j in oneto(nn)
        c = Rp + j - one(I)

        for i in oneto(j - one(I))
            C[c] = smuladd(s, C[Rp + i - one(I)], D₁₁[i, j], C[c], Val(:N), trans)
        end

        if !isintegral(s) && diag === Val(:N)
            C[c] = sprod(s, C[c], sstar(s, D₁₁[j, j]), Val(:N), trans)
        end
    end

    if ispositive(na)
        sfwd_scatter_small!(s, trans, Val(:U), C, fsep, L₂₁, Rp, nn, na, nrhs, bnd)
    end

    return
end

# ------------------------------------------------------------------------------
# 2f. REPLACE `function strsx_fwd_upd_small_tmp!(` #2 of 4 in strsx.jl — the method with
#     `C::AbstractMatrix{T}`, `trans::N_OR_R`, `::Val{:U}`, `::Val{:R}`.
#     Change: new trailing argument `bnd`; the separator scatter loop moved into
#     sfwd_scatter_small! (PART 3), which is the same loop when bnd === nothing.
# ------------------------------------------------------------------------------

function strsx_fwd_upd_small_tmp!(
        s::AbstractSemiring,
        C::AbstractMatrix{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::N_OR_R,
        ::Val{:U},
        ::Val{:R},
        bnd = nothing,
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    Z = sizeof(T)
    sC = stride(C, 2)

    @preserve C begin
        pC = pointer(C)

        @inbounds for j in oneto(nn)
            c = Rp + j - one(I)
            pc = pC + (c - one(I)) * sC * Z

            for i in oneto(j - one(I))
                d = Rp + i - one(I)
                saxpy_kern!(s, Val(:N), trans, Val(:R), pc, pC + (d - one(I)) * sC * Z, D₁₁[i, j], nrhs)
            end

            if !isintegral(s) && diag === Val(:N)
                v = sstar(s, D₁₁[j, j])

                for k in oneto(nrhs)
                    C[k, c] = sprod(s, C[k, c], v, Val(:N), trans)
                end
            end
        end
    end

    if ispositive(na)
        sfwd_scatter_small!(s, trans, Val(:U), C, fsep, L₂₁, Rp, nn, na, nrhs, bnd)
    end

    return
end

# ------------------------------------------------------------------------------
# 2g. REPLACE `function strsx_fwd_upd_small_tmp!(` #3 of 4 in strsx.jl — the method with
#     `C::AbstractVector{T}`, `trans::T_OR_C`, `::Val{:L}`, `::Val{:R}`.
#     Change: new trailing argument `bnd`; the separator scatter loop moved into
#     sfwd_scatter_small! (PART 3), which is the same loop when bnd === nothing.
# ------------------------------------------------------------------------------

function strsx_fwd_upd_small_tmp!(
        s::AbstractSemiring,
        C::AbstractVector{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::T_OR_C,
        ::Val{:L},
        ::Val{:R},
        bnd = nothing,
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    @inbounds for i in oneto(nn)
        ci = Rp + i - one(I)

        if !isintegral(s) && diag === Val(:N)
            C[ci] = sprod(s, C[ci], sstar(s, D₁₁[i, i]), Val(:N), trans)
        end

        for j in i + one(I):nn
            cj = Rp + j - one(I)
            C[cj] = smuladd(s, C[ci], D₁₁[j, i], C[cj], Val(:N), trans)
        end
    end

    if ispositive(na)
        sfwd_scatter_small!(s, trans, Val(:L), C, fsep, L₂₁, Rp, nn, na, nrhs, bnd)
    end

    return
end

# ------------------------------------------------------------------------------
# 2h. REPLACE `function strsx_fwd_upd_small_tmp!(` #4 of 4 in strsx.jl — the method with
#     `C::AbstractMatrix{T}`, `trans::T_OR_C`, `::Val{:L}`, `::Val{:R}`.
#     Change: new trailing argument `bnd`; the separator scatter loop moved into
#     sfwd_scatter_small! (PART 3), which is the same loop when bnd === nothing.
# ------------------------------------------------------------------------------

function strsx_fwd_upd_small_tmp!(
        s::AbstractSemiring,
        C::AbstractMatrix{T},
        D₁₁::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nn::I,
        na::I,
        nrhs::I,
        f::I,
        diag::Val,
        trans::T_OR_C,
        ::Val{:L},
        ::Val{:R},
        bnd = nothing,
    ) where {T, I}
    Rp = pointers(res)[f]
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)

    Z = sizeof(T)
    sC = stride(C, 2)

    @preserve C begin
        pC = pointer(C)

        @inbounds for i in oneto(nn)
            ci = Rp + i - one(I)
            pci = pC + (ci - one(I)) * sC * Z

            if !isintegral(s) && diag === Val(:N)
                v = sstar(s, D₁₁[i, i])

                for k in oneto(nrhs)
                    C[k, ci] = sprod(s, C[k, ci], v, Val(:N), trans)
                end
            end

            for j in i + one(I):nn
                cj = Rp + j - one(I)
                saxpy_kern!(s, Val(:N), trans, Val(:R), pC + (cj - one(I)) * sC * Z, pci, D₁₁[j, i], nrhs)
            end
        end
    end

    if ispositive(na)
        sfwd_scatter_small!(s, trans, Val(:L), C, fsep, L₂₁, Rp, nn, na, nrhs, bnd)
    end

    return
end

# ------------------------------------------------------------------------------
# 2i. REPLACE `function strsx_fwd_1_tmp!(` in strsx.jl (there is only one).
#     Change: new trailing argument `bnd`; the separator scatter moved into
#     sfwd_scatter_1! (PART 3), which is the same loop when bnd === nothing.
# ------------------------------------------------------------------------------

function strsx_fwd_1_tmp!(
        s::AbstractSemiring,
        C::AbstractVecOrMat{T},
        Dval::AbstractVector{T},
        Lval::AbstractVector{T},
        Dptr::AbstractVector{I},
        Lptr::AbstractVector{I},
        res::AbstractGraph{I},
        sep::AbstractGraph{I},
        nrhs::I,
        f::I,
        trans::Val{TRANS},
        uplo::Val{UPLO},
        diag::Val{DIAG},
        side::Val{SIDE},
        bnd = nothing,
    ) where {T, I, TRANS, UPLO, SIDE, DIAG}
    #
    # na is the size of the separator at node f
    #
    #     na = | sep(f) |
    #
    na = eltypedegree(sep, f)
    #
    # fsep is the separator at node f
    #
    #     fsep = sep(f)
    #
    fsep = neighbors(sep, f)
    Dp = Dptr[f]
    Lp = Lptr[f]
    Rp = pointers(res)[f]
    #
    #          res(f)
    #     L = [ d₁₁ ] res(f)
    #         [ l₂₁ ] sep(f)
    #
    d₁₁ = Dval[Dp]
    l₂₁ = view(Lval, Lp:Lp + na - one(I))
    #
    #   c₁ ← d₁₁* c₁
    #
    if !isintegral(s) && DIAG === :N
        v = sstar(s, d₁₁)

        if C isa AbstractVector
            if SIDE === :L
                C[Rp] = sprod(s, v, C[Rp], trans, Val(:N))
            else
                C[Rp] = sprod(s, C[Rp], v, Val(:N), trans)
            end
        else
            @inbounds for k in oneto(nrhs)
                if SIDE === :L
                    C[Rp, k] = sprod(s, v, C[Rp, k], trans, Val(:N))
                else
                    C[k, Rp] = sprod(s, C[k, Rp], v, Val(:N), trans)
                end
            end
        end
    end

    if ispositive(na)
        #
        #   M₂ ← l₂₁ c₁       C₂ ← C₂ + M₂
        #
        sfwd_scatter_1!(s, trans, side, C, fsep, l₂₁, Rp, na, nrhs, bnd)
    end

    return
end


# ##############################################################################
# PART 3 — chordal/strsx_tp.jl   (NEW FILE: copy everything below this banner)
# ##############################################################################

# ===== subtree parallelism for strsx! =====
#
# The fronts fstrt:fstop are postordered, so every subtree is a
# contiguous range of fronts, and (since residuals are numbered
# by front) a contiguous range of rows. We partition the fronts
# into
#
#   - chunks: disjoint unions of complete subtrees, each a
#     contiguous range of fronts, processed in parallel with
#     one thread per chunk
#
#   - the top: every other front, processed sequentially with
#     fine-grain parallelism
#
# Backward (gather) sweeps read sep(f) and write res(f). Once the
# top is done, chunks touch disjoint rows, so they need no
# synchronization.
#
# Forward (scatter) sweeps write sep(f). Let r be the root of a
# subtree in a chunk. Then, for every front f in the subtree,
#
#   sep(f) ⊆ rows(subtree) ∪ sep(r),
#
# and since sep(f) is sorted, the rows outside of the chunk form
# a suffix of sep(f). We redirect that suffix into a private
# accumulator indexed by the union of the separators of the
# chunk roots, and fold the accumulators into the right-hand
# side (in chunk order, so the result is deterministic) before
# processing the top.

# minimum work (stored entries × right-hand sides)
const STRSX_TP_WORK = 1 << 16

# chunks per worker targeted when merging small subtrees
const STRSX_TP_SPLIT = 4

# fixed per-front cost, in units of stored entries
const STRSX_TP_FRONT = 16

# minimum predicted speedup
const STRSX_TP_GAIN = 1.25

# slowdown of a chunk relative to a sequential sweep over the
# same fronts (boundary handling, lost locality); measured at
# roughly 1.2-1.4 on one thread
const STRSX_TP_EFF = 1.3

# (functions, so that tests can force the parallel path)
strsx_tp_minwork() = STRSX_TP_WORK
strsx_tp_gain() = STRSX_TP_GAIN

# maximum number of fronts expanded by the partitioner
const STRSX_TP_EXPAND = 4096

struct TPPlan{I}
    fd::Vector{I}       # first descendant (subtree start) of every front
    cbnd::Vector{Int}   # upper bound on the boundary rows of chunk c
    rptr::Vector{Int}   # the subtree roots in chunk c are
    rts::Vector{I}      #   rts[rptr[c]:rptr[c + 1] - 1]
    order::Vector{Int}  # chunks by decreasing work
    nw::Int
end

# chunk c is the fronts tpstrt(plan, c):tpstop(plan, c), derived from the
# roots: a chunk is a union of adjacent subtrees, so it starts at the first
# descendant of its smallest root and ends at its largest root.
@inline tpnc(plan::TPPlan) = length(plan.rptr) - 1
@inline tpstrt(plan::TPPlan, c::Integer) = @inbounds plan.fd[plan.rts[plan.rptr[c]]]
@inline tpstop(plan::TPPlan, c::Integer) = @inbounds plan.rts[plan.rptr[c + 1] - 1]

#
# first descendant of every front in fstrt:fstop (a union of complete
# subtrees). Fronts are postordered, so a subtree rooted at f is
# fd[f]:f; fd[f] = min over the subtree, propagated leaf-to-root in
# one ascending pass (children precede their parent).
#
function tpfdesc(S::ChordalSymbolic{I}, fstrt::I, fstop::I) where {I}
    pnt = S.pnt
    fd = Vector{I}(undef, convert(Int, fstop))

    @inbounds for f in fstrt:fstop
        fd[f] = f
    end

    @inbounds for f in fstrt:fstop
        p = pnt[f]

        if fstrt <= p <= fstop && fd[f] < fd[p]
            fd[p] = fd[f]
        end
    end

    return fd
end

#
# work of the fronts fa:fb, in units of stored entries of the
# triangular factor, computed in O(1) from the prefix sums Dptr
# and Lptr
#
@inline function tpwork(S::ChordalSymbolic{I}, fa::I, fb::I) where {I}
    @inbounds nd = convert(Int, S.Dptr[fb + one(I)] - S.Dptr[fa])
    @inbounds nl = convert(Int, S.Lptr[fb + one(I)] - S.Lptr[fa])
    return nd >> 1 + nl + STRSX_TP_FRONT * convert(Int, fb - fa + one(I))
end

# max-heap of (work, root)
@inline function tpheap_push!(h::Vector{Tuple{Int, I}}, x::Tuple{Int, I}) where {I}
    push!(h, x)
    i = length(h)

    @inbounds while i > 1
        j = i >> 1
        first(h[j]) >= first(h[i]) && break
        h[i], h[j] = h[j], h[i]
        i = j
    end

    return h
end

@inline function tpheap_pop!(h::Vector{Tuple{Int, I}}) where {I}
    @inbounds x = h[1]
    y = pop!(h)
    n = length(h)

    if n > 0
        @inbounds h[1] = y
        i = 1

        @inbounds while true
            l = 2i; r = l + 1; j = i
            l <= n && first(h[l]) > first(h[j]) && (j = l)
            r <= n && first(h[r]) > first(h[j]) && (j = r)
            j == i && break
            h[i], h[j] = h[j], h[i]
            i = j
        end
    end

    return x
end

#
# Partition the fronts fstrt:fstop (a union of complete subtrees) into parallel
# chunks and a serial top: expand subtrees until each is under a work target
# (total / (STRSX_TP_SPLIT * nt)), then merge adjacent small ones into chunks.
#
function tpplan(S::ChordalSymbolic{I}, fd::Vector{I}, nrhs::Integer, nt::Integer, fstrt::I, fstop::I) where {I}
    if fstop <= fstrt
        return nothing
    end

    total = tpwork(S, fstrt, fstop)

    if total * nrhs < strsx_tp_minwork()
        return nothing
    end

    chd = S.chd
    sep = S.sep
    res = S.res
    roots = neighbors(chd, nv(res) + one(I))
    ra = searchsortedfirst(roots, fstrt)
    rb = searchsortedlast(roots, fstop)

    if ra > rb || roots[rb] != fstop
        return nothing
    end

    target = cld(total, STRSX_TP_SPLIT * nt)
    heap = Tuple{Int, I}[]
    #
    # a subtree rooted at r spans fd[r]:r, so its work and its
    # first front both come from fd — no running cursor needed
    #
    for i in ra:rb
        r = roots[i]
        tpheap_push!(heap, (tpwork(S, fd[r], r), r))
    end

    nexp = 0

    while !isempty(heap) && first(heap[1]) > target
        nexp += 1

        if nexp > STRSX_TP_EXPAND
            return nothing
        end

        _, f = tpheap_pop!(heap)

        for c in neighbors(chd, f)
            tpheap_push!(heap, (tpwork(S, fd[c], c), c))
        end
    end

    sort!(heap; by = x -> x[2])
    rlast = pointers(res)[fstop + one(I)] - one(I)

    cwork = Int[]
    cbnd = Int[]
    rptr = Int[]
    rts = I[]

    for (w, r) in heap
        na = eltypedegree(sep, r)

        if ispositive(na) && neighbors(sep, r)[na] > rlast
            return nothing
        end
        #
        # the previous root is the current chunk's last front; the new
        # subtree starts at fd[r], so it extends the chunk iff adjacent
        #
        if !isempty(rts) && rts[end] + one(I) == fd[r] && cwork[end] + w <= target
            cwork[end] += w
            cbnd[end] += na
        else
            push!(cwork, w)
            push!(cbnd, na)
            push!(rptr, length(rts) + 1)
        end

        push!(rts, r)
    end

    push!(rptr, length(rts) + 1)
    nc = length(rptr) - 1

    if nc < 2
        return nothing
    end

    wpar = sum(cwork)
    wtop = total - wpar
    nw = min(nt, nc)
    tpar = wtop + STRSX_TP_EFF * max(maximum(cwork), cld(wpar, nw))

    if total < strsx_tp_gain() * tpar
        return nothing
    end

    order = sortperm(cwork; rev = true)
    return TPPlan{I}(fd, cbnd, rptr, rts, order, nw)
end

# ranges of top fronts between consecutive chunks
function tpgaps(plan::TPPlan{I}, fstrt::I, fstop::I) where {I}
    gaps = UnitRange{I}[]
    a = fstrt

    for c in oneto(tpnc(plan))
        cs = tpstrt(plan, c)

        if a < cs
            push!(gaps, a:cs - one(I))
        end

        a = tpstop(plan, c) + one(I)
    end

    if a <= fstop
        push!(gaps, a:fstop)
    end

    return gaps
end

# ===== private boundary accumulators =====

#
# The rows of the top fronts are numbered 1, 2, ..., ntop. They
# form one contiguous interval of rows per gap: gap g starts at
# row rs[g], and its rows are numbered from ro[g] + 1.
#
struct TPTop{I}
    rs::Vector{I}
    ro::Vector{I}
end

@inline function tpindex(top::TPTop{I}, u::I) where {I}
    g = searchsortedlast(top.rs, u)
    return @inbounds top.ro[g] + (u - top.rs[g]) + one(I)
end

struct TPBoundary{I, P}
    hi::I               # last row of the chunk
    P::P                # private accumulator
    top::TPTop{I}       # numbering of the top rows
    slot::FVector{I}    # slot in P of each top row (by number)
    sbuf::FVector{I}    # scratch
end

#
# the slots in P of the rows fsep[k + 1:na], all of which are
# top rows; both fsep and the gaps are sorted, so after one
# binary search the translation costs O(1) per row
#
@inline function tpslots!(bnd::TPBoundary{I}, fsep::AbstractVector{I}, k::I, na::I) where {I}
    rs = bnd.top.rs
    ro = bnd.top.ro
    ng = length(rs)
    g = searchsortedlast(rs, fsep[k + one(I)])

    @inbounds for i in k + one(I):na
        u = fsep[i]

        while g < ng && rs[g + 1] <= u
            g += 1
        end

        bnd.sbuf[i - k] = bnd.slot[ro[g] + (u - rs[g]) + one(I)]
    end

    return view(bnd.sbuf, oneto(na - k))
end

#
# number of rows of fsep inside the chunk; the rows outside
# form a (short) suffix, so scan backwards
#
@inline function tpsplit(bnd::TPBoundary{I}, fsep::AbstractVector{I}, na::I) where {I}
    k = na

    @inbounds while ispositive(k) && fsep[k] > bnd.hi
        k -= one(I)
    end

    return k
end

function tpbuffer(B::AbstractVector, Pbuf::AbstractVector, off::Int, nU::Integer, nrhs::Integer, ::Val)
    return view(Pbuf, off + 1:off + nU)
end

function tpbuffer(B::AbstractMatrix, Pbuf::AbstractVector, off::Int, nU::Integer, nrhs::Integer, ::Val{SIDE}) where {SIDE}
    if SIDE === :L
        return reshape(view(Pbuf, off + 1:off + nU * nrhs), nU, nrhs)
    else
        return reshape(view(Pbuf, off + 1:off + nU * nrhs), nrhs, nU)
    end
end

# ===== forward scatter kernels =====
#
# These implement the scatter step of strsx_fwd_upd_small_tmp!
# and strsx_fwd_1_tmp!. The target D is either C itself or a
# private accumulator; ind holds the target rows, and a₀
# is the offset of ind in sep(f).
#

@inline function tpcoef(L₂₁::AbstractMatrix, a, b, ::Val{:U})
    return @inbounds L₂₁[b, a]
end

@inline function tpcoef(L₂₁::AbstractMatrix, a, b, ::Val{:L})
    return @inbounds L₂₁[a, b]
end

@inline function sfwd_scatter_small!(
        s::AbstractSemiring,
        trans::Val,
        uplo::Val,
        C::AbstractVecOrMat,
        fsep::AbstractVector{I},
        L₂₁::AbstractMatrix,
        Rp::I,
        nn::I,
        na::I,
        nrhs::I,
        bnd,
    ) where {I}
    if isnothing(bnd) || tpsplit(bnd, fsep, na) == na
        sfwd_scatter_small_kern!(s, trans, uplo, C, fsep, zero(I), C, L₂₁, Rp, nn, nrhs)
    else
        k = tpsplit(bnd, fsep, na)
        sfwd_scatter_small_kern!(s, trans, uplo, C, view(fsep, oneto(k)), zero(I), C, L₂₁, Rp, nn, nrhs)
        sfwd_scatter_small_kern!(s, trans, uplo, bnd.P, tpslots!(bnd, fsep, k, na), k, C, L₂₁, Rp, nn, nrhs)
    end

    return
end

@inline function sfwd_scatter_small_kern!(
        s::AbstractSemiring,
        trans::Val,
        uplo::Val,
        D::AbstractVector{T},
        ind::AbstractVector{I},
        a₀::I,
        C::AbstractVector{T},
        L₂₁::AbstractMatrix{T},
        Rp::I,
        nn::I,
        nrhs::I,
    ) where {T, I}
    @inbounds for a in eachindex(ind)
        c = ind[a]

        for b in oneto(nn)
            D[c] = smuladd(s, C[Rp + b - one(I)], tpcoef(L₂₁, a₀ + a, b, uplo), D[c], Val(:N), trans)
        end
    end

    return
end

@inline function sfwd_scatter_small_kern!(
        s::AbstractSemiring,
        trans::Val,
        uplo::Val,
        D::AbstractMatrix{T},
        ind::AbstractVector{I},
        a₀::I,
        C::AbstractMatrix{T},
        L₂₁::AbstractMatrix{T},
        Rp::I,
        nn::I,
        nrhs::I,
    ) where {T, I}
    Z = sizeof(T)
    sC = stride(C, 2)
    sD = stride(D, 2)

    @preserve C D begin
        pC = pointer(C)
        pD = pointer(D)

        @inbounds for a in eachindex(ind)
            pd = pD + (ind[a] - one(I)) * sD * Z

            for b in oneto(nn)
                d = Rp + b - one(I)
                saxpy_kern!(s, Val(:N), trans, Val(:R), pd, pC + (d - one(I)) * sC * Z, tpcoef(L₂₁, a₀ + a, b, uplo), nrhs)
            end
        end
    end

    return
end

@inline function sfwd_scatter_1!(
        s::AbstractSemiring,
        trans::Val,
        side::Val,
        C::AbstractVecOrMat,
        fsep::AbstractVector{I},
        l₂₁::AbstractVector,
        Rp::I,
        na::I,
        nrhs::I,
        bnd,
    ) where {I}
    if isnothing(bnd) || tpsplit(bnd, fsep, na) == na
        sfwd_scatter_1_kern!(s, trans, side, C, fsep, zero(I), C, l₂₁, Rp, nrhs)
    else
        k = tpsplit(bnd, fsep, na)
        sfwd_scatter_1_kern!(s, trans, side, C, view(fsep, oneto(k)), zero(I), C, l₂₁, Rp, nrhs)
        sfwd_scatter_1_kern!(s, trans, side, bnd.P, tpslots!(bnd, fsep, k, na), k, C, l₂₁, Rp, nrhs)
    end

    return
end

@inline function sfwd_scatter_1_kern!(
        s::AbstractSemiring,
        trans::Val,
        ::Val{SIDE},
        D::AbstractVector{T},
        ind::AbstractVector{I},
        a₀::I,
        C::AbstractVector{T},
        l₂₁::AbstractVector{T},
        Rp::I,
        nrhs::I,
    ) where {T, I, SIDE}
    v = C[Rp]

    @inbounds for a in eachindex(ind)
        c = ind[a]

        if SIDE === :L
            D[c] = smuladd(s, l₂₁[a₀ + a], v, D[c], trans, Val(:N))
        else
            D[c] = smuladd(s, v, l₂₁[a₀ + a], D[c], Val(:N), trans)
        end
    end

    return
end

@inline function sfwd_scatter_1_kern!(
        s::AbstractSemiring,
        trans::Val,
        ::Val{SIDE},
        D::AbstractMatrix{T},
        ind::AbstractVector{I},
        a₀::I,
        C::AbstractMatrix{T},
        l₂₁::AbstractVector{T},
        Rp::I,
        nrhs::I,
    ) where {T, I, SIDE}
    if SIDE === :L
        @inbounds for k in oneto(nrhs)
            v = C[Rp, k]

            for a in eachindex(ind)
                c = ind[a]
                D[c, k] = smuladd(s, l₂₁[a₀ + a], v, D[c, k], trans, Val(:N))
            end
        end
    else
        Z = sizeof(T)
        sC = stride(C, 2)
        sD = stride(D, 2)

        @preserve C D begin
            pD = pointer(D)
            pr = pointer(C) + (Rp - one(I)) * sC * Z

            @inbounds for a in eachindex(ind)
                saxpy_kern!(s, Val(:N), trans, Val(:R), pD + (ind[a] - one(I)) * sD * Z, pr, l₂₁[a₀ + a], nrhs)
            end
        end
    end

    return
end

# ===== prebuilt, allocation-free schedule =====
#
# DivisionSchedule holds the whole plan (partition + top-numbering +
# accumulator storage) and the per-worker scratch, built once for a
# given (S, nt, nrhs). A solve reuses all of it and allocates nothing.
# The slot maps stay zeroed because each chunk resets the top rows it
# touched, so there is no per-solve fill!.
#
struct DivisionSchedule{T, I}
    nt::Int
    fd::Vector{I}       # first descendant of every front (whole tree)
    plan::Union{Nothing, TPPlan{I}}
    gaps::Vector{UnitRange{I}}
    top::TPTop{I}
    ntop::I
    Poff::Vector{Int}
    Uoff::Vector{Int}
    Ulen::Vector{I}
    Pbuf::FVector{T}
    Ubuf::FVector{I}
    next::Threads.Atomic{Int}
    Ms::Vector{FVector{T}}
    slots::Vector{FVector{I}}
    sbufs::Vector{FVector{I}}
end

function DivisionSchedule{T}(S::ChordalSymbolic{I}; nt::Integer = Threads.nthreads(), nrhs::Integer = 1) where {T, I}
    o = one(I); N = nv(S.res); nF = convert(Int, S.nFval)
    fd = tpfdesc(S, o, N)
    plan = tpplan(S, fd, nrhs, nt, o, N)

    if isnothing(plan)
        e0 = FVector{T}(undef, 0); e1 = FVector{I}(undef, 0)
        return DivisionSchedule{T, I}(convert(Int, nt), fd, nothing, UnitRange{I}[], TPTop{I}(I[], I[]), zero(I),
            Int[], Int[], I[], e0, e1, Threads.Atomic{Int}(0), FVector{T}[], FVector{I}[], FVector{I}[])
    end

    nw = min(convert(Int, nt), plan.nw)
    gaps = tpgaps(plan, o, N)
    top, ntop = tpnumber(S.res, gaps)
    Poff, Uoff, Ulen, Pbuf, Ubuf = tpstorage(T, plan, ntop, nrhs)
    Ms = [FVector{T}(undef, nF * convert(Int, nrhs)) for _ in oneto(nw)]
    slots = FVector{I}[]

    for _ in oneto(nw)
        v = FVector{I}(undef, convert(Int, ntop)); fill!(v, zero(I)); push!(slots, v)
    end

    sbufs = [FVector{I}(undef, nF) for _ in oneto(nw)]
    return DivisionSchedule{T, I}(convert(Int, nt), fd, plan, gaps, top, ntop, Poff, Uoff, Ulen, Pbuf, Ubuf,
        Threads.Atomic{Int}(0), Ms, slots, sbufs)
end

# cached (allocation-free) driver: plan/gaps/top/storage/scratch all live in sched
function strsx_tp_tmp!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
        fstrt::I,
        fstop::I,
        sched::DivisionSchedule{T, I},
    ) where {SIDE, TRANS, UPLO, T, I}
    isnothing(sched.plan) && return false

    if B isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end

    if B isa AbstractMatrix && nrhs >= 4nt && SIDE === :L
        return false
    end

    nw = length(sched.Ms)

    if !isnothing(pool)
        nw = min(nw, length(pool))
    end

    nw < 2 && return false
    Threads.atomic_xchg!(sched.next, 0)

    if isforward(UPLO, TRANS, SIDE)
        strsx_tp_fwd_sched!(s, side, trans, diag, A, B, W, pool, nt, nw, nrhs, sched)
    else
        strsx_tp_bwd_sched!(s, side, trans, diag, A, B, W, pool, nt, nw, nrhs, sched)
    end

    return true
end

function strsx_tp_fwd_sched!(s, side::Val, trans, diag, A, B, W, pool, nt, nw, nrhs::I, sched::DivisionSchedule{T, I}) where {T, I}
    plan = sched.plan
    tasks = Vector{Task}(undef, nw - 1)

    for w in 2:nw
        tasks[w - 1] = @spawn strsx_tp_fwd_worker!(s, side, trans, diag, A, B, sched.Ms[w], sched.slots[w], sched.sbufs[w], tppool(pool, w), plan, sched.next, sched.top, sched.Pbuf, sched.Poff, sched.Ubuf, sched.Uoff, sched.Ulen, nrhs)
    end

    strsx_tp_fwd_worker!(s, side, trans, diag, A, B, sched.Ms[1], sched.slots[1], sched.sbufs[1], tppool(pool, 1), plan, sched.next, sched.top, sched.Pbuf, sched.Poff, sched.Ubuf, sched.Uoff, sched.Ulen, nrhs)

    for t in tasks
        wait(t)
    end

    for c in oneto(tpnc(plan))
        m = sched.Ulen[c]
        iszero(m) && continue
        P = tpbuffer(B, sched.Pbuf, sched.Poff[c], m, nrhs, side)
        U = view(sched.Ubuf, sched.Uoff[c] + 1:sched.Uoff[c] + convert(Int, m))

        if B isa AbstractVector
            sscatteradd!(s, trans, B, P, U)
        else
            sscatteradd!(s, trans, B, P, U, side)
        end
    end

    for g in sched.gaps
        strsx_rng_tmp!(s, side, trans, diag, A, B, W, pool, nt, first(g), last(g), nothing)
    end

    return B
end

function strsx_tp_bwd_sched!(s, side, trans, diag, A, B, W, pool, nt, nw, nrhs::I, sched::DivisionSchedule{T, I}) where {T, I}
    plan = sched.plan

    for g in reverse(sched.gaps)
        strsx_rng_tmp!(s, side, trans, diag, A, B, W, pool, nt, first(g), last(g), nothing)
    end

    tasks = Vector{Task}(undef, nw - 1)

    for w in 2:nw
        tasks[w - 1] = @spawn strsx_tp_bwd_worker!(s, side, trans, diag, A, B, sched.Ms[w], tppool(pool, w), plan, sched.next)
    end

    strsx_tp_bwd_worker!(s, side, trans, diag, A, B, sched.Ms[1], tppool(pool, 1), plan, sched.next)

    for t in tasks
        wait(t)
    end

    return B
end

# ===== drivers =====

#
# Solve with subtree parallelism. Returns false (without
# touching B) if it is not worthwhile.
#
function strsx_tp_tmp!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val{TRANS},
        diag::Val,
        A::ChordalTriangular{<:Any, UPLO, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
        fstrt::I,
        fstop::I,
    ) where {SIDE, TRANS, UPLO, T, I}
    if B isa AbstractVector
        nrhs = one(I)
    elseif SIDE === :L
        nrhs = convert(I, size(B, 2))
    else
        nrhs = convert(I, size(B, 1))
    end
    #
    # wide right-hand sides are split by column instead
    #
    if B isa AbstractMatrix && nrhs >= 4nt && SIDE === :L
        return false
    end

    fd = tpfdesc(A.S, fstrt, fstop)
    plan = tpplan(A.S, fd, nrhs, nt, fstrt, fstop)

    if isnothing(plan)
        return false
    end

    nw = plan.nw

    if !isnothing(pool)
        nw = min(nw, length(pool))
    end

    if nw < 2
        return false
    end

    if isforward(UPLO, TRANS, SIDE)
        strsx_tp_fwd!(s, side, trans, diag, A, B, W, pool, nt, nw, nrhs, plan, fstrt, fstop)
    else
        strsx_tp_bwd!(s, side, trans, diag, A, B, W, pool, nt, nw, nrhs, plan, fstrt, fstop)
    end

    return true
end

function tppool(pool, w::Int)
    if isnothing(pool)
        return nothing
    else
        return view(pool, w:w)
    end
end

function strsx_tp_bwd!(
        s::AbstractSemiring,
        side::Val,
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
        nw::Int,
        nrhs::I,
        plan::TPPlan{I},
        fstrt::I,
        fstop::I,
    ) where {T, I}
    #
    # top first (root to leaves) ...
    #
    for g in reverse(tpgaps(plan, fstrt, fstop))
        strsx_rng_tmp!(s, side, trans, diag, A, B, W, pool, nt, first(g), last(g), nothing)
    end
    #
    # ... then the chunks, in parallel
    #
    next = Threads.Atomic{Int}(0)
    Mlen = length(W.Mval)
    tasks = Vector{Task}(undef, nw - 1)

    for w in 2:nw
        poolw = tppool(pool, w)
        Mw = FVector{T}(undef, Mlen)
        tasks[w - 1] = @spawn strsx_tp_bwd_worker!(s, side, trans, diag, A, B, Mw, poolw, plan, next)
    end

    strsx_tp_bwd_worker!(s, side, trans, diag, A, B, FVector{T}(undef, Mlen), tppool(pool, 1), plan, next)

    for task in tasks
        wait(task)
    end

    return B
end

function strsx_tp_bwd_worker!(
        s::AbstractSemiring,
        side::Val,
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractVecOrMat,
        Mw::FVector{T},
        pool,
        plan::TPPlan{I},
        next::Threads.Atomic{Int},
    ) where {T, I}
    W = DivisionWorkspace{T}(Mw)
    nc = length(plan.order)

    while true
        i = Threads.atomic_add!(next, 1) + 1
        i > nc && break
        c = plan.order[i]
        strsx_rng_tmp!(s, side, trans, diag, A, B, W, pool, 1, tpstrt(plan, c), tpstop(plan, c), nothing)
    end

    return
end

function strsx_tp_fwd!(
        s::AbstractSemiring,
        side::Val{SIDE},
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractVecOrMat,
        W::DivisionWorkspace{T},
        pool,
        nt::Integer,
        nw::Int,
        nrhs::I,
        plan::TPPlan{I},
        fstrt::I,
        fstop::I,
    ) where {SIDE, T, I}
    S = A.S
    res = S.res
    gaps = tpgaps(plan, fstrt, fstop)
    #
    # (the setup lives in separate functions so that
    # the variables captured by @spawn below are never
    # reassigned, which would box them)
    #
    top, ntop = tpnumber(res, gaps)
    Poff, Uoff, Ulen, Pbuf, Ubuf = tpstorage(T, plan, ntop, nrhs)
    nc = tpnc(plan)
    #
    # chunks, in parallel ...
    #
    next = Threads.Atomic{Int}(0)
    Mlen = length(W.Mval)
    nF = convert(Int, S.nFval)
    ntopI = convert(Int, ntop)
    tasks = Vector{Task}(undef, nw - 1)

    for w in 2:nw
        poolw = tppool(pool, w)
        Mw = FVector{T}(undef, Mlen)
        slotw = FVector{I}(undef, ntopI); fill!(slotw, zero(I))
        sbufw = FVector{I}(undef, nF)
        tasks[w - 1] = @spawn strsx_tp_fwd_worker!(s, side, trans, diag, A, B, Mw, slotw, sbufw, poolw, plan, next, top, Pbuf, Poff, Ubuf, Uoff, Ulen, nrhs)
    end

    let Mw = FVector{T}(undef, Mlen), slotw = FVector{I}(undef, ntopI), sbufw = FVector{I}(undef, nF)
        fill!(slotw, zero(I))
        strsx_tp_fwd_worker!(s, side, trans, diag, A, B, Mw, slotw, sbufw, tppool(pool, 1), plan, next, top, Pbuf, Poff, Ubuf, Uoff, Ulen, nrhs)
    end

    for task in tasks
        wait(task)
    end
    #
    # ... fold the accumulators into B, in chunk order ...
    #
    for c in oneto(nc)
        m = Ulen[c]
        iszero(m) && continue
        P = tpbuffer(B, Pbuf, Poff[c], m, nrhs, side)
        U = view(Ubuf, Uoff[c] + 1:Uoff[c] + convert(Int, m))

        if B isa AbstractVector
            sscatteradd!(s, trans, B, P, U)
        else
            sscatteradd!(s, trans, B, P, U, side)
        end
    end
    #
    # ... then the top (leaves to root)
    #
    for g in gaps
        strsx_rng_tmp!(s, side, trans, diag, A, B, W, pool, nt, first(g), last(g), nothing)
    end

    return B
end

#
# number the rows of the top fronts 1, 2, ..., ntop
#
function tpnumber(res::AbstractGraph{I}, gaps::Vector{UnitRange{I}}) where {I}
    rs = Vector{I}(undef, length(gaps))
    ro = Vector{I}(undef, length(gaps))
    ntop = zero(I)

    @inbounds for (i, g) in enumerate(gaps)
        rs[i] = pointers(res)[first(g)]
        ro[i] = ntop
        ntop += pointers(res)[last(g) + one(I)] - rs[i]
    end

    return TPTop{I}(rs, ro), ntop
end

#
# storage for the private accumulators; chunk c
# has at most min(cbnd[c], ntop) boundary rows
#
function tpstorage(::Type{T}, plan::TPPlan{I}, ntop::I, nrhs::I) where {T, I}
    nc = tpnc(plan)
    Poff = Vector{Int}(undef, nc)
    Uoff = Vector{Int}(undef, nc)
    Ulen = Vector{I}(undef, nc)
    nP = nU = 0

    for c in oneto(nc)
        m = min(plan.cbnd[c], convert(Int, ntop))
        Poff[c] = nP
        Uoff[c] = nU
        nP += m * convert(Int, nrhs)
        nU += m
    end

    Pbuf = FVector{T}(undef, nP)
    Ubuf = FVector{I}(undef, nU)
    return Poff, Uoff, Ulen, Pbuf, Ubuf
end

function strsx_tp_fwd_worker!(
        s::AbstractSemiring,
        side::Val,
        trans::Val,
        diag::Val,
        A::ChordalTriangular{<:Any, <:Any, T, I},
        B::AbstractVecOrMat,
        Mw::FVector{T},
        slot::FVector{I},
        sbuf::FVector{I},
        pool,
        plan::TPPlan{I},
        next::Threads.Atomic{Int},
        top::TPTop{I},
        Pbuf::FVector{T},
        Poff::Vector{Int},
        Ubuf::FVector{I},
        Uoff::Vector{Int},
        Ulen::Vector{I},
        nrhs::I,
    ) where {T, I}
    S = A.S
    res = S.res
    sep = S.sep

    W = DivisionWorkspace{T}(Mw)
    nc = length(plan.order)

    while true
        i = Threads.atomic_add!(next, 1) + 1
        i > nc && break
        c = plan.order[i]
        fa = tpstrt(plan, c)
        fb = tpstop(plan, c)
        uo = Uoff[c]
        #
        # U = ⋃ { sep(r) : r is a subtree root in the chunk }
        #
        m = zero(I)

        @inbounds for j in plan.rptr[c]:plan.rptr[c + 1] - 1
            for u in neighbors(sep, plan.rts[j])
                t = tpindex(top, u)

                if iszero(slot[t])
                    m += one(I)
                    slot[t] = m
                    Ubuf[uo + m] = u
                end
            end
        end

        Ulen[c] = m
        P = tpbuffer(B, Pbuf, Poff[c], m, nrhs, side)
        szerorec!(s, P, trans)

        hi = pointers(res)[fb + one(I)] - one(I)
        bnd = TPBoundary(hi, P, top, slot, sbuf)
        strsx_rng_tmp!(s, side, trans, diag, A, B, W, pool, 1, fa, fb, bnd)

        @inbounds for j in oneto(m)
            slot[tpindex(top, Ubuf[uo + j])] = zero(I)
        end
    end

    return
end
