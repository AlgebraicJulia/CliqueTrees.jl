#   julia --project=. -t 16 cuda/experiments/jl_reg.jl grid3d-25 grid2d-150   (needs 2 n×n matrices on the GPU)
# Is the C++ "reg" L-sweep variant's gain a language effect? Write the same kernel in Julia
# (exact-nn dispatch, residual values in registers, separator loop unrolled 4×) and time the
# batched L levels (level-by-level schedule) for: Julia simple (src), Julia reg (this file),
# C++ port, C++ reg. Values do not matter for timing (no data-dependent control flow).
include(joinpath(@__DIR__, "..", "..", "bench", "bench_cuda_backend.jl"))
const SG = SemiringGPU; const SC = SemiringCUDA
using .SemiringGPU.Semiring: smuladd

@generated function down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, ::Val{NN}) where {NN}
    xs = [Symbol(:x, j) for j in 1:NN]
    loads = [:($(xs[j]) = C[t, Rp + $(j - 1)]) for j in 1:NN]
    upd = [:($(xs[j]) = smuladd(s, c, Lval[Lp + $(j - 1) * na + r - 1], $(xs[j]), Val(:N), trans)) for j in 1:NN]
    solve = Expr[]
    for j in NN:-1:1, k in (j + 1):NN
        push!(solve, :($(xs[j]) = smuladd(s, $(xs[k]), Dval[Dp + $(j - 1) * NN + $(k - 1)], $(xs[j]), Val(:N), trans)))
    end
    stores = [:(C[t, Rp + $(j - 1)] = $(xs[j])) for j in 1:NN]
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(loads...)
            r = 1
            while r + 3 <= na      # 4 separator loads in flight
                c1 = C[t, Stgt[Sp + r - 1]]; c2 = C[t, Stgt[Sp + r]]; c3 = C[t, Stgt[Sp + r + 1]]; c4 = C[t, Stgt[Sp + r + 2]]
                c = c1; $(upd...); r += 1
                c = c2; $(upd...); r += 1
                c = c3; $(upd...); r += 1
                c = c4; $(upd...); r += 1
            end
            while r <= na
                c = C[t, Stgt[Sp + r - 1]]
                $(upd...)
                r += 1
            end
            $(solve...)
            $(stores...)
        end
        return
    end
end

function downward_kernel_reg!(s, trans, C::AbstractMatrix{T}, order, off::Int, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    t = threadIdx().x + (blockIdx().y - 1) * blockDim().x
    t > size(C, 1) && return
    @inbounds begin
        f = order[off + blockIdx().x]
        Rp = Rptr[f]; nn = Rptr[f + 1] - Rp
        Sp = Sptr[f]; na = Sptr[f + 1] - Sp
        Dp = Dptr[f]; Lp = Lptr[f]
        if nn == 1
            down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(1))
        elseif nn == 2
            down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(2))
        elseif nn == 3
            down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(3))
        elseif nn == 4
            down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(4))
        elseif nn == 5
            down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(5))
        elseif nn == 6
            down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(6))
        elseif nn == 7
            down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(7))
        elseif nn == 8
            down_reg!(s, trans, C, t, Rp, Sp, na, Dp, Lp, Stgt, Dval, Lval, Val(8))
        else
            for j in 1:nn
                acc = C[t, Rp + j - 1]
                for r in 1:na
                    acc = smuladd(s, C[t, Stgt[Sp + r - 1]], Lval[Lp + (j - 1) * na + r - 1], acc, Val(:N), trans)
                end
                C[t, Rp + j - 1] = acc
            end
            for j in nn:-1:1
                acc = C[t, Rp + j - 1]
                for k in (j + 1):nn
                    acc = smuladd(s, C[t, Rp + k - 1], Dval[Dp + (j - 1) * nn + k - 1], acc, Val(:N), trans)
                end
                C[t, Rp + j - 1] = acc
            end
        end
    end
    return
end

# all batched L levels (small fronts only; the dense large fronts are left out: same for everyone)
function lsweep_jl(kf, G, D; tb = 64)
    s = G.s; nb = cld(size(D, 1), tb)
    kernel = @cuda launch = false kf(s, Val(:N), D, G.down, 0, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval)
    for l in 1:(length(G.downptr) - 1)
        strt = G.downptr[l]; nfl = G.downptr[l + 1] - strt
        nfl > 0 && kernel(s, Val(:N), D, G.down, strt - 1, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval; threads = tb, blocks = (nfl, nb))
    end
end
function lsweep_cu(G, D, variant; tb = 64)
    for l in 1:(length(G.downptr) - 1)
        strt = G.downptr[l]; nfl = G.downptr[l + 1] - strt
        nfl > 0 && SC.downward_batched_cuda!(G, D, G.down, strt - 1, nfl, tb, variant)
    end
end

for name in ARGS
    A = read_mtx(joinpath(MTX, name * ".mtx")); n = size(A, 1)
    F = ChordalSLU(MinPlus(), A); copyto!(F, A)
    P = FactorPlan(F; large = 256, graph = false, nstreams = 8); factorize!(P)
    G = GPUSLU(P; large = 8192)
    D = CuMatrix{Float32}(undef, n, n); M = CuMatrix{Float32}(undef, n, G.maxna)
    setschedule!(false); closure_gpu!(D, G; M); CUDA.synchronize()
    # same result check: one sweep from the same input with each kernel
    D0 = copy(D)
    lsweep_jl(SG.downward_kernel_simple!, G, D); h1 = fingerprint(D); copyto!(D, D0)
    lsweep_jl(downward_kernel_reg!, G, D); h2 = fingerprint(D); copyto!(D, D0)
    lsweep_cu(G, D, 1); h3 = fingerprint(D); copyto!(D, D0)
    D0 = nothing; GC.gc(); CUDA.reclaim()
    wait_gpu()
    fs = [() -> lsweep_jl(SG.downward_kernel_simple!, G, D), () -> lsweep_jl(downward_kernel_reg!, G, D), () -> lsweep_cu(G, D, 0), () -> lsweep_cu(G, D, 1)]
    t = map(fs) do f
        f(); CUDA.synchronize()
        minimum(@elapsed((f(); CUDA.synchronize())) for _ in 1:5)
    end
    @printf("%-10s batched L levels (ms): Julia simple %.1f | Julia reg (same kernel as C++ reg) %.1f | C++ port %.1f | C++ reg %.1f   [results equal: %s]\n",
        name, (1e3 .* t)..., h1 == h2 == h3)
    k = @cuda launch = false downward_kernel_reg!(G.s, Val(:N), D, G.down, 0, G.Rptr, G.Sptr, G.Stgt, G.Dptr, G.Lptr, G.LDval, G.LLval)
    println("   Julia reg kernel: ", CUDA.registers(k), " registers")
    global D = M = G = P = F = nothing; GC.gc(); CUDA.reclaim()
end
