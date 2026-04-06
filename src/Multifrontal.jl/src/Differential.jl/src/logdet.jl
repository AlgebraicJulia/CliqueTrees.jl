function LinearAlgebra.logdet(A::SparseMatrixCSC, F::ChordalCholesky)
    return logdet(Hermitian(A), F)
end

function LinearAlgebra.logdet(A::HermOrSymSparse, F::ChordalCholesky)
    return logdet(F)
end

# ===== frule =====

function logdet_frule_impl(A::HermOrSymSparse, F::ChordalCholesky, dA::HermOrSymSparse)
     y = logdet(A, F)
    dy = symdot(selinv(A, F), dA)
    return y, dy
end

function logdet_rrule_frule_impl!(ΣA::HermOrSymSparse, dΣA::HermOrSymSparse, A::HermOrSymSparse, dA::HermOrSymSparse, F::ChordalCholesky, Δy::Number, dΔy::Number)
    if !iszero(Δy)
        B, dB = selinv_frule_impl(A, F, dA)
        scldia!(B, 1 / 2)
        scldia!(dB, 1 / 2)
        selaxpy!(2Δy,  B,  ΣA)
        selaxpy!(2dΔy, B, dΣA)
        selaxpy!(2Δy, dB, dΣA)
    elseif !iszero(dΔy)
        B = selinv(A, F)
        scldia!(B, 1 / 2)
        selaxpy!(2dΔy, B, dΣA)
    end

    return ΣA, dΣA
end

# ===== rrule =====

Base.@noinline function logdet_rrule_impl!(ΣA::HermOrSymSparse, A::HermOrSymSparse, F::ChordalCholesky, y::Number, Δy::Number)
    if !iszero(Δy)
        B = selinv(A, F)
        scldia!(B, 1/2)
        selaxpy!(2Δy, B, ΣA)
    end

    return ΣA
end

function logdet_rrule_rrule_impl!(ΣA::HermOrSymSparse, A::HermOrSymSparse, F::ChordalCholesky, Δy::Number, ΔΣA::HermOrSymSparse)
    B = selinv(A, F)

    if !iszero(Δy)
         cB = copyto!(similar(F),   B)
        cΔB = copyto!(similar(F), ΔΣA)
         dA = scldia!(project(A, fisher!(cΔB, F, cB; inv=false)), 1 / 2)
        selaxpy!(-2Δy, dA, ΣA)
    end

    scldia!(B, 1 / 2)
    ΣΔy = 2dot(parent(B), parent(ΔΣA))
    return ΣA, ΣΔy
end
