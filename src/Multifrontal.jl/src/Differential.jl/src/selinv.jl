function selinv(A::SparseMatrixCSC, F::ChordalCholesky)
    return selinv(Hermitian(A), F)
end

function selinv(A::HermOrSymSparse, F::ChordalCholesky)
    Y = copy(F)
    return project(A, selinv!(Y))
end

# ===== frule =====

function selinv_frule_impl(A::HermOrSymSparse, F::ChordalCholesky, dA::HermOrSymSparse)
     cB = selinv!(copy(F))
      B = project(A, cB)
    cdA = copyto!(similar(F), dA)
     dB = project(A, fisher!(cdA, F, cB; inv=false))
    rmul!(parent(dB), -1)
    return B, dB
end

# ===== rrule =====

function selinv_rrule_impl!(ΣA::HermOrSymSparse, A::HermOrSymSparse, F::ChordalCholesky, B::HermOrSymSparse, ΔB::HermOrSymSparse)
     cB = selinv!(copy(F))
    cΔB = copyto!(similar(F), ΔB)
     dA = scldia!(project(A, fisher!(scldia!(cΔB, 2), F, cB; inv=false)), 1 / 2)
    selaxpy!(-1, dA, ΣA)
    return ΣA
end

