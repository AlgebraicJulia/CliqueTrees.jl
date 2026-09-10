# ===== gelqt! =====
#
# gelqt!(A, Tm, work)
#     A = L Q by blocked Householder with block size mb = size(Tm, 1). On exit
#     the lower trapezoid of A holds L, its strict upper part the reflectors,
#     and Tm (mb × min(m, n)) the compact WY block factors — unlike gelqf!,
#     which keeps only the scalar factors tau. `work` is a resizable workspace
#     (a Vector), grown to the required mb * max(m, n) and reused across calls.
#     (LAPACK ?GELQT.)
#
for (gelqt, elty) in (
        (:dgelqt_, :Float64),
        (:sgelqt_, :Float32),
        (:zgelqt_, :ComplexF64),
        (:cgelqt_, :ComplexF32),
    )
    @eval begin
        function gelqt!(A::AbstractMatrix{$elty}, Tm::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            require_one_based_indexing(A, Tm, work)
            chkstride1(A, Tm)
            m = size(A, 1)
            n = size(A, 2)
            mb = size(Tm, 1)
            @assert size(Tm, 2) >= min(m, n) && 1 <= mb <= max(1, min(m, n))
            length(work) < mb * max(m, n) && resize!(work, mb * max(m, n))
            info = Ref{BlasInt}()

            ccall((BLAS.@blasfunc($gelqt), BLAS.libblastrampoline), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}),
                m, n, mb,
                A, max(1, stride(A, 2)), Tm, max(1, stride(Tm, 2)),
                work, info)

            LAPACK.chklapackerror(info[])
            return A, Tm
        end
    end
end
