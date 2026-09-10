# ===== geqrt! =====
#
# geqrt!(A, Tm, work)
#     A = Q R by blocked Householder with block size nb = size(Tm, 1). On exit
#     the upper trapezoid of A holds R, its strict lower part the reflectors,
#     and Tm (nb × min(m, n)) the compact WY block factors — unlike geqrf!,
#     which keeps only the scalar factors tau. `work` is a resizable workspace
#     (a Vector), grown to the required nb * max(m, n) and reused across calls.
#     (LAPACK ?GEQRT.)
#
for (geqrt, elty) in (
        (:dgeqrt_, :Float64),
        (:sgeqrt_, :Float32),
        (:zgeqrt_, :ComplexF64),
        (:cgeqrt_, :ComplexF32),
    )
    @eval begin
        function geqrt!(A::AbstractMatrix{$elty}, Tm::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            require_one_based_indexing(A, Tm, work)
            chkstride1(A, Tm)
            m = size(A, 1)
            n = size(A, 2)
            nb = size(Tm, 1)
            @assert size(Tm, 2) >= min(m, n) && 1 <= nb <= max(1, min(m, n))
            length(work) < nb * max(m, n) && resize!(work, nb * max(m, n))
            info = Ref{BlasInt}()

            ccall((BLAS.@blasfunc($geqrt), BLAS.libblastrampoline), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}),
                m, n, nb,
                A, max(1, stride(A, 2)), Tm, max(1, stride(Tm, 2)),
                work, info)

            LAPACK.chklapackerror(info[])
            return A, Tm
        end
    end
end
