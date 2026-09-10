# ===== tpqrt! =====
#
# tpqrt!(A, B, Tm, work)
#     [A; B] = Q [R; 0] with A (n × n) upper triangular on entry, R on exit;
#     B (m × n) holds the Householder reflectors V on exit; Tm (nb × n) the
#     compact WY block factors.  (LAPACK ?TPQRT with l = 0: B is rectangular.)
#
for (tpqrt, elty) in (
        (:dtpqrt_, :Float64),
        (:stpqrt_, :Float32),
        (:ztpqrt_, :ComplexF64),
        (:ctpqrt_, :ComplexF32),
    )
    @eval begin
        function tpqrt!(A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}, Tm::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            require_one_based_indexing(A, B, Tm, work)
            chkstride1(A, B, Tm)
            n = size(A, 1)
            m = size(B, 1)
            nb = size(Tm, 1)
            @assert size(A, 2) == n && size(B, 2) == n && size(Tm, 2) >= n
            @assert 1 <= nb <= n && length(work) >= nb * n
            info = Ref{BlasInt}()

            ccall((BLAS.@blasfunc($tpqrt), BLAS.libblastrampoline), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                m, n, 0, nb,
                A, max(1, stride(A, 2)), B, max(1, stride(B, 2)),
                Tm, max(1, stride(Tm, 2)), work, info)

            LAPACK.chklapackerror(info[])
            return A, B, Tm
        end
    end
end
