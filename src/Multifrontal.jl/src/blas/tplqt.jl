# ===== tplqt! =====
#
# tplqt!(A, B, Tm, work)
#     [A B] = [L 0] Q with A (m × m) lower triangular on entry, L on exit;
#     B (m × n) holds the Householder reflectors V on exit; Tm (mb × m) the
#     compact WY block factors.  (LAPACK ?TPLQT with l = 0: B is rectangular.)
#
for (tplqt, elty) in (
        (:dtplqt_, :Float64),
        (:stplqt_, :Float32),
        (:ztplqt_, :ComplexF64),
        (:ctplqt_, :ComplexF32),
    )
    @eval begin
        function tplqt!(A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}, Tm::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            require_one_based_indexing(A, B, Tm, work)
            chkstride1(A, B, Tm)
            m = size(A, 1)
            n = size(B, 2)
            mb = size(Tm, 1)
            @assert size(A, 2) == m && size(B, 1) == m && size(Tm, 2) >= m
            @assert 1 <= mb <= m && length(work) >= mb * m
            info = Ref{BlasInt}()

            ccall((BLAS.@blasfunc($tplqt), BLAS.libblastrampoline), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                m, n, 0, mb,
                A, max(1, stride(A, 2)), B, max(1, stride(B, 2)),
                Tm, max(1, stride(Tm, 2)), work, info)

            LAPACK.chklapackerror(info[])
            return A, B, Tm
        end
    end
end
