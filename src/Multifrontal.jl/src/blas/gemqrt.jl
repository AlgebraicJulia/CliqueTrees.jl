# ===== gemqrt! =====
#
# gemqrt!(side, trans, V, Tm, C, work)
#     overwrite C with op(Q) C (side = L) or C op(Q) (side = R) for the Q from
#     geqrt! (reflectors in V, block factors in Tm); op is selected by trans
#     (N / T / C). Always applies via the blocked ?LARFB — unlike ormqr!,
#     which falls back to the unblocked level-2 path when length(tau) is small.
#     `work` is a resizable workspace, grown to the required size and reused
#     across calls.  (LAPACK ?GEMQRT.)
#
for (gemqrt, elty) in (
        (:dgemqrt_, :Float64),
        (:sgemqrt_, :Float32),
        (:zgemqrt_, :ComplexF64),
        (:cgemqrt_, :ComplexF32),
    )
    @eval begin
        function gemqrt!(side::AbstractChar, trans::AbstractChar, V::AbstractMatrix{$elty}, Tm::AbstractMatrix{$elty}, C::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            require_one_based_indexing(V, Tm, C, work)
            chkstride1(V, Tm, C)
            m = size(C, 1)
            n = size(C, 2)
            k = size(Tm, 2)
            nb = size(Tm, 1)
            @assert size(V, 2) >= k && size(V, 1) == (side == 'L' ? m : n)
            length(work) < nb * (side == 'L' ? n : m) && resize!(work, nb * (side == 'L' ? n : m))
            info = Ref{BlasInt}()

            ccall((BLAS.@blasfunc($gemqrt), BLAS.libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Clong, Clong),
                side, trans, m, n, k, nb,
                V, max(1, stride(V, 2)), Tm, max(1, stride(Tm, 2)),
                C, max(1, stride(C, 2)), work, info, 1, 1)

            LAPACK.chklapackerror(info[])
            return C
        end
    end
end

function gemqrt!(side::Val, trans::Val, V::AbstractMatrix{T}, Tm::AbstractMatrix{T}, C::AbstractMatrix{T}, work::AbstractVector{T}) where {T <: BlasFloat}
    return gemqrt!(char(side), char(trans), V, Tm, C, work)
end

function gemqrt!(side::Val, ::Val{:C}, V::AbstractMatrix{T}, Tm::AbstractMatrix{T}, C::AbstractMatrix{T}, work::AbstractVector{T}) where {T <: BlasReal}
    return gemqrt!(side, Val(:T), V, Tm, C, work)
end
