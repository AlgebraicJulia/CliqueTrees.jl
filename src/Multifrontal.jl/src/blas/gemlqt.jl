# ===== gemlqt! =====
#
# gemlqt!(side, trans, V, Tm, C, work)
#     overwrite C with op(Q) C (side = L) or C op(Q) (side = R) for the Q from
#     gelqt! (reflectors in V, block factors in Tm); op is selected by trans
#     (N / T / C). Always applies via the blocked ?LARFB — unlike ormlq!,
#     which falls back to the unblocked level-2 path when length(tau) is small.
#     `work` is a resizable workspace, grown to the required size and reused
#     across calls.  (LAPACK ?GEMLQT.)
#
for (gemlqt, elty) in (
        (:dgemlqt_, :Float64),
        (:sgemlqt_, :Float32),
        (:zgemlqt_, :ComplexF64),
        (:cgemlqt_, :ComplexF32),
    )
    @eval begin
        function gemlqt!(side::AbstractChar, trans::AbstractChar, V::AbstractMatrix{$elty}, Tm::AbstractMatrix{$elty}, C::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            require_one_based_indexing(V, Tm, C, work)
            chkstride1(V, Tm, C)
            m = size(C, 1)
            n = size(C, 2)
            k = size(Tm, 2)
            mb = size(Tm, 1)
            @assert size(V, 1) >= k && size(V, 2) == (side == 'L' ? m : n)
            length(work) < mb * (side == 'L' ? n : m) && resize!(work, mb * (side == 'L' ? n : m))
            info = Ref{BlasInt}()

            ccall((BLAS.@blasfunc($gemlqt), BLAS.libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Clong, Clong),
                side, trans, m, n, k, mb,
                V, max(1, stride(V, 2)), Tm, max(1, stride(Tm, 2)),
                C, max(1, stride(C, 2)), work, info, 1, 1)

            LAPACK.chklapackerror(info[])
            return C
        end
    end
end

function gemlqt!(side::Val, trans::Val, V::AbstractMatrix{T}, Tm::AbstractMatrix{T}, C::AbstractMatrix{T}, work::AbstractVector{T}) where {T <: BlasFloat}
    return gemlqt!(char(side), char(trans), V, Tm, C, work)
end

function gemlqt!(side::Val, ::Val{:C}, V::AbstractMatrix{T}, Tm::AbstractMatrix{T}, C::AbstractMatrix{T}, work::AbstractVector{T}) where {T <: BlasReal}
    return gemlqt!(side, Val(:T), V, Tm, C, work)
end
