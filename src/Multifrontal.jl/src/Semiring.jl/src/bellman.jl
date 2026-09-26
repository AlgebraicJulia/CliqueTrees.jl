#
# Find the least solution to the
# fixpoint equation
#
#   Ax + b = x.
#
function bellman!(f, s::AbstractSemiring, side::Val, trans::Val, A::AbstractMatrix, b::AbstractVecOrMat; itmax::Integer = size(A, 1))
    if isidempotent(s)
        bellman_i!(f, s, side, trans, A, similar(b), b; itmax)
    else
        bellman!(f, s, side, trans, A, similar(b), similar(b), b; itmax)
    end

    return b
end

function bellman!(f, s::AbstractSemiring, side::Val{SIDE}, trans::Val, A::AbstractMatrix, u::AbstractVecOrMat, v::AbstractVecOrMat, b::AbstractVecOrMat; itmax::Integer = size(A, 1)) where {SIDE}
    i = 1
    #
    #   v ← b
    #
    copyto!(v, b)
    #
    #   u ← 0
    #
    szerorec!(s, u, trans)

    while !f(u, b) && i <= itmax
        i += 1
        #
        #   u ← b
        #
        copyto!(u, b)
        #
        #   b ← A u ⊕ v
        #
        copyto!(b, v)

        if SIDE === :L
            sgemx!(s, trans, Val(:N), b, A, u)
        else
            sgemx!(s, Val(:N), trans, b, u, A)
        end
    end

    return b
end

function bellman_i!(f, s::AbstractSemiring, side::Val{SIDE}, trans::Val, A::AbstractMatrix, u::AbstractVecOrMat, b::AbstractVecOrMat; itmax::Integer = size(A, 1)) where {SIDE}
    i = 1
    #
    #   u ← 0
    #
    szerorec!(s, u, trans)

    while !f(u, b) && i <= itmax
        i += 1
        #
        #   u ← b
        #
        copyto!(u, b)
        #
        #   b ← A u ⊕ b
        #
        if SIDE === :L
            sgemx!(s, trans, Val(:N), b, A, u)
        else
            sgemx!(s, Val(:N), trans, b, u, A)
        end
    end

    return b
end
