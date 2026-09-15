# The k-shortest paths semiring. The quantale s must be selective:
#
#   a + b ∈ {a, b}
#
# for all elements a and b.
#
# References:
#
#   - Gondran and Minoux, Graphs, Dioids and Semirings
#     Chapter 8, Collected Examples of Monoids, (Pre)-Semirings and Dioids
#     Section 1.1.5
#
struct Best{N, S <: AbstractQuantale} <: AbstractSemiring
    s::S
end

function Best{N}(s::S) where {N, S <: AbstractQuantale}
    return Best{N, S}(s)
end

function slte(s::Best{N}, a, b) where {N}
    i = j = N

    @inbounds b1 = b[1]

    @inbounds while i > 0
        ai = a[i]
        bj = b[j]

        slte(s.s, ai, b1) && return true
        slte(s.s, ai, bj) || return false

        if slte(s.s, bj, ai)
            i -= 1
        end

        j -= 1
    end

    return true
end

#
#   (⊤, …, ⊤)
#
function stop(s::Best{N}, ::Type{NTuple{N, V}}) where {N, V}
    return beststop(s.s, Val(N), V)
end

function beststop(s::AbstractSemiring, ::Val{1}, ::Type{V}) where {V}
    return (stop(s, V),)
end

function beststop(s::AbstractSemiring, ::Val{N}, ::Type{V}) where {N, V}
    return (beststop(s, Val(N - 1), V)..., stop(s, V))
end

#
#   (0, …, 0)
#
function szero(s::Best{N}, ::Type{NTuple{N, V}}) where {N, V}
    return bestzero(s.s, Val(N), V)
end

function bestzero(s::AbstractSemiring, ::Val{1}, ::Type{V}) where {V}
    return (szero(s, V),)
end

function bestzero(s::AbstractSemiring, ::Val{N}, ::Type{V}) where {N, V}
    return (bestzero(s, Val(N - 1), V)..., szero(s, V))
end

#
#   (0, …, 0, 1)
#
function sone(s::Best{N}, ::Type{NTuple{N, V}}) where {N, V}
    return bestone(s.s, Val(N), V)
end

function bestone(s::AbstractSemiring, ::Val{1}, ::Type{V}) where {V}
    return (sone(s, V),)
end

function bestone(s::AbstractSemiring, ::Val{N}, ::Type{V}) where {N, V}
    return (szero(s, V), bestone(s, Val(N - 1), V)...)
end

#
# given sorted sequences
#
#   a = (a₁, …, aₙ)
#   b = (b₁, …, bₙ),
#
# return the n greatest elements of
# the merged sequence
#
#   (a₁, …, aₙ, b₁, …, bₙ)
#
function splus(s::Best, a, b)
    return bestplus(s, a, b)
end

function bestplus(s::Best{N}, a::NTuple{A}, b::NTuple{B}) where {N, A, B}
    if A + B == N
        c = ()
    else
        a0..., an = a
        b0..., bn = b

        if sgte(s.s, an, bn)
            c = (bestplus(s, a0, b)..., an)
        else
            c = (bestplus(s, a, b0)..., bn)
        end
    end

    return c
end

#
# given sorted sequences
#
#   a = (a₁, …, aₙ)
#   b = (b₁, …, bₙ),
#
# return the n greatest elements of
# the matrix
#
#   [ a₁ + b₁ ⋯ aₙ + b₁ ]
#   [    ⋮         ⋮    ]
#   [ aₙ + b₁ ⋯ aₙ + bₙ ]
#
function sprod(s::Best, a, b)
    return bestprod(s, a, b)
end

function bestprod(s::Best, a::NTuple{1}, b::NTuple{1})
    return (sprod(s.s, a[1], b[1]),)
end

function bestprod(s::Best, a::NTuple{1}, b)
    b0..., bn = b
    return (bestprod(s, a, b0)..., sprod(s.s, a[1], bn))
end

function bestprod(s::Best, a, b)
    a0..., an = a
    return bestplus(s, bestprod(s, a0, b), bestprod(s, (an,), b))
end

function sstar(s::Best{N}, a::NTuple{N, V}) where {N, V}
    u = sone(s, NTuple{N, V})
    b = u

    for _ in 1:N - 1
        b = smuladd(s, a, b, u)
    end

    return b
end
