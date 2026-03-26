import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted, broadcasted, combine_eltypes

struct ChordalStyle <: AbstractArrayStyle{2} end

function ChordalStyle(::Val{2})
    return ChordalStyle()
end

function ChordalStyle(::Val{N}) where {N}
    return DefaultArrayStyle{N}()
end

function BroadcastStyle(::Type{<:ChordalTriangular})
    return ChordalStyle()
end

function BroadcastStyle(::ChordalStyle, ::ChordalStyle)
    return ChordalStyle()
end

function findtri(bc::Broadcasted)
    return findtri(bc.args)
end

function findtri(args::Tuple)
    return findtri(args[1], Base.tail(args))
end

function findtri(::Tuple{})
    return
end

function findtri(A::ChordalTriangular, ::Tuple)
    return A
end

function findtri(::Any, rest::Tuple)
    return findtri(rest)
end

function fzero(x::Number)
    return Some(x)
end

function fzero(::Type{T}) where {T}
    return Some(T)
end

function fzero(r::Ref)
    return Some(r[])
end

function fzero(A::ChordalTriangular{<:Any, <:Any, T}) where {T}
    return Some(zero(T))
end

function fzero(x)
    return
end

function fzero(bc::Broadcasted)
    args = map(fzero, bc.args)

    if all(!isnothing, args)
        return Some(bc.f(map(something, args)...))
    end

    return
end

function fzeropreserving(bc::Broadcasted)
    v = fzero(bc)
    return !isnothing(v) && iszero(something(v))
end

function checkzeropreserving(bc::Broadcasted{ChordalStyle})
    @assert fzeropreserving(bc)
end

function checknonunit(A::ChordalTriangular{DIAG}) where {DIAG}
    @assert DIAG === :N
end

function checknonunit(x)
    return
end

function checknonunit(bc::Broadcasted)
    for arg in bc.args
        checknonunit(arg)
    end

    return
end

function Base.similar(bc::Broadcasted{ChordalStyle}, ::Type{T}) where {T}
    return similar(findtri(bc), T)
end

function getdiag(A::ChordalTriangular)
    return A.Dval
end

function getdiag(x)
    return x
end

function getdiag(bc::Broadcasted)
    return broadcasted(bc.f, map(getdiag, bc.args)...)
end

function getoffd(A::ChordalTriangular)
    return A.Lval
end

function getoffd(x)
    return x
end

function getoffd(bc::Broadcasted)
    return broadcasted(bc.f, map(getoffd, bc.args)...)
end

function Base.copyto!(dest::ChordalTriangular, bc::Broadcasted{ChordalStyle})
    checkzeropreserving(bc)
    checknonunit(bc)
    checknonunit(dest)
    copyto!(dest.Dval, getdiag(bc))
    copyto!(dest.Lval, getoffd(bc))
    return dest
end

function Base.copy(bc::Broadcasted{ChordalStyle})
    dest = similar(bc, combine_eltypes(bc.f, bc.args))
    return copyto!(dest, bc)
end
