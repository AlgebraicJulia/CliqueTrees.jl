# ===== saxpy_kern! =====

@inline function smul(s::AbstractSemiring, tA::Val, tB::Val, ::Val{:L}, v, x, c)
    return smuladd(s, x, v, c, tA, tB)
end

@inline function smul(s::AbstractSemiring, tA::Val, tB::Val, ::Val{:R}, v, x, c)
    return smuladd(s, v, x, c, tA, tB)
end

@generated function saxpy_kern!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, pc::Ptr{T}, pv::Ptr{T}, sv::Integer, x::Tuple, ni::Integer) where {T}
    N = fieldcount(x)
    W = vecwidth(T)
    Z = sizeof(T)

    if N == 1
        U = 4
    else
        U = 2
    end

    p(k) = Symbol(:p_, k)
    c(u) = Symbol(:c_, u)

    init = Expr(:block)

    for k in 1:N
        push!(init.args, :($(p(k)) = pv + $(k - 1) * sv * $Z))
    end

    function pass(m)
        ex = Expr(:block, :(o = (i - 1) * $Z))

        for u in 1:m
            push!(ex.args, :($(c(u)) = vload(Vec{$W, $T}, pc + o + $((u - 1) * W * Z))))
        end

        for k in 1:N, u in 1:m
            push!(ex.args, :($(c(u)) = smul(s, tA, tB, side, vload(Vec{$W, $T}, $(p(k)) + o + $((u - 1) * W * Z)), x[$k], $(c(u)))))
        end

        for u in 1:m
            push!(ex.args, :(vstore($(c(u)), pc + o + $((u - 1) * W * Z))))
        end

        return ex
    end

    tail = Expr(:block, :(e = unsafe_load(pc, i)))

    for k in 1:N
        push!(tail.args, :(e = smul(s, tA, tB, side, unsafe_load($(p(k)), i), x[$k], e)))
    end

    push!(tail.args, :(unsafe_store!(pc, e, i)))

    return quote
        $init
        i = 1

        while i + $(U * W - 1) <= ni
            $(pass(U))
            i += $(U * W)
        end

        while i + $(W - 1) <= ni
            $(pass(1))
            i += $W
        end

        while i <= ni
            $tail
            i += 1
        end

        return
    end
end

@inline function saxpy_kern!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, pc::Ptr{T}, pv::Ptr{T}, x, ni::Integer) where {T}
    return saxpy_kern!(s, tA, tB, side, pc, pv, 0, (x,), ni)
end

# ===== saxpy! =====

function saxpy!(s::AbstractSemiring, tA::Val, tB::Val, side::Val, a, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    @assert length(x) == length(y)

    @preserve x y saxpy_kern!(s, tA, tB, side, pointer(y), pointer(x), a, length(y))

    return y
end
