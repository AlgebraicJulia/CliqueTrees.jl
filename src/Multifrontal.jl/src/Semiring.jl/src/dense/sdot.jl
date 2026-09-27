# ===== sdot_kern! =====

@generated function sdot_kern!(s::AbstractSemiring, tA::Val, tB::Val, op::Val, side::Val, pv::Ptr{T}, sv::Integer, px::Ptr{T}, nj::Integer, ::Val{N}) where {T, N}
    W = vecwidth(T)
    Z = sizeof(T)

    if N == 1
        A = 4
    else
        A = SGEMX_MV
    end

    p(k) = Symbol(:p_, k)
    d(k, a) = Symbol(:d_, k, :_, a)
    x(a) = Symbol(:x_, a)
    r(k) = Symbol(:r_, k)

    init = Expr(:block, :(z = szero(s, $T, op)))

    for k in 1:N
        push!(init.args, :($(p(k)) = pv + $(k - 1) * sv * $Z))

        for a in 1:A
            push!(init.args, :($(d(k, a)) = Vec{$W, $T}(z)))
        end
    end

    wide = Expr(:block, :(o = (j - 1) * $Z))

    for a in 1:A
        push!(wide.args, :($(x(a)) = vload(Vec{$W, $T}, px + o + $((a - 1) * W * Z))))
    end

    for k in 1:N, a in 1:A
        push!(wide.args, :($(d(k, a)) = smul(s, tA, tB, side, vload(Vec{$W, $T}, $(p(k)) + o + $((a - 1) * W * Z)), $(x(a)), $(d(k, a)))))
    end

    comb = Expr(:block)
    step = 1

    while step < A
        for a in 1:2step:A - step, k in 1:N
            push!(comb.args, :($(d(k, a)) = splus(s, $(d(k, a)), $(d(k, a + step)), op)))
        end

        step *= 2
    end

    narrow = Expr(:block, :(o = (j - 1) * $Z), :(xv = vload(Vec{$W, $T}, px + o)))

    for k in 1:N
        push!(narrow.args, :($(d(k, 1)) = smul(s, tA, tB, side, vload(Vec{$W, $T}, $(p(k)) + o), xv, $(d(k, 1)))))
    end

    lanes = Expr(:block)

    for k in 1:N
        push!(lanes.args, :($(r(k)) = sreduce(s, $(d(k, 1)), op)))
    end

    tail = Expr(:block, :(xs = unsafe_load(px, j)))

    for k in 1:N
        push!(tail.args, :($(r(k)) = smul(s, tA, tB, side, unsafe_load($(p(k)), j), xs, $(r(k)))))
    end

    return quote
        $init
        j = 1

        while j + $(A * W - 1) <= nj
            $wide
            j += $(A * W)
        end

        $comb

        while j + $(W - 1) <= nj
            $narrow
            j += $W
        end

        $lanes

        while j <= nj
            $tail
            j += 1
        end

        return ($(map(r, 1:N)...),)
    end
end

@inline function sdot_kern!(s::AbstractSemiring, tA::Val, tB::Val, op::Val, pa::Ptr{T}, pb::Ptr{T}, nj::Integer) where {T}
    return sdot_kern!(s, tA, tB, op, Val(:R), pa, 0, pb, nj, Val(1))[1]
end

# ===== sreduce =====

@generated function sreduce(s::AbstractSemiring, d::Vec{W, T}, op::Val) where {W, T}
    @assert ispow2(W)

    body = Expr(:block, :(v0 = d))
    w = W
    k = 0

    while w > 2
        h = w ÷ 2
        lo = Tuple(0:h - 1)
        hi = Tuple(h:w - 1)
        v = Symbol(:v, k)
        push!(body.args, :($(Symbol(:v, k + 1)) = splus(s, shufflevector($v, Val($lo)), shufflevector($v, Val($hi)), op)))
        w = h
        k += 1
    end

    v = Symbol(:v, k)

    if w == 2
        push!(body.args, :(splus(s, $v[1], $v[2], op)))
    else
        push!(body.args, :($v[1]))
    end

    return quote
        $(Expr(:meta, :inline))
        $body
    end
end

# ===== sdot =====

function sdot(s::AbstractSemiring, tA::Val, tB::Val, x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    @assert length(x) == length(y)

    op = compose(tA, tB)

    return @preserve x y sdot_kern!(s, tA, tB, op, pointer(x), pointer(y), length(x))
end
