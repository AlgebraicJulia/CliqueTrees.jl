struct Pred{S <: TropicalSemiring} <: AbstractQuantale
    s::S
end

struct Succ{S <: TropicalSemiring} <: AbstractQuantale
    s::S
end

const PredSucc = Union{Pred, Succ}

function slte(s::PredSucc, (av, ai, ah), (bv, bi, bh))
    return slte(s.s, av, bv) && (!slte(s.s, bv, av) || ah > bh || (ah == bh && ai >= bi))
end

function stop(s::PredSucc, ::Type{Tuple{V, I, H}}) where {V, I, H}
    return (stop(s.s, V), zero(I), zero(H))
end

function szero(s::PredSucc, ::Type{Tuple{V, I, H}}) where {V, I, H}
    return (szero(s.s, V), zero(I), zero(H))
end

function sone(s::PredSucc, ::Type{Tuple{V, I, H}}) where {V, I, H}
    return (sone(s.s, V), zero(I), zero(H))
end

function splus(s::PredSucc, (av, ai, ah), (bv, bi, bh))
    if !slte(s.s, av, bv) || (slte(s.s, bv, av) && (ah < bh || (ah == bh && ai <= bi)))
        cv, ci, ch = av, ai, ah
    else
        cv, ci, ch = bv, bi, bh
    end

    return (cv, ci, ch)
end

function sprod(s::PredSucc, (av, ai, ah), (bv, bi, bh))
    V = promote_eltype(av, bv)
    I = promote_eltype(ai, bi)
    H = promote_eltype(ah, bh)

    uv = szero(s.s, V)
    cv = sprod(s.s, av, bv)

    if av == uv || bv == uv || cv == uv
        cv = uv
        ci = zero(I)
        ch = zero(H)
    else
        if s isa Pred
            xi = convert(I, ai)
            yi = convert(I, bi)
        else
            xi = convert(I, bi)
            yi = convert(I, ai)
        end

        if yi < one(I)
            ci = xi
        else
            ci = yi
        end

        ch = ah + bh
    end

    return (cv, ci, ch)
end

function sstar(s::PredSucc, (av, ai, ah)::Tuple{V, I, H}) where {V, I, H}
    return (sstar(s.s, av), zero(I), zero(H))
end


