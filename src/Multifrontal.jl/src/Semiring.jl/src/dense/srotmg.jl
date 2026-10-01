#
# Given a matrix
#
#   N = [ p a ] ,  β = q* ,
#       [ b q ]
#
# compute the LU and UL factorizations 
#
#   N* = U₁* L₁* =  [ p  a  ]* [     ]*
#                   [    q' ]  [ σ   ]
#
#      = L₂* U₂* =  [ p'    ]* [   δ ]*
#                   [ b  q  ]  [     ]
#
# and return p', γ, δ, σ, β', where
#
#   β' = (U₁*)₂₂      γ = (L₂*)₂₁
#
@inline function srotmg(s::AbstractSemiring, p::T, β::T, a::T, b::T) where {T}
    δ  = sprod(s, a, β, Val(:N), Val(:N))
    p2 = smuladd(s, δ, b, p, Val(:N), Val(:N))

    if isintegral(s)
        γ  = sprod(s, β, b, Val(:N), Val(:N))
        σ  = b
        β2 = β
    else
        γ  = sprod(s, sprod(s, β, b, Val(:N), Val(:N)), sstar(s, p2), Val(:N), Val(:N))
        σ  = sprod(s, b, sstar(s, p), Val(:N), Val(:N))
        β2 = sprod(s, β, sstar(s, sprod(s, σ, δ, Val(:N), Val(:N))), Val(:N), Val(:N))
    end

    return p2, γ, δ, σ, β2
end
