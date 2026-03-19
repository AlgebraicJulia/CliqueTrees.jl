module KrylovExt

using CliqueTrees
using CliqueTrees.Multifrontal
using Krylov
using LinearAlgebra

function CliqueTrees.Multifrontal.cgworkspace(::Type{T}, n::Integer) where {T}
    return Krylov.CgWorkspace(n, n, CliqueTrees.FVector{T})
end

function CliqueTrees.Multifrontal.cgsolution(workspace::Krylov.CgWorkspace)
    return workspace.x
end

function CliqueTrees.Multifrontal.cg!(
        workspace::Krylov.CgWorkspace{T},
        H,
        b::AbstractVector{T},
        prec::AbstractVector{T},
        rtol::Real,
        itmax::Int,
    ) where {T}
    M = Diagonal(prec)
    Krylov.cg!(workspace, H, b; M, rtol, itmax, ldiv=true)
    return workspace
end

end

