module KrylovExt

using CliqueTrees
using CliqueTrees.Multifrontal
using Krylov

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
        rtol::Real,
        itmax::Int,
    ) where {T}
    Krylov.cg!(workspace, H, b; rtol, itmax)
    return workspace
end

end

