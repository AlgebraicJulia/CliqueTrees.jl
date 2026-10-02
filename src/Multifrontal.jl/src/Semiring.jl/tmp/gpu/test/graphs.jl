using SparseArrays, Random

function grid(nx, ny, ::Type{T}) where {T}
    id(i, j) = i + (j - 1) * nx
    I = Int[]; J = Int[]; V = T[]
    for j in 1:ny, i in 1:nx, (di, dj) in ((1, 0), (0, 1))
        i2, j2 = i + di, j + dj
        (i2 <= nx && j2 <= ny) || continue
        w = T(rand(1:100))
        append!(I, (id(i, j), id(i2, j2))); append!(J, (id(i2, j2), id(i, j))); append!(V, (w, w))
    end
    return sparse(I, J, V, nx * ny, nx * ny)
end

function grid3(nx, ::Type{T}) where {T}
    id(i, j, l) = i + (j - 1) * nx + (l - 1) * nx^2
    I = Int[]; J = Int[]; V = T[]
    for l in 1:nx, j in 1:nx, i in 1:nx, d in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        a, b, c = i + d[1], j + d[2], l + d[3]
        (a <= nx && b <= nx && c <= nx) || continue
        w = T(rand(1:100)); u = id(i, j, l); v = id(a, b, c)
        append!(I, (u, v)); append!(J, (v, u)); append!(V, (w, w))
    end
    return sparse(I, J, V, nx^3, nx^3)
end
