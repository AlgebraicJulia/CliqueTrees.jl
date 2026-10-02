# Larger grids for the big GPUs, same generators/weights as export_mtx.jl:
#   julia --project=. bench/export_more.jl grid2d-250 grid3d-40 ...
include(joinpath(@__DIR__, "bench_solve.jl"))

function grid3(nx)
    rng = Xoshiro(1); id(i, j, l) = i + (j - 1) * nx + (l - 1) * nx^2
    I = Int[]; J = Int[]; V = T[]
    for l in 1:nx, j in 1:nx, i in 1:nx, d in ((1,0,0),(0,1,0),(0,0,1))
        a, b, c = i + d[1], j + d[2], l + d[3]
        (a <= nx && b <= nx && c <= nx) || continue
        w = T(rand(rng, 1:100)); u = id(i, j, l); v = id(a, b, c)
        append!(I, (u, v)); append!(J, (v, u)); append!(V, (w, w))
    end
    return sparse(I, J, V, nx^3, nx^3)
end

function write_mtx(path, A)
    i, j, v = findnz(A)
    open(path, "w") do io
        println(io, "%%MatrixMarket matrix coordinate real general")
        println(io, size(A, 1), " ", size(A, 2), " ", length(v))
        for k in eachindex(v); println(io, i[k], " ", j[k], " ", Int(v[k])); end
    end
end

for name in ARGS
    kind, s = split(name, '-')
    A = kind == "grid2d" ? grid(parse(Int, s), parse(Int, s)) : grid3(parse(Int, s))
    write_mtx(joinpath(DATA, "mtx", name * ".mtx"), A)
    println(name, ": n=", size(A, 1))
end
