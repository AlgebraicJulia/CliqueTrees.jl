# Export the benchmark graphs (same weights as our benchmarks) as Matrix Market, for external codes.
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
        println(io, "% arc i -> j with weight A[i, j]; both directions listed; 1-based")
        println(io, size(A, 1), " ", size(A, 2), " ", length(v))
        for k in eachindex(v)
            println(io, i[k], " ", j[k], " ", Int(v[k]))
        end
    end
end

for (name, A) in ["grid2d-150" => grid(150, 150), "grid2d-180" => grid(180, 180), "grid3d-25" => grid3(25), "grid3d-30" => grid3(30)]
    write_mtx(joinpath(DATA, "mtx", name * ".mtx"), A)
    println(name, ": n=", size(A, 1), " nnz=", nnz(A))
end
