# Real graphs for the sweep, undirected, uniform weights in [1, 100] (fixed seed), written to data/mtx
# (both arc directions, as export_mtx.jl):
#   julia --project=. bench/export_real.jl ca-HepTh ca-AstroPh ca-CondMat email-Enron delaunay_n16 ...
include(joinpath(@__DIR__, "bench_solve.jl"))

const SNAP = Dict("ca-HepTh" => "ca-HepTh.txt.gz", "ca-AstroPh" => "ca-AstroPh.txt.gz", "ca-CondMat" => "ca-CondMat.txt.gz",
                  "ca-GrQc" => "ca-GrQc.txt.gz", "email-Enron" => "email-Enron.txt.gz",
                  "com-Amazon" => "com-amazon.ungraph.txt.gz", "com-DBLP" => "com-dblp.ungraph.txt.gz")

fetch(url, path) = (isfile(path) || Base.run(`curl -sSfL --max-time 900 -o $path $url`); path)

function edges_snap(path)
    I = Int[]; J = Int[]
    for line in eachline(GzipDecompressorStream(open(path)))
        startswith(line, '#') && continue
        a, b = split(line)
        push!(I, parse(Int, a)); push!(J, parse(Int, b))
    end
    ids = unique!(sort!(vcat(I, J)))                    # compact the vertex ids
    pos = Dict(v => k for (k, v) in enumerate(ids))
    return [pos[i] for i in I], [pos[j] for j in J], length(ids)
end

function edges_mtx(path)                                # pattern symmetric (SuiteSparse)
    I = Int[]; J = Int[]; n = 0; header = true
    for line in eachline(path)
        startswith(line, '%') && continue
        a = split(line)
        if header; n = parse(Int, a[1]); header = false; continue; end
        push!(I, parse(Int, a[1])); push!(J, parse(Int, a[2]))
    end
    return I, J, n
end

function weighted(I, J, n)
    E = triu(sparse(min.(I, J), max.(I, J), true, n, n), 1)
    i, j, _ = findnz(E)
    w = T.(rand(Xoshiro(1), 1:100, length(i)))
    return sparse(vcat(i, j), vcat(j, i), vcat(w, w), n, n, min)
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
    out = joinpath(DATA, "mtx", name * ".mtx")
    isfile(out) && continue
    if haskey(SNAP, name)
        I, J, n = edges_snap(fetch("https://snap.stanford.edu/data/" * SNAP[name], joinpath(DATA, SNAP[name])))
    else                                                # DIMACS10 meshes from the SuiteSparse collection
        tgz = fetch("https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/$name.tar.gz", joinpath(DATA, "$name.tar.gz"))
        isdir(joinpath(DATA, name)) || Base.run(`tar -xzf $tgz -C $DATA`)
        I, J, n = edges_mtx(joinpath(DATA, name, name * ".mtx"))
    end
    write_mtx(out, weighted(I, J, n))
    println(name, ": n=", n)
end
