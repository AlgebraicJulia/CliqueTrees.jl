# Parallel/serial ordering baselines on the main_table graph set (CPU, unit weights, fill metrics).
# usage: julia --project=ENV pbench.jl OUT METHODS THREADS name...
#   METHODS: comma list of AMD, METIS (serial, in-process), PARAMD, PARMETIS (external, per thread count)
#   THREADS: comma list (used by PARAMD/PARMETIS only; serial methods are recorded with threads=1)
# appends: host graph n m method threads nnzL flops tw time_s status
# metrics identical to rport/sbench.jl fillq (cliquetree supernodes: nnzL incl. diagonal, flops, tw=max col count-1).
using CliqueTrees, Graphs, SparseArrays, Serialization, Random, AMD, Metis
include(joinpath(@__DIR__, "gens.jl"))
const DIRS = split(get(ENV, "DATA", ""), ":")
const PARAMD = get(ENV, "PARAMD", "")
const PND = get(ENV, "PND", "")
const TMPD = get(ENV, "WORKDIR", mktempdir())
const CAP = get(ENV, "CAPS", "1500")
if haskey(ENV, "CORPUS")
    include(ENV["CORPUS"] * "/load.jl"); @eval using .TNCorpus
end
function loadg(name)
    startswith(name, "qaoaamp") && return TNCorpus.load(name).graph
    for d in DIRS
        f = joinpath(d, name * ".bin")
        if isfile(f)
            n, I, J = deserialize(f)
            return SimpleGraph(sparse(vcat(I, J), vcat(J, I), true, n, n, |))
        end
    end
    startswith(name, "lat2_") && return lat2(parse(Int, name[6:end]))
    startswith(name, "lat3_") && return lat3(parse(Int, name[6:end]))
    error("no data for $name")
end
function fillq(g, o)
    n = nv(g)
    @assert length(o) == n && isperm(o) "invalid permutation"
    _, tree = cliquetree(g; alg = o)
    nz = 0; fl = 0.0; mx = 0
    for bag in tree
        r = length(residual(bag)); s = length(separator(bag))
        nz += r * s + (r * (r + 1)) >> 1
        fl += r * Float64(s)^2 + Float64(s) * r * (r + 1) + r * (r + 1) * (2r + 1) / 6
        mx = max(mx, r + s)
    end
    return nz, fl, mx - 1
end
function writemtx(path, A)   # lower triangle, pattern symmetric (as paramd_cpu.jl)
    n = size(A, 1); cp = A.colptr; rv = A.rowval
    open(path, "w") do io
        println(io, "%%MatrixMarket matrix coordinate pattern symmetric")
        println(io, n, " ", n, " ", length(rv) ÷ 2)
        buf = IOBuffer()
        for j in 1:n, p in cp[j]:cp[j+1]-1
            i = rv[p]; i > j || continue
            println(buf, i, " ", j)
            buf.size > 1 << 22 && (write(io, take!(buf)))
        end
        write(io, take!(buf))
    end
end
function writecsr(path, A)
    n = size(A, 1)
    open(path, "w") do io
        write(io, Int32(n), Int32(length(A.rowval)))
        write(io, Int32.(A.colptr .- 1)); write(io, Int32.(A.rowval .- 1))
    end
end
ptime(txt, rx) = parse(Float64, match(rx, txt)[1])
function run_ext(cmd, tmo)
    out = IOBuffer()
    p = run(pipeline(ignorestatus(`timeout -k 20 $tmo $cmd`), stdout = out, stderr = out); wait = true)
    return p.exitcode, String(take!(out))
end
function run_paramd(path, nt, reps, tmo)
    best = Inf; o = nothing
    for rep in 1:reps
        rm(path * ".perm"; force = true)
        cmd = `$PARAMD --matrix $path --algo paramd --mult 1.1 --lim 8192 --mem 3.0 --seed 1`
        rc, txt = withenv("OMP_NUM_THREADS" => string(nt), "OMP_PROC_BIND" => "close", "OMP_PLACES" => "cores") do
            run_ext(cmd, tmo)
        end
        rc == 0 || return nothing, rc in (124, 137) ? "timeout" : "rc$rc"
        occursin("Ordering Time", txt) || return nothing, "crash"
        t = ptime(txt, r"Ordering Time = ([0-9.eE+-]+)")
        if t < best; best = t; o = parse.(Int, split(read(path * ".perm", String))) .+ 1; end
        best > tmo / 3 && break
    end
    return (o, best), "ok"
end
function run_parmetis(path, ord, P, n, reps, tmo)
    best = Inf; o = nothing
    for rep in 1:reps
        rm(ord; force = true)
        cmd = `mpirun --bind-to none --oversubscribe -n $P $PND $path $ord`
        rc, txt = run_ext(cmd, tmo)
        rc == 0 || return nothing, rc in (124, 137) ? "timeout" : "rc$rc"
        occursin("PNDTIME", txt) || return nothing, "crash"
        t = ptime(txt, r"PNDTIME ([0-9.eE+-]+)")
        if t < best
            best = t; ordv = Vector{Int32}(undef, n); read!(ord, ordv)
            o = invperm(Int.(ordv) .+ 1)
        end
        best > tmo / 3 && break
    end
    return (o, best), "ok"
end

outf, methods, threads = ARGS[1], split(ARGS[2], ","), parse.(Int, split(ARGS[3], ","))
out = open(outf, "a"); host = gethostname()
let g = lat2(10), bg = BipartiteGraph(g)
    permutation(bg, CliqueTrees.AMD()); permutation(bg, CliqueTrees.METIS())
end
tmo = parse(Int, CAP)
function emit(name, n, m, meth, nt, g, res, st)
    if res === nothing
        println(out, join((host, name, n, m, meth, nt, "", "", "", "", st), '\t'))
    else
        o, t = res; q = fillq(g, o)
        println(out, join((host, name, n, m, meth, nt, q[1], q[2], q[3], t, st), '\t'))
        println(rpad(name, 22), rpad(meth, 8), nt, " nnzL=", q[1], " tw=", q[3], " t=", round(t, digits = 4))
    end
    flush(out); flush(stdout)
end
for name in ARGS[4:end]
    g = try loadg(name) catch err; println("SKIP $name: ", sprint(showerror, err)); continue end
    n = nv(g); m = ne(g); bg = BipartiteGraph(g)
    reps = n < 1_000_000 ? 3 : 1
    println("== $name n=$n m=$m"); flush(stdout)
    for meth in methods
        if meth in ("AMD", "METIS")
            alg = meth == "AMD" ? CliqueTrees.AMD() : CliqueTrees.METIS()
            t = Inf; o = nothing
            for r in 1:reps
                t1 = @elapsed (o1 = permutation(bg, alg)[1])
                t1 < t && (t = t1; o = o1)
            end
            emit(name, n, m, meth, 1, g, (o, t), "ok")
        else
            A = adjacency_matrix(g); I_, J_, _ = findnz(A); k_ = I_ .!= J_; A = sparse(I_[k_], J_[k_], true, n, n); I_ = J_ = k_ = nothing   # strip self loops
            path = joinpath(TMPD, name * (meth == "PARAMD" ? ".mtx" : ".csr"))
            ord = path * ".ord"
            meth == "PARAMD" ? writemtx(path, A) : writecsr(path, A)
            for nt in threads
                res, st = meth == "PARAMD" ? run_paramd(path, nt, reps, tmo) : run_parmetis(path, ord, nt, n, reps, tmo)
                emit(name, n, m, meth, nt, g, res, st)
            end
            rm(path; force = true); rm(path * ".perm"; force = true); rm(ord; force = true)
            A = nothing
        end
    end
    g = nothing; bg = nothing; GC.gc()
end
close(out)
