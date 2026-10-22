# Persistent Julia worker for the cotengra plugin (ctg_cliquetrees.py).
#   julia -t <threads> --project=<jlenv> server.jl
# Protocol: one JSON object per line on stdin, one JSON line per reply on stdout (everything else
# goes to stderr). Requests:
#   {"cmd":"warm","methods":["hbtnd",...]}                       -> {"ok":true}
#   {"cmd":"order","method":m,"time":T,"threads":k,"seed":s,
#    "dims":[d_1..d_n],"inputs":[[labels]...],"output":[labels]}  -> {"order":[labels],"seconds":t,"bagwidth":w}
#   {"cmd":"treesa", ... same fields ...}                        -> {"ssa":[[a,b]...],"seconds":t}
#   {"cmd":"quit"}
# Labels are 1..n; dims[l] is the size of label l. "order" lists the labels with dim > 1 in
# elimination order (first eliminated first); size-1 labels never enter the line graph.
# Methods (draft Sec. 5.2, same definitions as tncorpus/prof/common.jl, copied to be self-contained):
#   bw    = BestWidth(MF(), MF(; strategy = 1), AMF(), MMD())
#   nd    = BestWidth of 8 ND(bw, METISND(; seed = s); width = 16, level = 12, scale = 100, imbalance = 300)
#   hbt   = SafeRules(SafeSeparators(HBT(; time, threads, seed), 4 minimal greedy orderings; maxmissing = 50))
#   hbtnd = the same with HBT(; init = nd)
using CliqueTrees, Graphs, Metis, Random, JSON
using OMEinsumContractionOrders
const CT = CliqueTrees
const OMEC = OMEinsumContractionOrders

bwalg() = BestWidth(MF(), MF(; strategy = 1), AMF(), MMD())
nd1(s; alg = bwalg()) = ND(alg, METISND(; seed = s); width = 16, level = 12, scale = 100, imbalance = 300)
nd8(; alg = bwalg()) = BestWidth(ntuple(s -> nd1(s; alg), 8)...)
minalgs() = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))
pipeline(alg) = SafeRules(SafeSeparators(alg, minalgs(); maxmissing = 50))

function linegraph(dims, inputs, output)
    keep = findall(>(1), dims)
    id = zeros(Int, length(dims)); id[keep] .= 1:length(keep)
    n = length(keep)
    adj = [Int[] for _ in 1:n]
    function clique!(t)
        vs = unique(Int[id[l] for l in t if id[l] > 0])
        for a in vs, b in vs
            a == b || push!(adj[a], b)
        end
    end
    foreach(clique!, inputs); clique!(output)
    colptr = Vector{Int}(undef, n + 1); colptr[1] = 1
    for v in 1:n
        sort!(adj[v]); unique!(adj[v]); colptr[v + 1] = colptr[v] + length(adj[v])
    end
    rowval = reduce(vcat, adj; init = Int[])
    g = CT.BipartiteGraph{Int, Int}(n, n, length(rowval), colptr, rowval)
    return keep, log2.(Float64.(dims[keep])), g
end

function algorithm(method, T, threads, seed)
    method == "bw" && return bwalg()
    method == "nd" && return pipeline(nd8())
    method == "hbt" && return pipeline(HBT(; time = T, threads, seed))
    method == "hbtnd" && return pipeline(HBT(; time = T, threads, seed, init = nd8()))
    error("unknown method $method")
end

function order(req)
    dims = Vector{Int}(req["dims"]); inputs = [Vector{Int}(t) for t in req["inputs"]]; output = Vector{Int}(req["output"])
    keep, w, g = linegraph(dims, inputs, output)
    isempty(keep) && return Dict("order" => Int[], "seconds" => 0.0, "bagwidth" => 0.0)
    method = req["method"]
    # bw / nd run without the SafeRules pipeline (tncorpus convention: bw bare, nd bare)
    alg = method == "nd" ? nd8() : algorithm(method, Float64(req["time"]), Int(req["threads"]), Int(req["seed"]))
    t0 = time(); o, _ = permutation(w, g, alg); t = time() - t0
    return Dict("order" => keep[o], "seconds" => t, "bagwidth" => Float64(treewidth(w, g, o)))
end

function tossa(ne, nt)
    ssa = Vector{Int}[]; nxt = Ref(nt)
    function go(x)
        OMEC.isleaf(x) && return x.tensorindex - 1
        @assert length(x.args) == 2
        a = go(x.args[1]); b = go(x.args[2])
        push!(ssa, [a, b]); id = nxt[]; nxt[] += 1
        id
    end
    nt > 1 && go(ne)
    ssa
end

# TreeSA: not time-bounded by itself (niters is a count). One calibration run with niters = 1
# gives the setup + per-iteration rate; the final run gets the iterations that fit the budget.
function treesa(req)
    dims = Vector{Int}(req["dims"]); ixs = [Vector{Int}(t) for t in req["inputs"]]; output = Vector{Int}(req["output"])
    nt = length(ixs); T = Float64(req["time"]); threads = Int(req["threads"])
    nt <= 2 && return Dict("ssa" => nt == 2 ? [[0, 1]] : Vector{Int}[], "seconds" => 0.0)
    code = OMEC.EinCode(ixs, output)
    sizes = Dict(i => dims[i] for i in eachindex(dims))
    score = ScoreFunction(sc_target = 1e4, sc_weight = 0.0)   # total time complexity only
    t0 = time()
    opt = optimize_code(code, sizes, TreeSA(ntrials = threads, niters = 1, score = score)); t1 = time() - t0
    left = T - (time() - t0)
    # a niters=1 run costs ~ init + 1 sweep; the sweep time is not separable, so assume the
    # whole calibration run is dominated by the sweeps (conservative: fewer iterations)
    ni = max(0, floor(Int, left / max(t1, 1e-3)))
    ni >= 2 && (opt = optimize_code(code, sizes, TreeSA(ntrials = threads, niters = ni, score = score)))
    return Dict("ssa" => tossa(opt, nt), "seconds" => time() - t0)
end

function warm(methods)
    # small random network: 3-regular-ish tensors on ~60 indices
    rng = Xoshiro(1); n = 60
    inputs = [sort(unique(rand(rng, 1:n, 3))) for _ in 1:80]
    dims = fill(2, n)
    for m in methods
        if m == "treesa"
            treesa(Dict("dims" => dims, "inputs" => inputs, "output" => Int[], "time" => 0.5, "threads" => Threads.nthreads()))
        else
            order(Dict("dims" => dims, "inputs" => inputs, "output" => Int[1, 2], "method" => m, "time" => 0.3,
                "threads" => Threads.nthreads(), "seed" => 1))
        end
    end
end

function main()
    out = stdout
    redirect_stdout(stderr)     # stray prints must not corrupt the protocol
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        reply = try
            req = JSON.parse(line)
            cmd = req["cmd"]
            cmd == "quit" && break
            cmd == "warm" ? (warm(req["methods"]); Dict("ok" => true)) :
            cmd == "order" ? order(req) :
            cmd == "treesa" ? treesa(req) : error("unknown cmd $cmd")
        catch e
            Dict("error" => sprint(showerror, e, catch_backtrace()))
        end
        println(out, JSON.json(reply)); flush(out)
    end
end
main()
