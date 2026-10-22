# B200 benchmark of GPU ND + batched GPU MF1 leaves (ND_NOTES.md)
# usage: julia --project=ENV -t1 ndbench.jl
# env: OUT (tsv), DATA (dirs with NAME.bin / NAME.mtx), NAMES (comma list), CFGS (comma list of kind:Lb[:key=val;...]):
#   gnd:L         GPU ND (GND.jl) + one grounds_batch call (MF1, relax 0.1, unit weights) on the leaves
#   gndd:L        GPU ND + device-resident leaves (grounds_batch_dev: leaf CSR and orders never leave the device)
#   race:L        policy.jl race_perm: serial AMF on one thread vs gndd+hyb (Lb=L); GND kept if done by slack x t(AMF)
#   auto:L        policy.jl: GPU ND (gndd + hyb=1, Lb=L) or serial AMD/AMF/MF1 from calibrated predicted times
#                 (key slack=..., in percent: 125 = 1.25 x AMD)
#   amd:0 / amf:0 serial AMD (SuiteSparse through CliqueTrees AMD(), needs AMD.jl) / AMF on this host's CPU (1 thread)
#   hybrid CPU+GPU refinement: key hyb=1 (cost model; calibrated once after the warm-up, hyb_calibrate!)
#   gndcpu:L      GPU ND + serial MF1 (CliqueTrees in this env) on the leaves (fill reference, not timed vs GPU)
#   metis:L       CPU recursive METIS separators + the same grounds_batch leaves
#   extra GNDOpts as key=val after a second colon, e.g. gnd:4096:rmax=4;tfm=20000
#      GRDIR: directory of Rounds.jl/GRounds.jl/GSmall.jl (default ..)
# rows: host gpu graph n m cfg Lb nnzL flops tw time extra   (time = graph upload .. permutation on host, min of REPS)
const NDD = @__DIR__
const RDIR = joinpath(NDD, "..")
using CUDA, CliqueTrees, Graphs, SparseArrays, Serialization, Metis
const HAVEAMD = Base.find_package("AMD") !== nothing
HAVEAMD && @eval using AMD
const GRDIR = get(ENV, "GRDIR", RDIR)     # where Rounds.jl / GRounds.jl / GSmall.jl come from
include(joinpath(GRDIR, "Rounds.jl")); using .Rounds
include(joinpath(GRDIR, "GRounds.jl")); using .GRounds
include(joinpath(NDD, "ndutil.jl"))
include(joinpath(NDD, "GND.jl")); using .GND
include(joinpath(NDD, "cpund.jl"))
include(joinpath(NDD, "policy.jl"))

const ALG = RoundsAlg(; score = :mf1, relax = 0.1)
const NEWAPI = isdefined(GRounds, :srcof)
const NW = parse(Int, get(ENV, "NW", NEWAPI ? "0" : "8"))
const host = gethostname()
const gpuname = CUDA.name(CUDA.device())

function gpu_leafsolve(tm)
    return csrs -> begin
        bt = Dict{Symbol, Any}()
        # grounds_batch API (Oct 20): wts = nothing for unit weights, nw = 0 picks the block shape;
        # orders[b] are views into one buffer, consumed before the next call
        orders, info = GRounds.grounds_batch_csr(csrs, NEWAPI ? nothing : [ones(length(p) - 1) for (p, _) in csrs], ALG; nw = NW, timing = bt)
        for (k, v) in bt
            v isa Number && (tm[Symbol("b_", k)] = v)
        end
        orders
    end
end
const MF1 = MF(; strategy = 1)
function gpu_leafdev(tm)
    return (uptr, uadj, loffH) -> begin
        bt = Dict{Symbol, Any}()
        dord, info = GRounds.grounds_batch_dev(uptr, uadj, loffH, ALG; timing = bt)
        for (k, v) in bt
            v isa Number && (tm[Symbol("b_", k)] = v)
        end
        dord
    end
end

function parsecfg(s)
    f = split(s, ":")
    kind = String(f[1]); L = parse(Int, f[2])
    kw = Dict{Symbol, Any}()
    if length(f) >= 3
        for kv in split(f[3], ";")
            k, v = split(kv, "=")
            k = Symbol(k)
            kw[k] = k in (:ref,) ? Symbol(v) : k in (:eps, :stall, :sepmax) ? parse(Float64, v) : k in (:compress, :mdeg, :hop2, :fm1, :fminc) ? parse(Bool, v) : parse(Int, v)
        end
    end
    return kind, L, kw
end

fmt(x) = x isa AbstractFloat ? round(x, sigdigits = 5) : x
tstr(tm) = join(["$k=$(v isa AbstractVector{<:Real} ? join(fmt.(v), ",") : fmt(v))" for (k, v) in sort(collect(tm); by = first) if !(v isa AbstractArray) || v isa AbstractVector{<:Real}], ";")

const PREP = Ref{Any}(nothing)
const RACEBG = Ref{Any}(nothing)      # a losing serial task of the race: waited for outside the timed region
serialalg_(a) = a == "AMD" ? CliqueTrees.AMD() : a == "AMF" ? AMF() : MF(; strategy = 1)
function runser(a, ptr, adj)
    n = length(ptr) - 1
    A = SparseMatrixCSC(n, n, collect(Int, ptr), collect(Int, adj), ones(Bool, length(adj)))
    return permutation(BipartiteGraph(A), serialalg_(a))[1]
end
prep!(ptr, adj) = (PREP[] = BipartiteGraph(csr_graph(ptr, adj)); nothing)
function run1(kind, L, kw, ptr, adj)
    tm = Dict{Symbol, Any}()
    if kind == "gnd"
        p = gnd_perm(ptr, adj; opts = GNDOpts(; Lb = L, kw...), leafsolve = gpu_leafsolve(tm), timing = tm)
    elseif kind == "gndd"
        p = gnd_perm(ptr, adj; opts = GNDOpts(; Lb = L, kw...), leafdev = gpu_leafdev(tm), timing = tm)
    elseif kind == "amd" || kind == "amf"
        # input prepared outside the timed region (as the CPU baselines: BipartiteGraph of the graph)
        p = permutation(PREP[], kind == "amd" ? CliqueTrees.AMD() : AMF())[1]
    elseif kind == "race" && (length(ptr) - 1) + length(adj) / 2 > race_cap()
        # above the calibrated race cap: GPU ND alone
        tm[:choice] = -1
        p = gnd_perm(ptr, adj; opts = GNDOpts(; Lb = L, hyb = 1, rmax = get(kw, :rmax, 8)), leafdev = gpu_leafdev(tm), timing = tm)
    elseif kind == "race"
        sl = get(kw, :slack, 125) / 100
        p, win, ser = race_perm((p_, a_, ab) -> gnd_perm(p_, a_; opts = GNDOpts(; Lb = L, hyb = 1, rmax = get(kw, :rmax, 8)), leafdev = gpu_leafdev(tm), timing = tm, abort = ab),
            runser, ptr, adj; slack = sl, serial = get(kw, :serial, 2) == 3 ? "AMD" : "AMF")
        tm[:choice] = win == "gnd" ? 0 : 2
        RACEBG[] = ser
    elseif kind == "auto"
        n = length(ptr) - 1
        sl = get(kw, :slack, 125) / 100
        ch, tp, ta = policy_choose(n, length(adj) ÷ 2; slack = sl)
        kw = filter(kv -> kv[1] != :slack, kw)
        tm[:choice] = ch == "gnd" ? 0 : ch == "MF1" ? 1 : ch == "AMF" ? 2 : 3; tm[:tpred] = tp; tm[:tamdpred] = ta
        if ch == "gnd"
            p = gnd_perm(ptr, adj; opts = GNDOpts(; Lb = L, hyb = 1, rmax = get(kw, :rmax, 8)), leafdev = gpu_leafdev(tm), timing = tm)
        else
            # the serial path includes building its input (the policy is end to end: host CSR in)
            p = runser(ch, ptr, adj)
        end
    elseif kind == "gndcpu"
        p = gnd_perm(ptr, adj; opts = GNDOpts(; Lb = L, kw...), leafsolve = cpu_leafsolve(MF1), timing = tm)
    elseif kind == "metis"
        p = metis_nd_perm(ptr, adj; Lb = L, leafsolve = gpu_leafsolve(tm), timing = tm)
    else
        error("cfg $kind")
    end
    return p, tm
end

# warm-up: every kernel and code path on a small graph
let (ptr, adj) = graph_csr(lat2(40))
    for kind in ("gnd", "metis"), L in (64, 400)
        run1(kind, L, Dict{Symbol, Any}(), ptr, adj)
    end
    run1("gnd", 64, Dict{Symbol, Any}(:rmax => 1), ptr, adj)
    run1("gnd", 64, Dict{Symbol, Any}(:tfm => 0), ptr, adj)
    run1("gnd", 64, Dict{Symbol, Any}(:pgmin => 0), ptr, adj)
    for h in (1, 2), kind in ("gnd", "gndd")
        run1(kind, 64, Dict{Symbol, Any}(:hyb => h), ptr, adj)
        run1(kind, 400, Dict{Symbol, Any}(:hyb => h), ptr, adj)
    end
    run1("gndd", 400, Dict{Symbol, Any}(), ptr, adj)
    prep!(ptr, adj)
    HAVEAMD && run1("amd", 0, Dict{Symbol, Any}(), ptr, adj)
    run1("amf", 0, Dict{Symbol, Any}(), ptr, adj)
    for kw in (Dict{Symbol, Any}(:hyb => 1, :kway => 2), Dict{Symbol, Any}(:hyb => 1, :kway => 3))
        run1("gndd", 64, kw, ptr, adj)
    end
    run1("race", 64, Dict{Symbol, Any}(), ptr, adj); RACEBG[] === nothing || wait(RACEBG[])
    run1("race", 64, Dict{Symbol, Any}(:slack => 1), ptr, adj); RACEBG[] === nothing || wait(RACEBG[])
end
tcal = @elapsed GND.hyb_calibrate!()
println("threads=", Threads.nthreads(), " calibration ", round(tcal, digits = 2), " s: ", GND.HCAL); flush(stdout)
if HAVEAMD && any(startswith(c, "auto") || startswith(c, "race") for c in split(get(ENV, "CFGS", ""), ","))
    let (p, a) = graph_csr(lat2(40)); for x in ("AMD", "AMF", "MF1"); runser(x, p, a); end; end
    tp = @elapsed policy_calibrate!((p, a) -> gnd_perm(p, a; opts = GNDOpts(; Lb = 256, hyb = 1), leafdev = gpu_leafdev(Dict{Symbol, Any}())), runser)
    println("policy calibration ", round(tp, digits = 2), " s: ", PCAL, " crossover n+m (slack 1.25) = ", round(policy_crossover()), " race cap n+m = ", race_cap())
    flush(stdout)
end
println(gpuname, " nsm=", CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)); flush(stdout)

out = open(ENV["OUT"], "a")
cfgs = [parsecfg(s) for s in split(get(ENV, "CFGS", "gnd:4096"), ",") if !isempty(s)]
reps = parse(Int, get(ENV, "REPS", "2"))
for name in split(get(ENV, "NAMES", ""), ",")
    isempty(name) && continue
    tl = @elapsed (ptr, adj) = try load_csr(String(name)) catch err; println("LOAD FAILED $name: $err"); continue end
    n = length(ptr) - 1; m = length(adj) ÷ 2
    g = nothing
    println(name, " n=", n, " m=", m, " load ", round(tl, digits = 1), " s"); flush(stdout)
    any(c -> c[1] in ("amd", "amf"), cfgs) && prep!(ptr, adj)
    for (kind, L, kw) in cfgs
        try
            t = Inf; tm = nothing; p = nothing
            for r in 1:(kind == "metis" || kind == "gndcpu" ? 1 : (kind == "amd" || kind == "amf") && n > 2_000_000 ? 1 : reps)
                GC.gc(false); CUDA.reclaim()
                t1 = @elapsed begin
                    p1, tm1 = run1(kind, L, kw, ptr, adj)
                end
                RACEBG[] === nothing || (wait(RACEBG[]); RACEBG[] = nothing)
                if t1 < t
                    t = t1; tm = tm1; p = p1
                end
            end
            g === nothing && (g = csr_graph(ptr, adj))
            q = fillq(g, p)
            cfg = kind * (isempty(kw) ? "" : ":" * join(["$k=$v" for (k, v) in kw], ";"))
            tm[:threads] = Threads.nthreads()
            println(out, join((host, gpuname, name, n, m, cfg, L, q[1], q[2], q[3], fmt(t), tstr(tm)), '\t')); flush(out)
            println(rpad(name, 16), rpad(cfg, 28), " L=", L, " nnzL=", q[1], " flops=", fmt(q[2]), " tw=", q[3], " t=", round(t, digits = 4),
                " | nd=", fmt(get(tm, :nd, NaN)), " leaves=", fmt(get(tm, :leaves, NaN)), " (gpu ", fmt(get(tm, :b_gpu, NaN)), ") nleaves=",
                get(tm, :nleaves, -1)); flush(stdout)
        catch err
            println("FAILED $name $kind:$L: ", first(sprint(showerror, err, catch_backtrace()), 1500)); flush(stdout)
        end
    end
    g = nothing; PREP[] = nothing; GC.gc(); CUDA.reclaim()
end
close(out)
println("DONE")
