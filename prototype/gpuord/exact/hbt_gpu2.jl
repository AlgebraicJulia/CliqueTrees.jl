# HBT (safe pipeline, 1 CPU thread), CPU-only vs CPU+GPU exact local triangulations, several seeds.
# usage: julia hbt_gpu2.jl T nseeds inst...
using CUDA, CliqueTrees, Graphs, Printf, Serialization, Statistics
const GO = dirname(@__DIR__)
include(joinpath(@__DIR__, "SeptwGPU.jl"))
include(get(ENV, "TNLOAD", "/path/to/tn-work/tncorpus/load.jl")); using .TNCorpus
const MINS = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))
hbt(T, seed) = SafeRules(SafeSeparators(HBT(; time = T, threads = 1, seed), MINS; maxmissing = 50))
T = parse(Float64, ARGS[1]); NS = parse(Int, ARGS[2])
permutation(ones(100), Graphs.grid([10, 10]); alg = hbt(0.3, 1))
println("GPU: ", CUDA.name(CUDA.device()), "  delta = ", round(1000calibrate!(), digits = 2), " ms")
res = Dict{String, Any}()
for nm in ARGS[3:end]
    inst = TNCorpus.load(nm); out = Any[]
    for seed in 1:NS, mode in (:cpu, :gpu)
        mode === :cpu ? hook_hbt_count!() : hook_hbt_escalate!()
        for d in (CPUC, ESC), k in keys(d); d[k] = 0; end
        Base.invokelatest() do
            t = @elapsed o, _ = permutation(inst.weights, inst.graph; alg = hbt(T, seed))
            w = treewidth(inst.weights, inst.graph; alg = o)
            push!(out, (; seed, mode, width = w, t, cpu = copy(CPUC), esc = copy(ESC)))
        end
    end
    res[nm] = out
    wc = [r.width for r in out if r.mode === :cpu]; wg = [r.width for r in out if r.mode === :gpu]
    cc = mean(r.cpu[:calls] for r in out if r.mode === :cpu); cf = mean(r.cpu[:found] for r in out if r.mode === :cpu)
    gc = mean(r.esc[:calls] for r in out if r.mode === :gpu); ge = mean(r.esc[:gpu] for r in out if r.mode === :gpu); gf = mean(r.esc[:gpu_found] for r in out if r.mode === :gpu)
    @printf("%-34s n=%6d T=%.0fs | widths CPU %s  CPU+GPU %s | mean exact calls: CPU %.0f (found %.0f) vs CPU+GPU %.0f (%.0f to GPU, %.0f found there)\n",
        nm, nv(inst.graph), T, string(round.(wc; digits = 2)), string(round.(wg; digits = 2)), cc, cf, gc, ge, gf)
    flush(stdout)
end
serialize(joinpath(GO, "results", "hbt_gpu2_$(replace(CUDA.name(CUDA.device()), ' ' => '_'))_T$(Int(T)).jls"), res)
