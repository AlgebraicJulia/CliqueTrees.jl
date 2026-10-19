# HBT (safe pipeline, 1 CPU thread) with and without GPU exact local triangulations.
# usage: julia hbt_gpu.jl T inst...   (mode from env HBTMODE = cpu | gpu | both)
using CUDA, CliqueTrees, Graphs, Printf, Serialization
const GO = dirname(@__DIR__)
include(joinpath(@__DIR__, "SeptwGPU.jl"))
include(get(ENV, "TNLOAD", "/tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/tncorpus/load.jl")); using .TNCorpus
const MINS = (MinimalChordal(MF()), MinimalChordal(MF(; strategy = 1)), MinimalChordal(AMF()), MinimalChordal(MMD()))
hbt(T) = SafeRules(SafeSeparators(HBT(; time = T, threads = 1), MINS; maxmissing = 50))
T = parse(Float64, ARGS[1]); mode = get(ENV, "HBTMODE", "both")
permutation(ones(100), Graphs.grid([10, 10]); alg = hbt(0.3))
println("GPU: ", CUDA.name(CUDA.device()), "  delta = ", round(1000calibrate!(), digits = 2), " ms")
res = Dict{String, Any}()
for nm in ARGS[2:end]
    inst = TNCorpus.load(nm)
    out = Dict{String, Any}()
    if mode in ("cpu", "both")
        # stock CliqueTrees (the hook is not installed yet in this branch)
        t = @elapsed o, _ = permutation(inst.weights, inst.graph; alg = hbt(T))
        out["cpu"] = (; width = treewidth(inst.weights, inst.graph; alg = o), t)
    end
    if mode in ("gpu", "both")
        hook_hbt_escalate!()
        for k in keys(ESC); ESC[k] = 0; end
        Base.invokelatest() do
            t = @elapsed o, _ = permutation(inst.weights, inst.graph; alg = hbt(T))
            out["gpu"] = (; width = treewidth(inst.weights, inst.graph; alg = o), t, esc = copy(ESC), rounds = GPUSTATS[:rounds], tgpu = GPUSTATS[:tgpu])
        end
    end
    res[nm] = out
    c = get(out, "cpu", nothing); g = get(out, "gpu", nothing)
    @printf("%-34s n=%6d | CPU HBT %.0fs: width %s | CPU+GPU HBT: width %s  (exact calls %d: %d CPU, %d escalated to GPU -> %d found, %d timeouts)\n",
        nm, nv(inst.graph), T, c === nothing ? "-" : string(round(c.width, digits = 2)), g === nothing ? "-" : string(round(g.width, digits = 2)),
        g === nothing ? 0 : g.esc[:calls], g === nothing ? 0 : g.esc[:cpu], g === nothing ? 0 : g.esc[:gpu], g === nothing ? 0 : g.esc[:gpu_found], g === nothing ? 0 : g.esc[:gpu_timeout])
    flush(stdout)
end
serialize(joinpath(GO, "results", "hbt_gpu_$(replace(CUDA.name(CUDA.device()), ' ' => '_'))_T$(Int(T)).jls"), res)
