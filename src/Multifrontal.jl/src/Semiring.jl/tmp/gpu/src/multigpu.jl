# ===== multi-GPU closure =====
#
# The rows of the closure are independent: row t holds the distances from one source, and the solve
# for a block of rows (sssp_gpu!) reads only the factor and writes only those rows. So every GPU gets
# its own copy of the factor (megabytes, next to gigabytes of output) and computes a contiguous block
# of rows with the single-GPU kernels, with no communication between GPUs. Each GPU holds only its
# block, so the output can be larger than one GPU's memory.
#
#   MG = MultiGPUSLU(F, factor; devices)      # factor: (LD, LL, UD, UL) on any device or the host
#   blocks = closure_multigpu!(MG)            # [(rows, D_g)], D_g = D[rows, :] on devices[g]
#
# Rows and columns are in elimination coordinates, as in closure_gpu!: D[i, j] = A*[rperm[i], rperm[j]].

struct MultiGPUSLU{G}
    devices::Vector{CuDevice}
    parts::Vector{G}                    # parts[g] lives on devices[g]
    n::Int
end

# per-device copies of the solve structure and factor. The factor arrays are copied through the host once.
function MultiGPUSLU(F::ChordalSLU, factor::NTuple{4, AbstractVector}; devices = collect(CUDA.devices()), large::Integer = 8192, ops::Bool = true)
    hfactor = map(Array, factor)
    parts = Vector{Any}(undef, length(devices))

    @sync for (g, d) in enumerate(devices)
        Threads.@spawn begin
            CUDA.device!(d)
            G = GPUSLU(F; large, factor = map(CuVector, hfactor))
            ops && precompute_ops!(G)
            CUDA.synchronize()
            parts[g] = G
        end
    end

    return MultiGPUSLU(collect(devices), [p for p in parts], parts[1].n)
end

MultiGPUSLU(P::FactorPlan; kw...) = MultiGPUSLU(P.F, (P.LD, P.LL, P.UD, P.UL); kw...)

# contiguous row blocks, as equal as possible
function row_blocks(n::Integer, ng::Integer)
    q, r = divrem(n, ng)
    stops = cumsum([q + (g <= r) for g in 1:ng])
    return [(g == 1 ? 1 : stops[g - 1] + 1):stops[g] for g in 1:ng]
end

"""
    closure_multigpu!(MG; out = nothing, timer = nothing) -> Vector{Tuple{UnitRange{Int}, CuMatrix}}

The closure in elimination coordinates, split by rows over `MG.devices`: block g is `D[rows_g, :]` on
device g. `out` may hold preallocated blocks (one per device, of the right sizes, on their devices).
With `timer`, `timer[g]` is device g's wall time in seconds.
"""
function closure_multigpu!(MG::MultiGPUSLU; out = nothing, timer = nothing)
    ng = length(MG.devices)
    rows = row_blocks(MG.n, ng)
    blocks = Vector{Any}(undef, ng)

    @sync for g in 1:ng
        Threads.@spawn begin
            CUDA.device!(MG.devices[g])
            G = MG.parts[g]
            T = eltype(G.LDval)
            k = length(rows[g])
            t0 = time()
            X = isnothing(out) ? (need_memory(k * G.n * sizeof(T), "a $k × $(G.n) closure block"); CuMatrix{T}(undef, k, G.n)) : out[g]
            @assert size(X) == (k, G.n)

            if k > 0
                sources = upload(Array(G.rperm)[rows[g]])
                M = CuMatrix{T}(undef, k, G.maxna)
                sssp_gpu!(X, G, sources; W = X, M, permute = false)
                CUDA.unsafe_free!(M)
            end

            CUDA.synchronize()
            isnothing(timer) || (timer[g] = time() - t0)
            blocks[g] = (rows[g], X)
        end
    end

    return [b for b in blocks]
end
