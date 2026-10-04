# ===== apsp_gpu: the closure in the original labels =====
#
# One call for the whole pipeline, with the tuned settings of bench/portable.jl:
#
#   F = ChordalSLU(s, A); copyto!(F, A)                            symbolic phase (CPU), entries of A
#   P = FactorPlan(F; large = 256, graph = false, nstreams = 8)     hybrid numeric LU: bottom subtrees on
#   factorize!(P)                                                    CPU threads, the top fronts on the GPU
#   G = GPUSLU(P; large = 8192); precompute_ops!(G)                 solve structure and the large fronts' dense operators
#   sssp_gpu!(D, G, 1:n; W = D, M, permute = false)                  D[i, j] = A*[i, p[j]],  p = rperm
#   D ← D[:, q],  q = p⁻¹ = cinvp                                    D[i, j] = A*[i, j]
#
# The sources are taken in the labels of A, so only the columns are in elimination order (the rows of
# the closure are independent: their order changes nothing but which row holds which source). D may
# fill most of the GPU, so its columns are relabelled in place through a buffer W of b rows:
#
#   rows I, b at a time:   W ← D[I, q];  D[I, :] ← W       now D[i, j] = A*[i, j]
#
# (relabel_rows! does the same for rows, for the blocks of the multi-GPU closure.)
#
# With several devices, closure_multigpu! leaves block g = D[rows_g, :] on devices[g] (rows_g a range
# of elimination order: the sources p[rows_g]). Each device relabels the columns of its own block
# (pass 1); :host then scatters the rows of every block into H[p[rows_g], :].

"""
    apsp_gpu(A; semiring = MinPlus(), devices = [CUDA.device()], output = :device)
    apsp_gpu(A, sources; semiring = MinPlus(), output = :device)

The closure A* of the square sparse matrix `A` over `semiring`, on the GPU, in the vertex labels of `A`:
`D[i, j] = A*[i, j]`, the ⊕ over all paths i → j of the ⊗ of their arc weights, where `A[i, j]` is the
weight of the arc i → j. With the default min-plus semiring this is all-pairs shortest paths: `D[i, j]`
is the distance from i to j (`Inf` when j cannot be reached from i).

- `output = :device` returns a `CuMatrix` (on `devices[1]`), `output = :host` a `Matrix`.
- `columns = :elimination` (one GPU, `output = :device`) skips the last step, which moves the columns into
  the labels of `A` (a pass over the n × n result), and returns `(D, cols)` with `D[i, k] = A*[i, cols[k]]`:
  the same values, its columns in the solver's elimination order. For codes that read the result through
  a vertex map, or compare with solvers that also return their own order.
- `sources`: only the rows `D[sources, :]`, a k × n matrix with row t = A*[sources[t], :] (any order,
  repeats allowed). For when the n × n closure does not fit, or only some sources are needed; call it
  for blocks of sources to stream the closure.
- `devices`: with more than one GPU the rows are split over them (each holds only its block, so the
  closure may exceed one GPU's memory; a device may be repeated). `:host` assembles the whole `Matrix`;
  `:device` returns one block per device, `[(sources_g, D_g), ...]` with
  `D_g[t, j] = A*[sources_g[t], j]` on `devices[g]` (`sources_g::Vector{Int}`, columns in the labels of `A`).

The element type is `eltype(A)`: Float32 or Float64 for min-plus and max-plus (Int32 and Int64 are
rejected, their infinity overflows; see `check_semiring`), also Int32 for max-min, Float64 for
plus-times. `semiring` is one of `CliqueTrees.Multifrontal.Semiring`'s (`using SemiringGPU.Semiring:
MaxPlus, MaxMin, PlusProd`). Undirected graphs (a symmetric pattern) and directed graphs whose strongly
connected components do not reach one another are supported; other input throws an `ArgumentError`, as
does a non-square `A`. A result that does not fit in GPU memory is an error that says so before any work.

The solver settings come from `with_config` and the `SEMIRINGGPU_*` environment variables (see
`GPUConfig`), e.g. `with_config(() -> apsp_gpu(A); merge = 1)`.

Every call runs the whole pipeline (symbolic phase, numeric factorization, solve) and frees its GPU
workspace before returning. For repeated solves on one graph, use the lower-level API, which keeps the
factorization and gives results in elimination coordinates (`p = F.rperm`):

    F = ChordalSLU(MinPlus(), A); copyto!(F, A)           # symbolic phase, entries of A
    P = FactorPlan(F; large = 256, graph = false, nstreams = 8)
    factorize!(P)                                         # new weights, same pattern: copyto!(F, A₂); factorize!(P)
    G = GPUSLU(P; large = 8192); precompute_ops!(G)
    closure_gpu!(D, G; M)                                 # D[i, j] = A*[p[i], p[j]] (n × n), M: n × G.maxna
    sssp_gpu!(X, G, CuVector(sources))                    # X[t, j] = A*[sources[t], j], original labels

"""
function apsp_gpu(A::SparseMatrixCSC; semiring::AbstractSemiring = Semiring.MinPlus(), devices::AbstractVector = [CUDA.device()],
        output::Symbol = :device, columns::Symbol = :original, buffer::Integer = 0)
    #
    # buffer (internal, for tests): elements of the relabelling buffer, 0 = automatic (see relabel_length)
    #
    check_apsp(A, semiring, output)
    isempty(devices) && throw(ArgumentError("apsp_gpu: no devices given"))
    all(d -> d isa CuDevice, devices) || throw(ArgumentError("apsp_gpu: devices must be CuDevices, e.g. collect(CUDA.devices())"))
    columns in (:original, :elimination) || throw(ArgumentError("apsp_gpu: columns must be :original or :elimination, not $(repr(columns))"))
    columns === :original || (output === :device && length(devices) == 1) ||
        throw(ArgumentError("apsp_gpu: columns = :elimination needs output = :device and one device"))

    if length(devices) > 1
        return without_early_gc(() -> apsp_multigpu(A, semiring, collect(CuDevice, devices), output, buffer))
    end

    return without_early_gc(() -> CUDA.device!(() -> apsp_single(A, semiring, output, buffer; columns), first(devices)))
end

function apsp_gpu(A::SparseMatrixCSC{T}, sources::AbstractVector{<:Integer}; semiring::AbstractSemiring = Semiring.MinPlus(),
        output::Symbol = :device) where {T}
    check_apsp(A, semiring, output)
    return without_early_gc(() -> apsp_sources(A, sources, semiring, output))
end

function apsp_sources(A::SparseMatrixCSC{T}, sources::AbstractVector{<:Integer}, semiring::AbstractSemiring, output::Symbol) where {T}
    n = size(A, 1)
    k = length(sources)
    bad = findfirst(v -> !(1 <= v <= n), sources)
    isnothing(bad) || throw(ArgumentError("apsp_gpu: sources must be vertices in 1:$n, but sources[$bad] = $(sources[bad])"))

    if iszero(k)
        return output === :host ? Matrix{T}(undef, 0, n) : CuMatrix{T}(undef, 0, n)
    end

    advice = "use fewer sources per call"
    need_apsp_memory(2 * k * n * sizeof(T), "$k rows of the closure and their workspace", advice)       # before any work

    X = with_host_buffers() do B
        P = apsp_factor(semiring, A; host = B)
        G = GPUSLU(P; large = 8192)
        precompute_ops!(G)
        need_apsp_memory((2 * k * n + k * G.maxna) * sizeof(T), "$k rows of the closure and their workspace", advice)
        X = CuMatrix{T}(undef, k, n)
        W = similar(X)
        M = CuMatrix{T}(undef, k, G.maxna)
        #
        #   X ← B A*,  B = [e_{s₁}; …; e_{s_k}]     (sources and columns in the labels of A)
        #
        sssp_gpu!(X, G, CuVector{Int}(sources); W, M, permute = true)
        CUDA.synchronize()
        CUDA.unsafe_free!(W); CUDA.unsafe_free!(M)
        free_solver!(G); free_plan!(P)
        X
    end

    output === :device && return X
    H = Array(X)
    CUDA.unsafe_free!(X)
    return H
end

# ===== single GPU =====

function apsp_single(A::SparseMatrixCSC{T}, s::AbstractSemiring, output::Symbol, buffer::Integer;
        H::Union{Nothing, Matrix{T}} = nothing, columns::Symbol = :original) where {T}
    n = size(A, 1)
    iszero(n) && return output === :host ? something(H, Matrix{T}(undef, 0, 0)) : CuMatrix{T}(undef, 0, 0)
    advice = "split the rows over several GPUs (devices = [...]) or compute blocks of rows (apsp_gpu(A, sources))"
    need_apsp_memory(n^2 * sizeof(T), "the $n × $n closure", advice)                                  # before any work
    H = @step "host matrix" (output === :host && isnothing(H) ? host_matrix(T, n, n) : H)
    Dt = allocate_async(T, n, n)        # the result: fresh device memory is mapped while the host orders the graph

    return with_host_buffers() do B
        try
            P, st = apsp_factor(s, A; host = B, solve = 8192)
            G = @step "solve setup" GPUSLU(P; large = 8192, structure = structure_of(st))
            @step "operators" precompute_ops!(G)
            need_apsp_memory(n * G.maxna * sizeof(T), "the workspace of the $n × $n closure", advice)
            release = () -> (free_solver!(G); free_plan!(P))                     # (freed before the relabelling buffer)
            if columns === :elimination
                cols = Vector{Int}(P.F.cperm)
                (closure_labels(G, buffer, release; D = Dt, relabel = false), cols)
            else
                deliver(closure_labels(G, buffer, release; D = Dt), H)
            end
        catch
            discard(Dt)
            rethrow()
        end
    end
end

#
#   an m × n device matrix, allocated on another thread (on the current device): a large fresh allocation
#   costs ~1.5 ms/GB (the driver maps and clears the pages), time the host can spend on the symbolic phase
function allocate_async(::Type{T}, m::Integer, n::Integer) where {T}
    dev = CUDA.device()

    return Threads.@spawn begin
        CUDA.device!(dev)
        X = CuMatrix{T}(undef, m, n)
        CUDA.synchronize()
        X
    end
end

function fetch_result(t::Task)
    try
        return fetch(t)
    catch err
        err isa TaskFailedException ? throw(err.task.exception) : rethrow()
    end
end

discard(t::Task) = try CUDA.unsafe_free!(fetch(t)) catch end

#
#   D[i, :] ← A*[i, p]   every vertex a source, in the labels of A (the rows need no relabelling: the
#                        closure costs the same in any order of its rows), columns in elimination order
#   D ← D[:, q]          now D[i, j] = A*[i, j]
#
# release() runs between the two (it frees the factorization when it is not kept).
#
function closure_labels(G::GPUSLU{<:Any, T}, buffer::Integer, release; D::Union{Nothing, Task} = nothing, relabel::Bool = true) where {T}
    n = G.n
    D, M = @step "allocate" (isnothing(D) ? CuMatrix{T}(undef, n, n) : fetch_result(D), CuMatrix{T}(undef, n, G.maxna))
    #
    # in the original labels: every column written where it belongs (ColMapped), no relabel after
    #
    mapped = relabel && MAPPED_CLOSURE[] && factor_fill(G) <= MAPPED_MAX_FILL * n
    W = mapped ? ColMapped(D, G.rperm) : D
    @step "closure" sssp_gpu!(W, G, CuVector{Int}(1:n); W, M, permute = false)
    CUDA.synchronize()
    @step "free" (CUDA.unsafe_free!(M); release())
    (relabel && !mapped) || return D
    return @step "relabel columns" relabel_cols(D, G.cinvp, buffer)
end

# the closure writes the original labels' columns directly (false: in elimination order, then a relabel)
const MAPPED_CLOSURE = Ref(true)
#
# ... when the factor has at most this many entries per vertex. The relabel costs a read and a write of
# the n × n result; the mapped closure costs its dense fronts' GEMMs reading and writing through index
# views (a few percent of them). With little fill the relabel is a large part of the call (B200 / RTX PRO
# 6000, cold calls: luxembourg_osm 1.13× / 1.59×, delaunay_n16 1.06× / 1.29× faster mapped); with much,
# the dense fronts are (grid3d-30 0.81× / 0.96×, ca-CondMat 0.85× / 0.95×). On the 30-graph suite every
# graph with ≤ 30 entries per vertex (as counted here, after amalgamation) gained or tied and those with
# ≥ 45 lost or tied: a threshold chosen on that suite.
#
const MAPPED_MAX_FILL = 36

# entries of the solve's L (U has the same, amalgamated fronts): Σ over fronts of nn (nn + 1) / 2 + nn na
function factor_fill(G::GPUSLU)
    get!(G.cache, :fill) do
        sum(f -> (nn = Int(G.hRptr[f + 1] - G.hRptr[f]); nn * (nn + 1) ÷ 2 + nn * Int(G.hSptr[f + 1] - G.hSptr[f])), 1:G.nf; init = 0)
    end::Int
end

# D on the device, or copied into the host matrix H (and freed)
function deliver(D::CuMatrix, H)
    isnothing(H) && return D
    size(H) == size(D) || throw(DimensionMismatch("apsp_gpu!: the output is $(size(H)), the closure $(size(D))"))
    @step "to host" copyto!(H, D)
    CUDA.unsafe_free!(D)
    return H
end

#
# A host matrix for the result, on transparent huge pages where the OS offers them (Linux with THP in
# madvise mode, as on HiPerGator): the copy from the GPU first-touches every page of a fresh matrix,
# and 2 MB pages fault 512× less often than 4 KB ones (an n × n Float32 result of n = 27000 took 0.6 s
# in page faults; 0.22 s on huge pages). A matrix reused across calls (apsp_gpu!) has no faults at all.
#
function host_matrix(::Type{T}, m::Integer, n::Integer) where {T}
    H = Matrix{T}(undef, m, n)

    if Sys.islinux() && sizeof(H) >= 2^21
        p0 = UInt(pointer(H)); a0 = (p0 + 4095) & ~UInt(4095); len = (p0 + sizeof(H) - a0) & ~UInt(4095)
        ccall(:madvise, Cint, (Ptr{Cvoid}, Csize_t, Cint), Ptr{Cvoid}(a0), len, 14)         # MADV_HUGEPAGE (a hint)
    end

    return H
end

# ===== plans: repeated solves on one graph =====

"""
    plan = apsp_plan(A; semiring = MinPlus())
    D = apsp_gpu(plan, A₂; output = :device)       # A₂: the pattern of A, any weights
    apsp_gpu!(H, plan, A₂)                          # into a host matrix H (n × n), e.g. reused across calls

The closure of many matrices with one sparsity pattern (one graph, changing weights). Everything that
depends only on the pattern is done once and kept on the GPU: the symbolic factorization, the
factorization plan, the solve's structure, schedules and merge maps. Each call then copies the weights,
factorizes (the GPU part replayed as a CUDA graph after the first call), refreshes the solve's factor,
recomputes the dense operators, and runs the closure. Results are as `apsp_gpu(A₂)`'s.

The plan holds the factorization and the solver on the GPU until it is garbage collected (or
`free_plan!(plan)` is called). A matrix with another pattern is an `ArgumentError`.
"""
mutable struct APSPPlan{Sem <: AbstractSemiring, T, I}
    s::Sem
    pattern::UInt                   # hash of the pattern of A (colptr, rowval)
    F::ChordalSLU{Sem, T, I}
    P::FactorPlan{Sem, T, I}
    G::Union{Nothing, GPUSLU{Sem, T, I}}
    amal::Any                       # the solve's merge (its maps refresh the solve's factor), or nothing
end

pattern_hash(A::SparseMatrixCSC) = hash((size(A), A.colptr, A.rowval))

function apsp_plan(A::SparseMatrixCSC{T}; semiring::AbstractSemiring = Semiring.MinPlus()) where {T}
    check_apsp(A, semiring, :device)
    Q, S = @step "symbolic" Semiring.ssymbolic(A; alg = elimination_algorithm())
    F = @step "factor storage" ChordalSLU(semiring, T, S, Q.perm, Q.invp, Q.perm, Q.invp)
    check_coupling(F)
    P = @step "plan" FactorPlan(F; large = 256, graph = true, nstreams = 8)
    return APSPPlan(semiring, pattern_hash(A), F, P, nothing, nothing)
end

function apsp_gpu(plan::APSPPlan{<:Any, T}, A::SparseMatrixCSC{T}; output::Symbol = :device, buffer::Integer = 0) where {T}
    check_apsp(A, plan.s, output)
    H = output === :host ? host_matrix(T, size(A)...) : nothing
    return without_early_gc(() -> apsp_planned(plan, A, H, buffer))
end

"""
    apsp_gpu!(H, A; semiring = MinPlus())
    apsp_gpu!(H, plan, A)

As `apsp_gpu(A; output = :host)` (or with a plan), into the n × n host matrix `H`. Reusing `H` across
calls saves the page faults of a fresh matrix, which can cost more than the closure itself.
"""
function apsp_gpu!(H::Matrix{T}, A::SparseMatrixCSC{T}; semiring::AbstractSemiring = Semiring.MinPlus(), buffer::Integer = 0) where {T}
    check_apsp(A, semiring, :host)
    size(H) == size(A) || throw(DimensionMismatch("apsp_gpu!: H is $(size(H)), A is $(size(A))"))
    return without_early_gc(() -> apsp_single(A, semiring, :host, buffer; H))
end

function apsp_gpu!(H::Matrix{T}, plan::APSPPlan{<:Any, T}, A::SparseMatrixCSC{T}; buffer::Integer = 0) where {T}
    check_apsp(A, plan.s, :host)
    size(H) == size(A) || throw(DimensionMismatch("apsp_gpu!: H is $(size(H)), A is $(size(A))"))
    return without_early_gc(() -> apsp_planned(plan, A, H, buffer))
end

function apsp_planned(plan::APSPPlan{<:Any, T}, A::SparseMatrixCSC{T}, H, buffer::Integer) where {T}
    pattern_hash(A) == plan.pattern || throw(ArgumentError("apsp_gpu: A does not have the pattern of the plan's matrix; make a new plan"))
    n = size(A, 1)
    iszero(n) && return isnothing(H) ? CuMatrix{T}(undef, 0, 0) : H
    advice = "split the rows over several GPUs (apsp_gpu(A; devices = [...])) or compute blocks of rows (apsp_gpu(A, sources))"
    need_apsp_memory(n^2 * sizeof(T), "the $n × $n closure", advice)
    @step "copy entries" copyto!(plan.F, A)
    @step "factorize" factorize!(plan.P)

    if isnothing(plan.G)
        plan.G = @step "solve setup" GPUSLU(plan.P; large = 8192)
        plan.amal = solve_amalgamation(plan.F)
    else
        @step "refresh factor" refresh_factor!(plan)
    end

    G = plan.G
    @step "operators" (free_ops!(G); precompute_ops!(G))
    need_apsp_memory((n^2 + n * G.maxna) * sizeof(T), "the $n × $n closure and its workspace", advice)
    D = closure_labels(G, buffer, () -> nothing)
    return deliver(D, H)
end

# the merge GPUSLU(P) made for the solve (as GPUSLU's constructor calls it: a hit in the merge cache)
function solve_amalgamation(F::ChordalSLU{<:Any, <:Any, I}) where {I}
    config().merge > 1 || return nothing
    S = F.S.S; n = size(F, 1); nf = Int(MF.nv(S.res))
    Rptr = Vector{I}(view(MF.pointers(S.res), 1:(nf + 1))); Sptr = Vector{I}(view(MF.pointers(S.sep), 1:(nf + 1)))
    return amalgamation(amalgamation_key(S.Dptr), I, config().merge, config().merge_alpha, Rptr, Sptr,
        Vector{I}(view(MF.targets(S.sep), 1:(Sptr[end] - 1))), Vector{I}(view(S.Dptr, 1:(nf + 1))),
        Vector{I}(view(S.Lptr, 1:(nf + 1))), Vector{I}(view(S.pnt, 1:nf)), Vector{I}(view(S.idx, 1:n)))
end

# the solve's factor from the new factorization: shared with the plan when the solve does not merge
# fronts, else gathered again through the merge's maps (as amalgamate_values)
function refresh_factor!(plan::APSPPlan{<:Any, T}) where {T}
    G = plan.G; P = plan.P
    G.LDval === P.LD && return
    A = plan.amal; z = szero(plan.s, T, Val(:N))
    amalgamate_gather!(G.LDval, P.LD, P.LL, A.mLD, z); amalgamate_gather!(G.LLval, P.LL, P.LL, A.mLL, z)
    amalgamate_gather!(G.UDval, P.UD, P.UL, A.mUD, z); amalgamate_gather!(G.ULval, P.UL, P.UL, A.mUL, z)
    return
end

function free_plan!(plan::APSPPlan)
    isnothing(plan.G) || (plan.G.LDval === plan.P.LD || free_solver!(plan.G); free_ops!(plan.G))
    free_plan!(plan.P)
    plan.G = nothing
    return
end

# ===== multi-GPU =====

function apsp_multigpu(A::SparseMatrixCSC{T}, s::AbstractSemiring, devices::Vector{CuDevice}, output::Symbol, buffer::Integer) where {T}
    n = size(A, 1)
    ng = length(devices)
    iszero(n) && return output === :host ? Matrix{T}(undef, 0, 0) : Tuple{Vector{Int}, CuMatrix{T}}[]
    rows = row_blocks(n, ng)
    #
    # every device must hold its blocks (a repeated device holds several)
    #
    for d in unique(devices)
        k = sum(length(rows[g]) for g in 1:ng if devices[g] == d)

        CUDA.device!(d) do
            need_apsp_memory(k * n * sizeof(T), "$k rows of the $n × $n closure", "use more GPUs or compute blocks of rows (apsp_gpu(A, sources))")
        end
    end

    H = output === :host ? host_matrix(T, n, n) : nothing
    P = CUDA.device!(() -> apsp_factor(s, A), first(devices))
    p = Vector{Int}(P.F.rperm)
    MG = MultiGPUSLU(P; devices)  # copies the factor to every device, through the host
    CUDA.device!(() -> free_plan!(P), first(devices))
    #
    #   D[rows_g, :] ← A*[p[rows_g], p]   on devices[g]
    #
    blocks = closure_multigpu!(MG)
    #
    #   D[rows_g, :] ← D[rows_g, q]       (pass 1 on each device), and for :host  H[p[rows_g], :] ← D[rows_g, :]
    #
    @sync for g in 1:ng
        Threads.@spawn begin
            CUDA.device!(devices[g])
            G = MG.parts[g]
            r, X = blocks[g]
            free_solver!(G)
            k = size(X, 1)
            len = relabel_length(T, k, n, buffer)

            if ispositive(k)
                W = CuVector{T}(undef, len)
                relabel_cols!(X, G.cinvp, W)
                CUDA.unsafe_free!(W)
            end

            if !isnothing(H)
                scatter_rows!(H, p[r], X, len)
                CUDA.unsafe_free!(X)
            end

            CUDA.synchronize()
        end
    end

    isnothing(H) || return H
    return [(p[r], X) for (r, X) in blocks]
end

# ===== pipeline and checks =====

function check_apsp(A::SparseMatrixCSC{T}, s::AbstractSemiring, output::Symbol) where {T}
    output in (:device, :host) || throw(ArgumentError("apsp_gpu: output must be :device or :host, not $(repr(output))"))

    if size(A, 1) != size(A, 2)
        throw(ArgumentError("apsp_gpu: A must be square (the weighted adjacency matrix of a graph), not $(size(A, 1)) × $(size(A, 2))"))
    end

    if !(isbitstype(T) && sizeof(T) in (4, 8))
        throw(ArgumentError("apsp_gpu: element type $T is not supported on the GPU (the atomic ⊕ needs 4- or 8-byte isbits elements, e.g. Float32 or Float64)"))
    end

    check_semiring(s, T)
    return
end

#
# Symbolic phase and hybrid numeric factorization (bench/portable.jl's settings). The coupling check
# of GPUSLU, as an ArgumentError before the numeric work.
#
function check_coupling(F::ChordalSLU)
    if ispositive(MF.ne(F.S.N))
        throw(ArgumentError("apsp_gpu: coupling between strongly connected components (a directed graph whose components " *
            "reach one another) is not supported on the GPU yet; use the CPU solver, CliqueTrees.Multifrontal.Semiring.mstar(semiring, A)"))
    end

    return
end

#
# Host memory outside Julia's heap (Libc.malloc), for the host copy of the factor in a one-shot call:
# the factor lives through the call, so on the heap the collections the call triggers promoted it,
# and a full collection of the session's heap (~65 ms on HPG) was later needed to reclaim it. Off the
# heap the collector neither counts nor scans it; with_host_buffers frees it when the call returns
# (nothing reads the host factor after factorize!: the solve uses the device copy).
#
struct HostBuffers
    ptrs::Vector{Ptr{Cvoid}}        # blocks from Libc.malloc, freed when the call returns
    arena::Bool                     # the call holds the pinned arena (below)
    used::Base.RefValue{Int}        # bytes taken from the arena
    want::Base.RefValue{Int}        # bytes asked for in all
end

#
# A pinned host arena for those buffers: uploads from pinned memory run at the bus rate (pageable
# copies are staged by the driver), and memory reused across calls has no page faults to take. Pinning
# is slow (~3 GB/s on the HPG hosts), so the arena is made once, grown only when a call needed more (the
# next call then fits), and kept. One call at a time holds it; a call that finds it busy, or too small,
# uses malloc'd memory as before.
#
mutable struct PinnedArena
    mem::Any                        # CUDA host memory (portable: pinned for every context), or nothing
    size::Int
    busy::Bool
    off::Bool                       # pinning failed once: do not try again
    grow::Any                       # the task growing it (after the call that needed more), or nothing
end

const ARENA = PinnedArena(nothing, 0, false, false, nothing)
const ARENA_LOCK = ReentrantLock()

function with_host_buffers(f)
    arena = lock(() -> (ARENA.busy || ARENA.off) ? false : (ARENA.busy = true), ARENA_LOCK)
    B = HostBuffers(Ptr{Cvoid}[], arena, Ref(0), Ref(0))

    try
        return f(B)
    finally
        foreach(Libc.free, B.ptrs)
        empty!(B.ptrs)

        if arena && B.want[] > ARENA.size   # grown off the call's path (it stays busy until then: a call meanwhile uses malloc)
            want = B.want[]; dev = CUDA.device()
            ARENA.grow = Threads.@spawn try CUDA.device!(dev); grow_arena!(want) finally lock(() -> (ARENA.busy = false), ARENA_LOCK) end
        elseif arena
            lock(() -> (ARENA.busy = false), ARENA_LOCK)
        end
    end
end

#   wait for a growing arena (before freeing or dropping it)
function settle_arena()
    t = ARENA.grow
    isnothing(t) || try wait(t) catch end
    ARENA.grow = nothing
    return
end

function grow_arena!(bytes::Integer)
    C = CUDA.CUDACore
    isnothing(ARENA.mem) || C.free(ARENA.mem)
    ARENA.mem = nothing; ARENA.size = 0
    size = cld(bytes + bytes ÷ 4, 2^21) * 2^21

    try
        ARENA.mem = C.alloc(C.HostMemory, size, C.MEMHOSTALLOC_PORTABLE)
        ARENA.size = size
    catch err
        ARENA.off = true
        @warn "apsp_gpu: could not pin $(Base.format_bytes(size)) of host memory; using pageable memory" exception = err maxlog = 1
    end

    return
end

function host_vector(B::HostBuffers, ::Type{T}, n::Integer) where {T}
    iszero(n) && return Vector{T}(undef, 0)
    bytes = cld(n * sizeof(T), 64) * 64
    B.want[] += bytes

    if B.arena && B.used[] + bytes <= ARENA.size
        p = pointer(ARENA.mem) + B.used[]
        B.used[] += bytes
        return unsafe_wrap(Array, Ptr{T}(p), n; own = false)
    end

    p = Libc.malloc(n * sizeof(T))
    p == C_NULL && throw(OutOfMemoryError())
    push!(B.ptrs, p)
    return unsafe_wrap(Array, Ptr{T}(p), n; own = false)
end

# ChordalSLU(s, T, S, perm...) with its value arrays in B
function host_slu(B::HostBuffers, s::AbstractSemiring, ::Type{T}, S, Q) where {T}
    nD = MF.ndz(S.S); nL = MF.nlz(S.S)
    LD = host_vector(B, T, nD); LL = host_vector(B, T, nL)
    UD = host_vector(B, T, nD); UL = host_vector(B, T, nL)
    N = host_vector(B, T, Int(MF.ne(S.N)))
    return ChordalSLU(s, S, LD, LL, UD, UL, N, Q.perm, Q.invp, Q.perm, Q.invp)
end

function apsp_factor(s::AbstractSemiring, A::SparseMatrixCSC{T}; host::Union{Nothing, HostBuffers} = nothing,
        solve::Union{Nothing, Integer} = nothing) where {T}
    # ChordalSLU(s, A), in its two parts: the symbolic factorization (ordering, elimination tree,
    # supernodes, structure) and the storage of the factor (in host buffers when given: see HostBuffers)
    Q, S = @step "symbolic" Semiring.ssymbolic(A; alg = elimination_algorithm())
    F = @step "factor storage" (isnothing(host) ? ChordalSLU(s, T, S, Q.perm, Q.invp, Q.perm, Q.invp) : host_slu(host, s, T, S, Q))

    check_coupling(F)

    # solve = the `large` of the solve to come: its structure (solve_structure) needs only the symbolic
    # factorization, so it is made on another thread while the factor is computed; returns (P, task).
    # Under the step timer it is left to GPUSLU, which times its steps.
    st = isnothing(solve) || !isnothing(STEPS[]) ? nothing : structure_async(F, solve)
    P = try
        factor_plan(F, A)
    catch
        st isa Task && try wait(st) catch end
        rethrow()
    end

    return isnothing(solve) ? P : (P, st)
end

function structure_async(F::ChordalSLU, large::Integer)
    dev = CUDA.device()

    return Threads.@spawn begin
        CUDA.device!(dev)
        st = solve_structure(F; large)
        CUDA.synchronize()                              # (the merge maps are made on this task's stream)
        st
    end
end

function factor_plan(F::ChordalSLU, A::SparseMatrixCSC)
    # the plan reads only the structure of the factor, so the entries of A are copied in meanwhile
    # (one after the other under the step timer, which times each on its own). The copy is waited
    # for even when the plan fails: it writes into F, whose host buffers the caller then frees.
    if isnothing(STEPS[])
        entries = Threads.@spawn copyto!(F, A)

        P = try
            FactorPlan(F; large = 256, graph = false, nstreams = 8)
        finally
            try wait(entries) catch end
        end

        fetch(entries)
    else
        @step "copy entries" copyto!(F, A)
        P = @step "plan" FactorPlan(F; large = 256, graph = false, nstreams = 8)
    end

    @step "factorize" factorize!(P)
    return P
end

structure_of(::Nothing) = nothing
structure_of(t::Task) = fetch_result(t)

#
# Device memory an allocation can get: free memory plus what CUDA.jl's pool holds but does not use (an
# earlier call's n × n result, freed into the pool, is there). Counting only free memory made every call
# after the first see a shortage and run a full collection of the session's heap (~50–220 ms).
#
function available_memory()
    free = CUDA.free_memory()
    cached = CUDA.cached_memory(); used = CUDA.used_memory()
    (ismissing(cached) || ismissing(used)) && return free
    return free + max(0, cached - used)
end

# memory that CUDA.jl's pool holds but does not use: an allocation up to this size needs no new device memory
function pool_spare()
    cached = CUDA.cached_memory(); used = CUDA.used_memory()
    (ismissing(cached) || ismissing(used)) && return 0
    return max(0, cached - used)
end

#
# Run f with CUDA.jl's early collections off: on every synchronize that has to wait, CUDA.jl collects
# garbage when the live device memory passes half of it. An apsp_gpu call keeps its n × n result (often
# that much) live and frees its arrays itself, so those collections (30–90 ms each) free nothing.
# Collections on an allocation that does not fit still happen.
#
function without_early_gc(f)
    C = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA
    isdefined(C, :_early_gc) || return f()
    r = getfield(C, :_early_gc)
    r isa Base.RefValue{Union{Nothing, Bool}} || return f()
    prev = r[]; r[] = false

    try
        return f()
    finally
        r[] = prev
    end
end

#
# need_memory, with advice for apsp_gpu's callers. When short, a full collection: the views a solve
# takes of its matrices keep them allocated until they are collected, and they may have aged past a
# young collection (an n × n result freed by an earlier call would otherwise still count as used).
#
function need_apsp_memory(bytes::Integer, what::AbstractString, advice::AbstractString)
    free = available_memory()

    if bytes > free                     # memory still held by unreachable arrays: collect them and look again
        GC.gc(true); CUDA.reclaim()
        free = available_memory()
    end

    if bytes > free
        error("apsp_gpu: not enough GPU memory for $what on $(CUDA.name(CUDA.device())): needs $(round(bytes / 2^30; digits = 2)) GiB, " *
              "$(round(free / 2^30; digits = 2)) GiB available; $advice")
    end

    return
end

#
# Return the GPU memory of a solve structure's factor and operators, and of a factorization plan's
# buffers, now instead of at the next garbage collection. Nothing may use them afterwards. (Memory
# still referenced by views taken during the solve is returned when those are collected, which
# CUDA.jl does by itself when an allocation would not fit.)
#
function free_solver!(G::GPUSLU)
    foreach(CUDA.unsafe_free!, (G.LDval, G.LLval, G.UDval, G.ULval))
    free_ops!(G)
    return
end

# the precomputed operators (they depend on the factor's values)
function free_ops!(G::GPUSLU)
    ops = G.ops[]

    if !isnothing(ops)
        foreach(CUDA.unsafe_free!, (ops.KL, ops.KU, ops.work[]))
        foreach(CUDA.unsafe_free!, values(ops.cols))
        G.ops[] = nothing
    end

    return
end

# (the merge index maps P.amal belong to the amalgamation cache and are kept)
function free_plan!(P::FactorPlan)
    foreach(CUDA.unsafe_free!, (P.LD, P.LL, P.UD, P.UL, P.Mb, P.Mg, P.Fg, P.relptr, P.reltgt, P.Fbufs..., P.merged...))
    return
end

# ===== relabelling in place =====

#
# Elements of the relabelling buffer for an m × n matrix: at most the matrix, 5% of the GPU memory an
# allocation can get (available_memory: free, or held unused by CUDA.jl's pool; counting only the free
# memory gave a call after an earlier one of the same size a buffer too small for the cycle path, and
# the row-block path through a tiny buffer took 13 s for grid2d-380 instead of 0.14 s) and 1 GiB, and at
# least one row and one column. `buffer > 0` forces a length (tests).
#
function relabel_length(::Type{T}, m::Integer, n::Integer, buffer::Integer) where {T}
    len = ispositive(buffer) ? Int(buffer) : min(m * n, available_memory() ÷ (20 * sizeof(T)), 2^30 ÷ sizeof(T))
    return max(len, m, n, 1)
end

#
#   X[:, j] ← X[:, q[j]], into a new matrix when the pool holds the memory (one coalesced pass, X is
#   freed), else in place through a buffer (relabel_cols!). `buffer > 0` forces the in-place path with
#   that buffer (tests).
#
function relabel_cols(X::CuMatrix{T}, q::CuVector, buffer::Integer) where {T}
    m, n = size(X)
    (iszero(m) || iszero(n)) && return X

    # Out of place only when CUDA.jl's pool already holds the memory: new device memory costs ~1.5 ms per
    # GB on a B200 (the driver clears it), so for a call that has to allocate it (the first call, as in
    # a process that runs once) the in-place pass through a buffer, one more read and write of X, is
    # cheaper: luxembourg_osm (52 GB) spent 80 ms more allocating than relabelling. It also halves the
    # peak memory of the call.
    if !ispositive(buffer) && m * n * sizeof(T) <= pool_spare()
        Y = CuMatrix{T}(undef, m, n)
        gather_cols!(vec(Y), X, 0, m, q)                                    # Y ← X[:, q]
        CUDA.unsafe_free!(X)
        return Y
    end

    len = relabel_length(T, m, n, buffer)
    !ispositive(buffer) && relabel_cycles!(X, Array(q), len) && return X

    W = CuVector{T}(undef, len)
    relabel_cols!(X, q, W)
    CUDA.unsafe_free!(W)
    return X
end

#
#   X[:, j] ← X[:, q[j]] in place, following the cycles of q: the cycle c₁, c₂ = q[c₁], … is cut into
#   segments of at most L columns; first the column after each segment (the next segment's first, or c₁)
#   is saved, then every segment moves its columns down one, X[:, c_t] ← X[:, c_{t+1}], and its last
#   column takes the saved one. One read and one write of every moved column, each a contiguous run
#   (the row-block path, relabel_cols!, reads and writes X twice, in short runs of every column: 4× the
#   traffic of luxembourg_osm's 52 GB at a third of the bandwidth). The saved columns take at most `len`
#   elements; returns false, doing nothing, when there are more cycles than that (many short cycles).
#
function relabel_cycles!(X::CuMatrix{T}, q::Vector{<:Integer}, len::Integer) where {T}
    m, n = size(X)
    seen = falses(n); moved = 0; cycles = 0

    for j in 1:n                                        # the cycles (fixed points stay)
        (seen[j] || q[j] == j) && continue
        c = j

        while !seen[c]
            seen[c] = true; moved += 1; c = q[c]
        end

        cycles += 1
    end

    iszero(cycles) && return true
    cols = len ÷ m                                      # columns the saved ones may take
    cycles >= cols && return false
    L = max(16, cld(moved, cols - cycles))              # segments: Σ cld(k, L) ≤ moved / L + cycles ≤ cols
    fill!(seen, false); pos = Int32[]; segptr = Int32[1]; nxt = Int32[]

    for j in 1:n
        (seen[j] || q[j] == j) && continue
        c = j; t = 0

        while !seen[c]
            seen[c] = true; push!(pos, c); t += 1; c = q[c]

            if t == L && !seen[c]                       # the segment is full and the cycle goes on
                push!(segptr, length(pos) + 1); push!(nxt, c); t = 0
            end
        end

        push!(segptr, length(pos) + 1); push!(nxt, j)   # (c = j: the last segment wraps to c₁)
    end

    @assert length(nxt) <= cols
    W = CuMatrix{T}(undef, m, length(nxt)); dpos, dptr, dnxt = CuVector(pos), CuVector(segptr), CuVector(nxt)
    bx, by = copy_grid(m, length(nxt))
    @cuda threads = 256 blocks = (bx, by) save_columns_kernel!(W, X, dnxt)               # W[:, s] ← X[:, nxt[s]]
    @cuda threads = 256 blocks = (bx, by) shift_segments_kernel!(X, W, dpos, dptr)
    foreach(CUDA.unsafe_free!, (W, dpos, dptr, dnxt))
    return true
end

function save_columns_kernel!(W, X, nxt)
    m = size(X, 1)
    r = (blockIdx().x - 1) * (256 * COPY_ROWS) + threadIdx().x
    s = blockIdx().y

    @inbounds while s <= length(nxt)
        src = (nxt[s] - 1) * m; dst = (s - 1) * m
        Base.Cartesian.@nexprs 8 k -> (v_k = r + (k - 1) * 256 <= m ? X[src + r + (k - 1) * 256] : zero(eltype(X)))
        Base.Cartesian.@nexprs 8 k -> (r + (k - 1) * 256 <= m && (W[dst + r + (k - 1) * 256] = v_k))
        s += gridDim().y
    end

    return
end

#   segment s (columns pos[segptr[s]:segptr[s + 1] - 1]): X[:, pos[t]] ← X[:, pos[t + 1]], the last ← W[:, s].
#   Each thread reads every column's rows before writing them (program order), so no barrier is needed.
function shift_segments_kernel!(X, W, pos, segptr)
    m = size(X, 1)
    r = (blockIdx().x - 1) * (256 * COPY_ROWS) + threadIdx().x
    s = blockIdx().y

    @inbounds while s < length(segptr)
        e = segptr[s + 1] - 1
        t = segptr[s]

        while t <= e
            dst = (pos[t] - 1) * m
            src = t < e ? (pos[t + 1] - 1) * m : (s - 1) * m
            Y = t < e ? X : W
            Base.Cartesian.@nexprs 8 k -> (v_k = r + (k - 1) * 256 <= m ? Y[src + r + (k - 1) * 256] : zero(eltype(X)))
            Base.Cartesian.@nexprs 8 k -> (r + (k - 1) * 256 <= m && (X[dst + r + (k - 1) * 256] = v_k))
            t += 1
        end

        s += gridDim().y
    end

    return
end

#
#   X[:, j] ← X[:, q[j]]   for every column (q a permutation of 1:n), b = length(W) ÷ n rows at a time
#
function relabel_cols!(X::CuMatrix{T}, q::CuVector, W::CuVector{T}) where {T}
    m, n = size(X)
    (iszero(m) || iszero(n)) && return X
    b = min(m, length(W) ÷ n)
    @assert b >= 1

    for i0 in 0:b:(m - 1)
        bi = min(b, m - i0)
        gather_cols!(W, X, i0, bi, q)                                  # W ← X[I, q]
        put_rows!(X, W, i0, bi)                                        # X[I, :] ← W
    end

    return X
end

#
#   Y[i, :] ← X[q[i], :]   for every row (q a permutation of 1:m), c = length(W) ÷ m columns at a time.
#
# Y is X itself (in place) or a host matrix of the same size, into which each block of columns
# (contiguous in both) is copied straight from W.
#
function relabel_rows!(Y::Union{CuMatrix{T}, Matrix{T}}, X::CuMatrix{T}, q::CuVector, W::CuVector{T}) where {T}
    m, n = size(X)
    @assert size(Y) == (m, n)
    (iszero(m) || iszero(n)) && return Y
    c = min(n, length(W) ÷ m)
    @assert c >= 1

    for j0 in 0:c:(n - 1)
        cj = min(c, n - j0)
        launch2d(gather_rows_kernel!, m, cj, W, X, j0, cj, q)          # W ← X[q, J]
        copyto!(Y, j0 * m + 1, W, 1, m * cj)                           # Y[:, J] ← W
    end

    return Y
end

#
#   H[src[t], :] ← X[t, :]   (X on the GPU): X downloaded c = len ÷ k columns at a time
#
function scatter_rows!(H::Matrix{T}, src::Vector{Int}, X::CuMatrix{T}, len::Integer) where {T}
    k, n = size(X)
    (iszero(k) || iszero(n)) && return H
    c = min(n, len ÷ k)
    hb = Vector{T}(undef, k * c)

    for j0 in 0:c:(n - 1)
        cj = min(c, n - j0)
        copyto!(hb, 1, X, j0 * k + 1, k * cj)                          # X[:, J] is contiguous

        @inbounds for t in 1:cj, i in 1:k
            H[src[i], j0 + t] = hb[i + (t - 1) * k]
        end
    end

    return H
end

# W[t + (j - 1) b] ← X[i₀ + t, q[j]]   (t ≤ b: rows i₀+1:i₀+b of X, their columns gathered)
#
# The column copies of the relabelling move the whole n × n result, so they run at the bandwidth of
# the device only with few, long-lived blocks: a block copies COPY_ROWS × 256 rows of a column (8
# loads in flight per thread), then the same rows of every gridDim().y-th column. (One block per 256
# entries of a column made 51 million blocks for n = 115k, which ran at a third of the bandwidth.)
#
const COPY_ROWS = 8
const COPY_BLOCKS = 2^15

function copy_grid(b::Integer, n::Integer)
    bx = cld(b, 256 * COPY_ROWS)
    return bx, min(n, max(1, cld(COPY_BLOCKS, bx)))
end

#   W[t + (j - 1) b] ← X[i₀ + t, q[j]]   (t ≤ b, every column j)
function gather_cols!(W, X, i0::Integer, b::Integer, q)
    n = size(X, 2)
    (iszero(b) || iszero(n)) && return
    @cuda threads = 256 blocks = copy_grid(b, n) gather_cols_kernel!(W, X, Int(i0), Int(b), q)
    return
end

function gather_cols_kernel!(W, X, i0, b, q)
    m = size(X, 1)
    r = (blockIdx().x - 1) * (256 * COPY_ROWS) + threadIdx().x
    j = blockIdx().y

    @inbounds while j <= size(X, 2)
        src = (q[j] - 1) * m + i0; dst = (j - 1) * b
        Base.Cartesian.@nexprs 8 k -> (v_k = r + (k - 1) * 256 <= b ? X[src + r + (k - 1) * 256] : zero(eltype(X)))
        Base.Cartesian.@nexprs 8 k -> (r + (k - 1) * 256 <= b && (W[dst + r + (k - 1) * 256] = v_k))
        j += gridDim().y
    end

    return
end

#   X[i₀ + t, j] ← W[t + (j - 1) b]
function put_rows!(X, W, i0::Integer, b::Integer)
    n = size(X, 2)
    (iszero(b) || iszero(n)) && return
    @cuda threads = 256 blocks = copy_grid(b, n) put_rows_kernel!(X, W, Int(i0), Int(b))
    return
end

function put_rows_kernel!(X, W, i0, b)
    m = size(X, 1)
    r = (blockIdx().x - 1) * (256 * COPY_ROWS) + threadIdx().x
    j = blockIdx().y

    @inbounds while j <= size(X, 2)
        src = (j - 1) * b; dst = (j - 1) * m + i0
        Base.Cartesian.@nexprs 8 k -> (v_k = r + (k - 1) * 256 <= b ? W[src + r + (k - 1) * 256] : zero(eltype(W)))
        Base.Cartesian.@nexprs 8 k -> (r + (k - 1) * 256 <= b && (X[dst + r + (k - 1) * 256] = v_k))
        j += gridDim().y
    end

    return
end

# W[i + (t - 1) m] ← X[q[i], j₀ + t]   (t ≤ c: columns j₀+1:j₀+c of X, their rows gathered)
function gather_rows_kernel!(W, X, j0, c, q)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    t = blockIdx().y
    m = size(X, 1)

    if i <= m
        @inbounds while t <= c
            W[i + (t - 1) * m] = X[q[i], j0 + t]
            t += gridDim().y
        end
    end

    return
end
