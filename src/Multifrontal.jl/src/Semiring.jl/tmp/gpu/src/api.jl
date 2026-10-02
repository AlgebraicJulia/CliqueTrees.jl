# ===== apsp_gpu: the closure in the original labels =====
#
# One call for the whole pipeline, with the tuned settings of bench/portable.jl:
#
#   F = ChordalSLU(s, A); copyto!(F, A)                            symbolic phase (CPU), entries of A
#   P = FactorPlan(F; large = 256, graph = false, nstreams = 8)     hybrid numeric LU: bottom subtrees on
#   factorize!(P)                                                    CPU threads, the top fronts on the GPU
#   G = GPUSLU(P; large = 8192); precompute_ops!(G)                 solve structure and the large fronts' dense operators
#   closure_gpu!(D, G; M)                                            D[i, j] = A*[p[i], p[j]],  p = rperm
#   D ← D[q, q],  q = p⁻¹ = cinvp                                    D[i, j] = A*[i, j]
#
# The symbolic phase permutes A symmetrically (rperm = cperm), so one permutation relabels both the
# rows and the columns. D may fill most of the GPU, so it is relabelled in place through a buffer W of
# b rows (or columns), as permuterows! / permutecols! do on the CPU:
#
#   pass 1, rows I, b at a time:       W ← D[I, q];  D[I, :] ← W       now D[i, j] = A*[p[i], j]
#   pass 2, columns J, b at a time:    W ← D[q, J];  D[:, J] ← W       now D[i, j] = A*[i, j]
#
# With output = :host, pass 2 copies W straight into the host matrix instead of back into D.
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
        output::Symbol = :device, buffer::Integer = 0)
    #
    # buffer (internal, for tests): elements of the relabelling buffer, 0 = automatic (see relabel_length)
    #
    check_apsp(A, semiring, output)
    isempty(devices) && throw(ArgumentError("apsp_gpu: no devices given"))
    all(d -> d isa CuDevice, devices) || throw(ArgumentError("apsp_gpu: devices must be CuDevices, e.g. collect(CUDA.devices())"))

    if length(devices) > 1
        return apsp_multigpu(A, semiring, collect(CuDevice, devices), output, buffer)
    end

    return CUDA.device!(() -> apsp_single(A, semiring, output, buffer), first(devices))
end

function apsp_gpu(A::SparseMatrixCSC{T}, sources::AbstractVector{<:Integer}; semiring::AbstractSemiring = Semiring.MinPlus(),
        output::Symbol = :device) where {T}
    check_apsp(A, semiring, output)
    n = size(A, 1)
    k = length(sources)
    bad = findfirst(v -> !(1 <= v <= n), sources)
    isnothing(bad) || throw(ArgumentError("apsp_gpu: sources must be vertices in 1:$n, but sources[$bad] = $(sources[bad])"))

    if iszero(k)
        return output === :host ? Matrix{T}(undef, 0, n) : CuMatrix{T}(undef, 0, n)
    end

    advice = "use fewer sources per call"
    need_apsp_memory(2 * k * n * sizeof(T), "$k rows of the closure and their workspace", advice)       # before any work
    P = apsp_factor(semiring, A)
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
    output === :device && return X
    H = Array(X)
    CUDA.unsafe_free!(X)
    return H
end

# ===== single GPU =====

function apsp_single(A::SparseMatrixCSC{T}, s::AbstractSemiring, output::Symbol, buffer::Integer) where {T}
    n = size(A, 1)
    iszero(n) && return output === :host ? Matrix{T}(undef, 0, 0) : CuMatrix{T}(undef, 0, 0)
    advice = "split the rows over several GPUs (devices = [...]) or compute blocks of rows (apsp_gpu(A, sources))"
    need_apsp_memory(n^2 * sizeof(T), "the $n × $n closure", advice)                                  # before any work
    H = output === :host ? Matrix{T}(undef, n, n) : nothing
    P = apsp_factor(s, A)
    G = GPUSLU(P; large = 8192)
    precompute_ops!(G)
    need_apsp_memory((n^2 + n * G.maxna) * sizeof(T), "the $n × $n closure and its workspace", advice)
    D = CuMatrix{T}(undef, n, n)
    M = CuMatrix{T}(undef, n, G.maxna)
    #
    #   D ← A*, in elimination coordinates
    #
    closure_gpu!(D, G; M)
    CUDA.synchronize()
    q = G.cinvp
    CUDA.unsafe_free!(M)
    free_solver!(G); free_plan!(P)                     # before the buffer, which may then be larger
    #
    #   D ← D[q, q]
    #
    W = CuVector{T}(undef, relabel_length(T, n, n, buffer))
    relabel_cols!(D, q, W)
    relabel_rows!(isnothing(H) ? D : H, D, q, W)
    CUDA.unsafe_free!(W)
    isnothing(H) && return D
    CUDA.unsafe_free!(D)
    return H
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

    H = output === :host ? Matrix{T}(undef, n, n) : nothing
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
function apsp_factor(s::AbstractSemiring, A::SparseMatrixCSC)
    F = ChordalSLU(s, A)

    if ispositive(MF.ne(F.S.N))
        throw(ArgumentError("apsp_gpu: coupling between strongly connected components (a directed graph whose components " *
            "reach one another) is not supported on the GPU yet; use the CPU solver, CliqueTrees.Multifrontal.Semiring.mstar(semiring, A)"))
    end

    copyto!(F, A)
    P = FactorPlan(F; large = 256, graph = false, nstreams = 8)
    factorize!(P)
    return P
end

#
# need_memory, with advice for apsp_gpu's callers. When short, a full collection: the views a solve
# takes of its matrices keep them allocated until they are collected, and they may have aged past a
# young collection (an n × n result freed by an earlier call would otherwise still count as used).
#
function need_apsp_memory(bytes::Integer, what::AbstractString, advice::AbstractString)
    free = CUDA.free_memory()

    if bytes > free                     # memory cached by CUDA.jl's pool is not free: release it and look again
        GC.gc(true); CUDA.reclaim()
        free = CUDA.free_memory()
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
# Elements of the relabelling buffer for an m × n matrix: at most the matrix, 5% of the free GPU memory
# and 1 GiB, and at least one row and one column. `buffer > 0` forces a length (tests).
#
function relabel_length(::Type{T}, m::Integer, n::Integer, buffer::Integer) where {T}
    len = ispositive(buffer) ? Int(buffer) : min(m * n, CUDA.free_memory() ÷ (20 * sizeof(T)), 2^30 ÷ sizeof(T))
    return max(len, m, n, 1)
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
        launch2d(gather_cols_kernel!, bi, n, W, X, i0, bi, q)          # W ← X[I, q]
        launch2d(put_rows_kernel!, bi, n, X, W, i0, bi)                # X[I, :] ← W
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
function gather_cols_kernel!(W, X, i0, b, q)
    t = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = blockIdx().y

    if t <= b
        @inbounds while j <= size(X, 2)
            W[t + (j - 1) * b] = X[i0 + t, q[j]]
            j += gridDim().y
        end
    end

    return
end

# X[i₀ + t, j] ← W[t + (j - 1) b]
function put_rows_kernel!(X, W, i0, b)
    t = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = blockIdx().y

    if t <= b
        @inbounds while j <= size(X, 2)
            X[i0 + t, j] = W[t + (j - 1) * b]
            j += gridDim().y
        end
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
