# ===== layered L sweep =====
#
# rowmajor_down_kernel! gives each row one thread, which walks every front below the top of the tree:
# a serial chain of ~nf dependent gathers per thread, and only k threads. On a large GPU that is a
# few percent of the resident thread slots (22.5k rows against 303k slots on a B200), so the sweep is
# bound by the latency of that chain, not by bandwidth.
#
# The rows are independent, and so are disjoint subtrees within a row. This cuts the fronts below the
# top into layers of regions: the bottom layer holds the maximal subtrees with fewer than m fronts, the
# next layer the maximal such subtrees of what is left, and so on. A launch per layer, top layer first,
# gives every (row, region) pair a thread, which walks its region parents first. Every front's
# ancestors are in its own region (earlier in the walk) or in a layer above, so they are final when it
# runs. With m ≈ √(2 nf), a balanced tree needs two or three layers and each thread walks O(√nf)
# fronts instead of nf. Same operations per front as rowmajor_down_kernel!.
#
const LAYERED = Ref(true)
const LAYER_M = Ref(0)               # region size in fronts (0: √(2 nf))
const LAYER_MAXREG = 65535           # gridDim.y

struct LayerPlan{I}
    layers::Vector{Tuple{CuVector{I}, CuVector{I}, Int}}   # top-down: (region pointers, fronts, number of regions)
    m::Int
    maxchain::Int                                          # longest walk of one thread, in fronts
end

function layer_plan(G::GPUSLU{<:Any, <:Any, I}) where {I}
    m = LAYER_M[]
    get!(G.cache, Symbol(:layers, m)) do
        istop = Array(G.istop); pnt = Array(G.pnt)
        nf = G.nf
        alive = .!istop
        m = LAYER_M[] > 0 ? max(2, LAYER_M[]) : max(16, isqrt(2 * count(alive)))   # m = 1 would leave no region
        #
        # full subtree sizes, to find each subtree's postorder range [f - fsz[f] + 1, f]
        #
        fsz = ones(Int, nf)

        for f in 1:nf
            p = pnt[f]
            iszero(p) || (fsz[p] += fsz[f])
        end

        sz = zeros(Int, nf)
        layers = Tuple{Vector{I}, Vector{I}}[]

        while any(alive)
            fill!(sz, 0)

            for f in 1:nf
                alive[f] || continue
                sz[f] += 1
                p = pnt[f]
                (!iszero(p) && alive[p]) && (sz[p] += sz[f])
            end

            # region roots: alive, below m (or a root of what is left), parent not a region member
            final = all(f -> !alive[f] || sz[f] < m, 1:nf)
            isroot(f) = alive[f] && (final || sz[f] < m) && (iszero(pnt[f]) || !alive[pnt[f]] || (!final && sz[pnt[f]] >= m))
            regptr = I[1]; regfronts = I[]

            for r in 1:nf
                isroot(r) || continue

                for f in r:-1:(r - fsz[r] + 1)          # reverse postorder: parents first
                    alive[f] && push!(regfronts, f)
                end

                push!(regptr, length(regfronts) + 1)
            end

            for f in regfronts
                alive[f] = false
            end

            @assert !isempty(regfronts)
            push!(layers, (regptr, regfronts))
        end

        reverse!(layers)                                # execute top-down
        out = Tuple{CuVector{I}, CuVector{I}, Int}[]
        maxchain = 0

        for (regptr, regfronts) in layers
            nreg = length(regptr) - 1
            maxchain += maximum(diff(regptr))
            #
            # gridDim.y is at most 65535: merge neighbouring regions of a layer if needed
            #
            if nreg > LAYER_MAXREG
                step = cld(nreg, LAYER_MAXREG)
                regptr = regptr[1:step:end]
                regptr[end] == length(regfronts) + 1 || push!(regptr, length(regfronts) + 1)
                nreg = length(regptr) - 1
            end

            push!(out, (upload(regptr), upload(regfronts), nreg))
        end

        LayerPlan{I}(out, m, maxchain)
    end
end

function layered_down_kernel!(s::AbstractSemiring, trans::Val, C::AbstractMatrix{T}, regptr, regfronts,
        Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval) where {T}
    t = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    g = blockIdx().y

    if t <= size(C, 1)
        @inbounds for i in regptr[g]:(regptr[g + 1] - 1)
            f = regfronts[i]
            downward_front_reg!(s, trans, C, t, f, Rptr, Sptr, Stgt, Dptr, Lptr, Dval, Lval)
        end
    end

    return
end
