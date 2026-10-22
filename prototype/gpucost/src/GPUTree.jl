# GPU mass scoring of elimination orderings by the exact contraction cost C of the induced tree.
#
# For each job (one ordering of the network's indices):
#   perturb_kernel!  (one thread per job)  copy a base ordering and apply random insertion moves
#   build_kernel!    (one thread per job)  bucket elimination with union-find over tensors -> binary
#                    tree (same rule as Host.bucket_tree), then subtree sizes and preorder numbers
#   cost_kernel!     (one warp per job)    every index walks its Steiner tree (leaves holding it, plus the
#                    root if open; leaves sorted by preorder, each walk stops at the LCA with the previous
#                    leaf), adding its log2 dim to the flops exponent of every contraction it is involved in
#                    and to the size exponent of every node it is a leg of; then warp reductions give
#                    log2 C = log2 sum_p 2^acc[p], the width W = max leg, and the largest contraction.
# The cost is exactly cotengra's contraction_cost() of that tree (Float32 sums; verified with cotengra).
# Node ids are 0-based here: leaves 0..T-1, internal T..2T-2 (children before parents), root 2T-2.
module GPUTree

using CUDA

export GNet, gnet, Scratch, scratch, perturb!, build!, gcost!, score_orders!, fetch_tree

struct GNet
    n::Int32
    T::Int32
    nopen::Int32
    maxocc::Int32
    occptr::CuVector{Int32}   # 0-based offsets, n+1
    occ::CuVector{Int32}      # 0-based tensor ids
    w::CuVector{Float32}
    isopen::CuVector{UInt8}
end

function gnet(net)
    occptr = Int32.(net.occptr .- 1); occ = Int32.(net.occ .- 1)
    maxocc = maximum(diff(net.occptr))
    GNet(Int32(net.n), Int32(net.T), Int32(length(net.open)), Int32(maxocc), CuArray(occptr), CuArray(occ), CuArray(Float32.(net.w)), CuArray(UInt8.(net.isopen)))
end

struct Scratch
    J::Int
    orders::CuMatrix{Int32}    # (n, J) 0-based index ids
    left::CuMatrix{Int32}      # (N, J)
    right::CuMatrix{Int32}
    parent::CuMatrix{Int32}
    sz::CuMatrix{Int32}
    pre::CuMatrix{Int32}
    rep::CuMatrix{Int32}       # (T, J)
    cur::CuMatrix{Int32}
    roots::CuMatrix{Int32}     # (max(T, maxocc), J)
    sbuf::CuMatrix{Int32}      # (32*maxocc, J)
    acc::CuMatrix{Float32}     # (N, J)
    leg::CuMatrix{Float32}
    log2C::CuVector{Float32}
    W::CuVector{Float32}
    maxop::CuVector{Float32}
end

function scratch(g::GNet, J::Int)
    n = Int(g.n); T = Int(g.T); N = 2T - 1; mo = Int(g.maxocc)
    Z(r) = CUDA.zeros(Int32, r, J)
    Scratch(J, Z(n), Z(N), Z(N), Z(N), Z(N), Z(N), Z(T), Z(T), Z(max(T, mo)), Z(32mo),
        CUDA.zeros(Float32, N, J), CUDA.zeros(Float32, N, J), CUDA.zeros(Float32, J), CUDA.zeros(Float32, J), CUDA.zeros(Float32, J))
end

# ----------------------------------------------------------------------------- RNG
@inline function xorshift(s::UInt64)
    s ⊻= s << 13; s ⊻= s >> 7; s ⊻= s << 17
    s
end
@inline function randint(s::UInt64, m::Int32)   # -> (state, 0..m-1)
    s = xorshift(s)
    s, Int32((s >> 33) % UInt64(m))
end

# ----------------------------------------------------------------------------- perturb
function perturb_kernel!(orders, base, bidx, nmoves, maxd, seeds, n::Int32, neff::Int32, J::Int32)
    j = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j > J && return
    b = bidx[j]
    @inbounds for t in Int32(1):n
        orders[t, j] = base[t, b]
    end
    s = seeds[j] * 0x9E3779B97F4A7C15 + UInt64(j)
    s = xorshift(xorshift(s | UInt64(1)))
    md = maxd[j]
    @inbounds for _ in Int32(1):nmoves[j]
        s, p = randint(s, neff)
        s, d = randint(s, Int32(2) * md + Int32(1))
        q = p + d - md
        q = q < Int32(0) ? Int32(0) : q >= neff ? neff - Int32(1) : q
        x = orders[p+1, j]
        if q > p
            for t in p:q-Int32(1)
                orders[t+1, j] = orders[t+2, j]
            end
        else
            for t in p:-Int32(1):q+Int32(1)
                orders[t+1, j] = orders[t, j]
            end
        end
        orders[q+1, j] = x
    end
    return
end

"""orders[:, j] = base[:, bidx[j]] with nmoves[j] random insertion moves of distance <= maxd[j], all
within the first n - nopen positions (base orderings end with the open indices, see Host.open_last)."""
function perturb!(sc::Scratch, g::GNet, base::CuMatrix{Int32}, bidx::CuVector{Int32}, nmoves::CuVector{Int32}, maxd::CuVector{Int32}, seeds::CuVector{UInt64}, J::Int = length(bidx))
    @cuda threads = 128 blocks = cld(J, 128) perturb_kernel!(sc.orders, base, bidx, nmoves, maxd, seeds, g.n, g.n - g.nopen, Int32(J))
end

# ----------------------------------------------------------------------------- build
@inline function findroot!(rep, x::Int32, j)
    @inbounds while rep[x+1, j] != x
        y = rep[rep[x+1, j]+1, j]
        rep[x+1, j] = y; x = y
    end
    x
end

function build_kernel!(left, right, parent, sz, pre, rep, cur, roots, orders, occptr, occ, isopen, n::Int32, T::Int32, J::Int32)
    j = Int32((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    j > J && return
    N = Int32(2) * T - Int32(1)
    @inbounds begin
        for t in Int32(0):T-Int32(1)
            rep[t+1, j] = t; cur[t+1, j] = t
        end
        nxt = T - Int32(1)
        for k in Int32(1):n
            v = orders[k, j]
            isopen[v+1] != 0x00 && continue
            nr = Int32(0)
            for e in occptr[v+1]:occptr[v+2]-Int32(1)
                r = findroot!(rep, occ[e+1], j)
                seen = false
                for q in Int32(1):nr
                    roots[q, j] == r && (seen = true)
                end
                if !seen
                    # insertion by cur (ascending)
                    c = cur[r+1, j]; q = nr
                    while q >= Int32(1) && cur[roots[q, j]+1, j] > c
                        roots[q+1, j] = roots[q, j]; q -= Int32(1)
                    end
                    roots[q+1, j] = r; nr += Int32(1)
                end
            end
            nr < Int32(2) && continue
            r1 = roots[1, j]; node = cur[r1+1, j]
            for q in Int32(2):nr
                nxt += Int32(1)
                left[nxt+1, j] = node; right[nxt+1, j] = cur[roots[q, j]+1, j]; node = nxt
                rep[roots[q, j]+1, j] = r1
            end
            cur[r1+1, j] = node
        end
        # leftover classes
        nr = Int32(0)
        for t in Int32(0):T-Int32(1)
            if findroot!(rep, t, j) == t
                c = cur[t+1, j]; q = nr
                while q >= Int32(1) && cur[roots[q, j]+1, j] > c
                    roots[q+1, j] = roots[q, j]; q -= Int32(1)
                end
                roots[q+1, j] = t; nr += Int32(1)
            end
        end
        node = cur[roots[1, j]+1, j]
        for q in Int32(2):nr
            nxt += Int32(1)
            left[nxt+1, j] = node; right[nxt+1, j] = cur[roots[q, j]+1, j]; node = nxt
        end
        # parent, subtree sizes, preorder
        parent[N, j] = Int32(-1)
        for x in Int32(0):T-Int32(1)
            sz[x+1, j] = Int32(1)
        end
        for p in T:N-Int32(1)
            l = left[p+1, j]; r = right[p+1, j]
            parent[l+1, j] = p; parent[r+1, j] = p
            sz[p+1, j] = Int32(1) + sz[l+1, j] + sz[r+1, j]
        end
        pre[N, j] = Int32(0)
        for p in N-Int32(1):-Int32(1):T
            l = left[p+1, j]; r = right[p+1, j]; pp = pre[p+1, j]
            pre[l+1, j] = pp + Int32(1); pre[r+1, j] = pp + Int32(1) + sz[l+1, j]
        end
    end
    return
end

function build!(sc::Scratch, g::GNet, J::Int = sc.J)
    @cuda threads = 64 blocks = cld(J, 64) build_kernel!(sc.left, sc.right, sc.parent, sc.sz, sc.pre, sc.rep, sc.cur, sc.roots, sc.orders, g.occptr, g.occ, g.isopen, g.n, g.T, Int32(J))
end

# ----------------------------------------------------------------------------- cost
@inline isanc(pre, sz, a::Int32, b::Int32, j) = @inbounds (pre[a+1, j] <= pre[b+1, j]) & (pre[b+1, j] < pre[a+1, j] + sz[a+1, j])

@inline function warpmax(x::Float32)
    for o in (16, 8, 4, 2, 1)
        x = max(x, shfl_xor_sync(0xffffffff, x, o))
    end
    x
end
@inline function warpsum(x::Float32)
    for o in (16, 8, 4, 2, 1)
        x += shfl_xor_sync(0xffffffff, x, o)
    end
    x
end

function cost_kernel!(acc, leg, log2C, Wout, maxop, parent, sz, pre, sbuf, occptr, occ, w, isopen, n::Int32, T::Int32, J::Int32, mo::Int32)
    tid = Int32(threadIdx().x) - Int32(1)
    lane = tid & Int32(31)
    j = Int32((blockIdx().x - 1) * (blockDim().x ÷ 32)) + (tid >> Int32(5)) + Int32(1)
    j > J && return
    N = Int32(2) * T - Int32(1); root = N - Int32(1)
    @inbounds begin
        x = lane
        while x < N
            acc[x+1, j] = 0.0f0; leg[x+1, j] = 0.0f0
            x += Int32(32)
        end
        sync_warp()
        i = lane
        while i < n
            e0 = occptr[i+1]; k = occptr[i+2] - e0
            op = isopen[i+1] != 0x00
            if k >= Int32(2) || op
                wi = w[i+1]
                # leaves sorted by preorder into this lane's buffer
                bo = lane * mo
                for q in Int32(0):k-Int32(1)
                    y = occ[e0+q+1]; py = pre[y+1, j]; r = q
                    while r >= Int32(1) && pre[sbuf[bo+r, j]+1, j] > py
                        sbuf[bo+r+1, j] = sbuf[bo+r, j]; r -= Int32(1)
                    end
                    sbuf[bo+r+1, j] = y
                end
                lastl = sbuf[bo+k, j]
                y = sbuf[bo+1, j]
                while true
                    top = op ? (y == root) : isanc(pre, sz, y, lastl, j)
                    y >= T && CUDA.@atomic acc[y+1, j] += wi
                    (!top || op) && CUDA.@atomic leg[y+1, j] += wi
                    top && break
                    y = parent[y+1, j]
                end
                for q in Int32(2):k
                    y = sbuf[bo+q, j]; prv = sbuf[bo+q-Int32(1), j]
                    while !isanc(pre, sz, y, prv, j)
                        y >= T && CUDA.@atomic acc[y+1, j] += wi
                        CUDA.@atomic leg[y+1, j] += wi
                        y = parent[y+1, j]
                    end
                end
            end
            i += Int32(32)
        end
        sync_warp()
        threadfence_block()
        m = -Inf32; wm = -Inf32
        x = lane
        while x < N
            x >= T && (m = max(m, acc[x+1, j]))
            wm = max(wm, leg[x+1, j])
            x += Int32(32)
        end
        m = warpmax(m); wm = warpmax(wm)
        s = 0.0f0
        x = T + lane
        while x < N
            s += exp2(acc[x+1, j] - m)
            x += Int32(32)
        end
        s = warpsum(s)
        if lane == Int32(0)
            log2C[j] = m + log2(s); Wout[j] = wm; maxop[j] = m
        end
    end
    return
end

function gcost!(sc::Scratch, g::GNet, J::Int = sc.J)
    wpb = 4
    @cuda threads = 32wpb blocks = cld(J, wpb) cost_kernel!(sc.acc, sc.leg, sc.log2C, sc.W, sc.maxop, sc.parent, sc.sz, sc.pre, sc.sbuf, g.occptr, g.occ, g.w, g.isopen, g.n, g.T, Int32(J), g.maxocc)
end

"""Score the orderings already in sc.orders[:, 1:J]."""
function score_orders!(sc::Scratch, g::GNet, J::Int = sc.J)
    build!(sc, g, J); gcost!(sc, g, J)
end

"""Host copy (1-based, Host.Tree layout) of job j's tree: (left, right) vectors over nodes 1..2T-1."""
function fetch_tree(sc::Scratch, g::GNet, j::Int)
    T = Int(g.T); N = 2T - 1
    l = Array(view(sc.left, :, j)) .+ 1; r = Array(view(sc.right, :, j)) .+ 1
    l[1:T] .= 0; r[1:T] .= 0
    l, r
end

end
