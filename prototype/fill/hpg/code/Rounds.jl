# Round-based (bulk-synchronous) greedy elimination orderings.
#
# Each round: score every live vertex, take the candidates whose score is at
# most min*(1+relax)+delta, pick among them a set I that is independent at
# distance 2 (closed neighbourhoods pairwise disjoint, i.e. pairwise distance
# >= 3), eliminate all of I at once and update the scores in bulk.
#
# Because the closed neighbourhoods of I are disjoint, every elimination
# touches the adjacency lists of its own clique only, and the deficiency
# updates of different eliminations commute; with integer (fixed-point)
# weights the round is exactly a sequential elimination of I in any order,
# and the result does not depend on the number of threads.
module Rounds

using CliqueTrees, Graphs, Random
using Base.Threads: @spawn, nthreads, Atomic, atomic_add!

export RoundsAlg, rounds

const WSCALE = 1024          # fixed-point scale of the vertex weights

@inline function atomadd!(x::Vector{Int}, i::Integer, d::Int)
    GC.@preserve x Core.Intrinsics.atomic_pointermodify(pointer(x, i), +, d, :monotonic)
    return nothing
end

@inline function atommin!(x::Vector{Int32}, i::Integer, d::Int32)
    GC.@preserve x Core.Intrinsics.atomic_pointermodify(pointer(x, i), min, d, :monotonic)
    return nothing
end

"""
    RoundsAlg(; score=:mf, relax=0.0, delta=0.0, select=:greedy, ties=:random,
              tail=0, seed=1, ntasks=0, lim=typemax(Int))

  - `score`: `:md` (weighted external degree, as MMD), `:mf` (weighted
    deficiency, as MF strategy 0), `:mf1` (deficiency / bag weight, as MF
    strategy 1)
  - `relax`, `delta`: candidates have score <= min*(1+relax) + delta
  - `select`: `:greedy` (scan candidates by score, claim closed
    neighbourhoods) or `:luby` (one pass: keep a candidate if it has the
    smallest rank in its distance-2 neighbourhood)
  - `ties`: `:random` (hash of vertex and seed) or `:index`
  - `tail`: finish with the serial algorithm once at most `tail` vertices are left,
    or once a round selects at most `tailsel` vertices with at most `tailmax` left
  - `compress`: merge indistinguishable vertices (initially and after each round)
  - `lim`: at most `lim` candidates per round
"""
Base.@kwdef struct RoundsAlg <: CliqueTrees.EliminationAlgorithm
    score::Symbol = :mf
    relax::Float64 = 0.0
    delta::Float64 = 0.0
    select::Symbol = :greedy
    ties::Symbol = :random
    tail::Int = 0
    tailsel::Int = 0
    tailmax::Int = 0
    compress::Bool = true
    tailalg::Any = nothing
    seed::UInt64 = 1
    ntasks::Int = 0
    lim::Int = typemax(Int)
end

serialalg(a::RoundsAlg) = a.tailalg !== nothing ? a.tailalg : a.score === :md ? MMD() : a.score === :mf ? MF() : MF(; strategy = 1)

# ---------------------------------------------------------------------------
# dynamic parallel for: f(task, i) for i in 1:n
function pfor(f::F, n0::Integer, ntasks::Integer, grain::Integer = 1) where {F}
    n = Int(n0)
    nt = min(Int(ntasks), cld(n, Int(grain)))
    if nt <= 1
        for i in 1:n
            f(1, i)
        end
        return nothing
    end
    ctr = Atomic{Int}(1)
    @sync for t in 1:nt
        @spawn begin
            while true
                i0 = atomic_add!(ctr, grain)
                i0 > n && break
                for i in i0:min(i0 + grain - 1, n)
                    f(t, i)
                end
            end
        end
    end
    return nothing
end

mutable struct Work
    mk::Vector{Int32}       # K membership stamp
    mk2::Vector{Int32}      # adjacency stamp
    s1::Int32
    s2::Int32
    sw::Vector{Int}         # weight of the K-neighbours of K[i]
    pa::Vector{Int32}       # missing pairs (K indices)
    pb::Vector{Int32}
    hk::Vector{Tuple{UInt64, Int, Int32}}
    mrg::Vector{Tuple{Int32, Int32}}
end

Work(n) = Work(zeros(Int32, n), zeros(Int32, n), 0, 0, Int[], Int32[], Int32[], Tuple{UInt64, Int, Int32}[], Tuple{Int32, Int32}[])

@inline function stamp1!(wk::Work)
    if wk.s1 == typemax(Int32)
        fill!(wk.mk, 0); wk.s1 = 0
    end
    return wk.s1 += Int32(1)
end

@inline function stamp2!(wk::Work)
    if wk.s2 == typemax(Int32)
        fill!(wk.mk2, 0); wk.s2 = 0
    end
    return wk.s2 += Int32(1)
end

struct State
    n::Int
    w::Vector{Int}
    adj::Vector{Vector{Int32}}
    wdeg::Vector{Int}
    def::Vector{Int}
    absorbed::Vector{Bool}
    snext::Vector{Int32}    # member chains of the supervariables
    slast::Vector{Int32}
end

function State(weights::AbstractVector, graph::AbstractGraph, ntasks::Int = 1)
    n = Int(nv(graph))
    w = [max(1, round(Int, WSCALE * weights[v])) for v in 1:n]
    adj = Vector{Vector{Int32}}(undef, n)
    wdeg = zeros(Int, n)
    pfor(n, ntasks, 4096) do t, v
        a = Int32[u for u in neighbors(graph, v) if u != v]
        sort!(a); unique!(a)
        adj[v] = a
        s = 0
        for u in a
            s += w[u]
        end
        wdeg[v] = s
    end
    return State(n, w, adj, wdeg, zeros(Int, n), falses(n) |> Vector{Bool}, zeros(Int32, n), collect(Int32, 1:n))
end

# initial weighted deficiency: ((sum w)^2 - sum w^2)/2 - (weight of edges inside N(v))
function initdef!(st::State, works::Vector{Work}, ntasks::Int)
    adj = st.adj; w = st.w; def = st.def
    pfor(st.n, ntasks, 64) do t, v
        wk = works[t]
        s = stamp1!(wk)
        mk = wk.mk
        av = adj[v]
        for a in av
            @inbounds mk[a] = s
        end
        W = 0; W2 = 0; tri = 0
        for a in av
            @inbounds wa = w[a]
            W += wa; W2 += wa * wa
            for b in adj[a]
                @inbounds if mk[b] == s
                    tri += wa * w[b]
                end
            end
        end
        @inbounds def[v] = ((W * W - W2) >> 1) - (tri >> 1)
    end
    return nothing
end

# from-scratch deficiency (for checking)
function deficiency(st::State, v::Integer)
    av = filter(u -> !st.absorbed[u], st.adj[v]); set = Set(av); w = st.w
    W = 0; W2 = 0; tri = 0
    for a in av
        W += w[a]; W2 += w[a]^2
        for b in st.adj[a]
            b in set && (tri += w[a] * w[b])
        end
    end
    return ((W * W - W2) >> 1) - (tri >> 1)
end

# eliminate v; the caller guarantees that the closed neighbourhoods of the
# vertices eliminated concurrently are pairwise disjoint
function eliminate!(st::State, wk::Work, v::Int32, needdef::Bool, K::Vector{Int32})
    adj = st.adj; w = st.w; def = st.def; wdeg = st.wdeg; absorbed = st.absorbed
    mk = wk.mk; mk2 = wk.mk2
    empty!(K)
    @inbounds for x in adj[v]
        absorbed[x] || push!(K, x)
    end
    d = length(K)
    @inbounds wv = w[v]
    s1 = stamp1!(wk)
    WK = 0
    @inbounds for i in 1:d
        x = K[i]; mk[x] = s1; WK += w[x]
    end
    length(wk.sw) < d && resize!(wk.sw, d)
    sw = wk.sw; pa = wk.pa; pb = wk.pb
    empty!(pa); empty!(pb)
    # K-neighbour weights and missing pairs inside K
    @inbounds for i in 1:d
        a = K[i]
        s2 = stamp2!(wk)
        swa = 0
        for b in adj[a]
            if mk[b] == s1
                mk2[b] = s2
                swa += w[b]
            end
        end
        sw[i] = swa
        for j in (i + 1):d
            if mk2[K[j]] != s2
                push!(pa, Int32(i)); push!(pb, Int32(j))
            end
        end
    end
    np = length(pa)
    if needdef
        # Wing-Huang: each new edge {a,b} lowers the deficiency of every
        # common neighbour c by w_a w_b, and raises those of a and b by the
        # weight of the new missing pairs, computed below from the common
        # neighbours outside K
        p = 1
        @inbounds while p <= np
            i = pa[p]; a = K[i]; wa = w[a]
            s2 = stamp2!(wk)
            for c in adj[a]
                mk2[c] = s2
            end
            while p <= np && pa[p] == i
                b = K[pb[p]]; wb = w[b]
                wab = wa * wb; wout = 0
                for c in adj[b]
                    if mk2[c] == s2 && c != v && !absorbed[c]
                        atomadd!(def, c, -wab)
                        if mk[c] != s1
                            wout += w[c]
                        end
                    end
                end
                if wout != 0
                    atomadd!(def, a, -wb * wout)
                    atomadd!(def, b, -wa * wout)
                end
                p += 1
            end
        end
        # removal of v and the new neighbours: with O = N(x) \ K \ v and
        # Nn = K \ N[x], def += -w_v W(O) + W(Nn) W(O) (the common-neighbour
        # part of the new pairs was subtracted above)
        @inbounds for i in 1:d
            x = K[i]; swx = sw[i]
            Wn = WK - w[x] - swx
            Wo = wdeg[x] - wv - swx
            atomadd!(def, x, Wo * (Wn - wv))
        end
    end
    @inbounds for i in 1:d
        x = K[i]
        wdeg[x] += WK - w[x] - sw[i] - wv
        ax = adj[x]
        j = 0
        for k in eachindex(ax)
            y = ax[k]
            if y != v && !absorbed[y]
                ax[j += 1] = y
            end
        end
        resize!(ax, j)
    end
    @inbounds for p in 1:np
        a = K[pa[p]]; b = K[pb[p]]
        push!(adj[a], b); push!(adj[b], a)
    end
    adj[v] = Int32[]
    return nothing
end

@inline function mix(v::Integer)
    x = UInt64(v) * 0x9e3779b97f4a7c15
    x ⊻= x >> 32
    return x * 0xd6e8feb86659fd93
end

# y joins the supervariable of x (x and y indistinguishable, both live)
@inline function absorb!(st::State, x::Int32, y::Int32)
    @inbounds begin
        st.w[x] += st.w[y]
        st.wdeg[x] -= st.w[y]
        st.absorbed[y] = true
        st.snext[st.slast[x]] = y
        st.slast[x] = st.slast[y]
    end
    return nothing
end

# initial compression: merge vertices with equal closed neighbourhoods, then
# rebuild the adjacency lists without the absorbed vertices
function compress!(st::State, works::Vector{Work}, ntasks::Int = 1)
    n = st.n; adj = st.adj; wk = works[1]
    hs = Vector{Tuple{UInt64, Int, Int32}}(undef, n)
    pfor(n, ntasks, 4096) do t, v
        h = mix(v)
        for u in adj[v]
            h += mix(u)
        end
        hs[v] = (h, length(adj[v]), Int32(v))
    end
    sort!(hs)
    nm = 0
    i = 1
    while i <= n
        j = i
        while j < n && hs[j + 1][1] == hs[i][1] && hs[j + 1][2] == hs[i][2]
            j += 1
        end
        for a in i:j
            x = hs[a][3]
            st.absorbed[x] && continue
            s = stamp2!(wk)
            wk.mk2[x] = s
            for u in adj[x]
                wk.mk2[u] = s
            end
            for b in (a + 1):j
                y = hs[b][3]
                st.absorbed[y] && continue
                ok = wk.mk2[y] == s
                if ok
                    for u in adj[y]
                        if wk.mk2[u] != s
                            ok = false; break
                        end
                    end
                end
                if ok
                    absorb!(st, x, y); nm += 1
                end
            end
        end
        i = j + 1
    end
    if nm > 0
        pfor(n, ntasks, 4096) do t, v
            st.absorbed[v] || filter!(u -> !st.absorbed[u], adj[v])
        end
        for v in 1:n
            st.absorbed[v] && (adj[v] = Int32[])
        end
    end
    return nm
end

# after a round: indistinguishable vertices inside one clique K have equal
# neighbourhoods outside K; find them (read-only pass)
function detect!(st::State, wk::Work, K::Vector{Int32})
    adj = st.adj; absorbed = st.absorbed
    empty!(wk.mrg)
    d = length(K)
    d < 2 && return nothing
    s1 = stamp1!(wk)
    @inbounds for x in K
        wk.mk[x] = s1
    end
    hk = wk.hk; empty!(hk)
    @inbounds for x in K
        h = zero(UInt64); c = 0
        for o in adj[x]
            if wk.mk[o] != s1 && !absorbed[o]
                h += mix(o); c += 1
            end
        end
        push!(hk, (h, c, x))
    end
    sort!(hk)
    i = 1
    @inbounds while i <= d
        j = i
        while j < d && hk[j + 1][1] == hk[i][1] && hk[j + 1][2] == hk[i][2]
            j += 1
        end
        if j > i
            for a in i:j
                x = hk[a][3]
                x == 0 && continue
                s = stamp2!(wk)
                for o in adj[x]
                    wk.mk2[o] = s
                end
                for b in (a + 1):j
                    y = hk[b][3]
                    y == 0 && continue
                    ok = true
                    for o in adj[y]
                        if wk.mk[o] != s1 && !absorbed[o] && wk.mk2[o] != s
                            ok = false; break
                        end
                    end
                    if ok
                        push!(wk.mrg, (x, y))
                        hk[b] = (hk[b][1], hk[b][2], Int32(0))
                    end
                end
            end
        end
        i = j + 1
    end
    return nothing
end

@inline function tiehash(v::Integer, seed::UInt64)
    x = (UInt64(v) + seed * 0x9e3779b97f4a7c15) * 0xbf58476d1ce4e5b9
    x ⊻= x >> 31
    x *= 0x94d049bb133111eb
    return x ⊻ (x >> 29)
end

@inline function key(st::State, sc::Symbol, v)
    @inbounds if sc === :md
        return Float64(st.wdeg[v])
    elseif sc === :mf
        return Float64(st.def[v])
    else
        return Float64(st.def[v]) / Float64(st.w[v] + st.wdeg[v])
    end
end

scoreunit(sc::Symbol) = sc === :md ? Float64(WSCALE) : sc === :mf ? Float64(WSCALE)^2 : Float64(WSCALE)

@inline function hash32(v::Int32, seed::UInt32)
    x = reinterpret(UInt32, v) * 0x9e3779b1 + seed * 0x85ebca6b
    x ⊻= x >> 16; x *= 0x7feb352d; x ⊻= x >> 15; x *= 0x846ca68b; x ⊻= x >> 16
    return x
end

# the priority of the GPU code: Float32 score bits, then a hash
@inline function fkey(st::State, sc::Symbol, v)
    @inbounds if sc === :md
        return Float32(st.wdeg[v])
    elseif sc === :mf
        return Float32(st.def[v])
    else
        return Float32(st.def[v]) / Float32(st.w[v] + st.wdeg[v])
    end
end

@inline packpri(k::Float32, v::Int32, seed::UInt32) = (UInt64(reinterpret(UInt32, k)) << 32) | UInt64(hash32(v, seed))

@inline function atommin!(x::Vector{UInt64}, i::Integer, d::UInt64)
    GC.@preserve x Core.Intrinsics.atomic_pointermodify(pointer(x, i), min, d, :monotonic)
    return nothing
end

# parallel stable filter of src into dst
function pfilter!(dst::Vector{T}, src::Vector{T}, pred::F, ntasks::Int) where {T, F}
    n = length(src)
    nt = min(ntasks, max(1, n ÷ 8192))
    if nt <= 1
        resize!(dst, n); j = 0
        @inbounds for x in src
            pred(x) && (dst[j += 1] = x)
        end
        return resize!(dst, j)
    end
    cnt = zeros(Int, nt + 1)
    ch = cld(n, nt)
    @sync for t in 1:nt
        @spawn begin
            local c = 0
            @inbounds for i in ((t - 1) * ch + 1):min(t * ch, n)
                pred(src[i]) && (c += 1)
            end
            cnt[t + 1] = c
        end
    end
    cumsum!(cnt, cnt)
    resize!(dst, cnt[end])
    @sync for t in 1:nt
        @spawn begin
            local j = cnt[t]
            @inbounds for i in ((t - 1) * ch + 1):min(t * ch, n)
                x = src[i]
                pred(x) && (dst[j += 1] = x)
            end
        end
    end
    return dst
end

"""
    rounds(weights, graph, alg::RoundsAlg; stats=nothing, check=false, timing=nothing)

Return `(order, index)`. If `stats` is a vector, push per-round
`(nalive, ncand, nsel)` tuples to it; if `timing` is a Dict, fill in the
time of each phase.
"""
function rounds(weights::AbstractVector, graph::AbstractGraph, alg::RoundsAlg; stats = nothing, check::Bool = false, timing = nothing)
    T0 = time()
    n = Int(nv(graph))
    ntasks = alg.ntasks > 0 ? alg.ntasks : nthreads()
    sc = alg.score
    needdef = sc !== :md
    st = State(weights, graph, ntasks)
    works = [Work(n) for _ in 1:ntasks]
    alg.compress && compress!(st, works, ntasks)
    needdef && initdef!(st, works, ntasks)
    T1 = time()

    order = Vector{Int}(undef, n); nord = 0
    alist = pfilter!(Int32[], collect(Int32, 1:n), v -> !st.absorbed[v], ntasks)
    alist2 = Int32[]
    keys = Vector{Float32}(undef, n)
    claim = zeros(Int32, n)
    best = fill(typemax(UInt64), n)
    unit = scoreunit(sc)
    seed = UInt32(alg.seed % UInt32)
    round = Int32(0)
    sel = Int32[]
    cand = Int32[]
    kbuf = Vector{Int32}[]
    mrgs = Vector{Tuple{Int32, Int32}}[]
    tm = zeros(5)
    small = 0

    function emit!(v)
        while v != 0
            order[nord += 1] = v
            v = st.snext[v]
        end
    end

    while !isempty(alist)
        nalive = length(alist)
        nalive <= alg.tail && break
        round += Int32(1)
        ta = time()
        # scores and candidates
        let alist = alist
            pfor(nalive, ntasks, 8192) do t, i
                local v
                @inbounds v = alist[i]
                @inbounds keys[v] = fkey(st, sc, v)
            end
        end
        kmin = Inf32
        @inbounds for v in alist
            kmin = min(kmin, keys[v])
        end
        thr = isinf(alg.relax) ? Inf32 : max(kmin, Float32(kmin * (1 + alg.relax) + alg.delta * unit))
        pfilter!(cand, alist, v -> @inbounds(keys[v] <= thr), ntasks)
        tb = time(); tm[1] += tb - ta
        empty!(sel)
        if alg.select === :greedy
            pri = [packpri(keys[v], v, seed) for v in cand]
            perm = sortperm(pri)
            nscan = min(length(perm), alg.lim)
            @inbounds for q in 1:nscan
                v = cand[perm[q]]
                ok = claim[v] != round
                if ok
                    for x in st.adj[v]
                        if claim[x] == round
                            ok = false; break
                        end
                    end
                end
                if ok
                    push!(sel, v)
                    claim[v] = round
                    for x in st.adj[v]
                        claim[x] = round
                    end
                end
            end
        else # :luby, one pass (as on the GPU)
            nc = length(cand)
            let cand = cand
                pfor(nc, ntasks, 256) do t, r
                    local v, p
                    @inbounds v = cand[r]
                    p = packpri(keys[v], v, seed)
                    atommin!(best, v, p)
                    for x in st.adj[v]
                        atommin!(best, x, p)
                    end
                end
                pfilter!(sel, cand, v -> begin
                    local p, ok
                    p = packpri(keys[v], v, seed)
                    ok = best[v] == p
                    if ok
                        for x in st.adj[v]
                            if best[x] != p
                                ok = false; break
                            end
                        end
                    end
                    ok
                end, ntasks)
                pfor(nc, ntasks, 256) do t, r
                    local v
                    @inbounds v = cand[r]
                    best[v] = typemax(UInt64)
                    for x in st.adj[v]
                        best[x] = typemax(UInt64)
                    end
                end
            end
        end
        @assert !isempty(sel)
        tc = time(); tm[2] += tc - tb
        # dense end: few and large eliminations per round, for two rounds
        ktot = 0
        for v in sel
            ktot += length(st.adj[v])
        end
        small = (length(sel) <= alg.tailsel && ktot >= 32 * length(sel)) ? small + 1 : 0
        if small >= 2 && nalive <= alg.tailmax
            break
        end
        stats === nothing || push!(stats, (nalive, length(cand), length(sel)))
        # eliminate
        ns = length(sel)
        while length(kbuf) < ns
            push!(kbuf, Int32[]); push!(mrgs, Tuple{Int32, Int32}[])
        end
        let sel = sel
            pfor(ns, ntasks, 1) do t, i
                @inbounds eliminate!(st, works[t], sel[i], needdef, kbuf[i])
            end
            td = time(); tm[3] += td - tc
            if alg.compress
                pfor(ns, ntasks, 1) do t, i
                    @inbounds begin
                        detect!(st, works[t], kbuf[i])
                        local mi = mrgs[i]; empty!(mi); append!(mi, works[t].mrg)
                    end
                end
                pfor(ns, ntasks, 1) do t, i
                    @inbounds for (x, y) in mrgs[i]
                        absorb!(st, x, y)
                    end
                end
            end
            tc = td
        end
        te = time(); tm[4] += te - tc
        @inbounds for v in sel
            emit!(v)
            st.absorbed[v] = true   # dead
        end
        # compact the live list
        pfilter!(alist2, alist, v -> !st.absorbed[v], ntasks)
        alist, alist2 = alist2, alist
        tm[5] += time() - te
        if check && needdef
            for v in alist
                @assert st.def[v] == deficiency(st, v) "deficiency mismatch at $v (round $round)"
            end
        end
    end
    T2 = time()

    nal = length(alist)
    if !isempty(alist)
        # serial tail on the remaining elimination graph
        m = length(alist)
        loc = zeros(Int, n)
        for (i, v) in enumerate(alist)
            loc[v] = i
        end
        sub = SimpleGraph(m)
        for (i, v) in enumerate(alist), u in st.adj[v]
            j = loc[u]
            j > i && add_edge!(sub, i, j)
        end
        wsub = [st.w[v] / WSCALE for v in alist]
        so, _ = permutation(wsub, sub, serialalg(alg))
        for i in so
            emit!(alist[i])
        end
    end
    @assert nord == n
    if timing !== nothing
        timing[:pre] = T1 - T0; timing[:rounds] = round; timing[:loop] = T2 - T1; timing[:tail] = nal
        timing[:post] = time() - T2
        timing[:score] = tm[1]; timing[:sel] = tm[2]; timing[:elim] = tm[3]; timing[:detect] = tm[4]; timing[:compact] = tm[5]
    end
    return order, invperm(order)
end

function CliqueTrees.permutation(weights::AbstractVector, graph::AbstractGraph{V}, alg::RoundsAlg; verbose::Bool = false) where {V}
    order, index = rounds(weights, graph, alg)
    return convert(Vector{V}, order), convert(Vector{V}, index)
end

end # module
