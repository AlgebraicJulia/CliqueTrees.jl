# Rounds on CPU threads with flat arrays: the data layout of the GPU code
# (one adjacency pool with a bump pointer, per-vertex start and length,
# state bytes) and the per-task scratch of Rounds.jl, so that a round
# allocates nothing and the garbage collector stays out of the threads' way.
module FRounds

using CliqueTrees, Graphs
using Base.Threads: @spawn, nthreads, Atomic, atomic_add!
using ..Rounds: RoundsAlg, WSCALE, pfor, pfilter!, packpri, atomadd!, atommin!, mix, serialalg, scoreunit

export frounds

const ALIVE = 0x00
const ABSORBED = 0x01
const DEAD = 0x02

mutable struct FWork
    mk::Vector{Int32}
    mk2::Vector{Int32}
    s1::Int32
    s2::Int32
    K::Vector{Int32}
    sw::Vector{Int}
    pa::Vector{Int32}
    pb::Vector{Int32}
    hk::Vector{Tuple{UInt64, Int32, Int32}}
end

FWork(n) = FWork(zeros(Int32, n), zeros(Int32, n), 0, 0, Int32[], Int[], Int32[], Int32[], Tuple{UInt64, Int32, Int32}[])

@inline function stamp1!(wk::FWork)
    if wk.s1 == typemax(Int32)
        fill!(wk.mk, 0); wk.s1 = 0
    end
    return wk.s1 += Int32(1)
end

@inline function stamp2!(wk::FWork)
    if wk.s2 == typemax(Int32)
        fill!(wk.mk2, 0); wk.s2 = 0
    end
    return wk.s2 += Int32(1)
end

mutable struct FState
    n::Int
    pool::Vector{Int32}
    ptop::Atomic{Int}
    astart::Vector{Int}
    alen::Vector{Int32}
    stt::Vector{UInt8}
    w::Vector{Int}
    wdeg::Vector{Int}
    def::Vector{Int}
    parent::Vector{Int32}
end

@inline function bsearch(pool::Vector{Int32}, s::Int, l::Integer, c::Int32)
    lo = s; hi = s + l - 1
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        x = pool[mid]
        x == c && return true
        if x < c
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return false
end

# parallel sum of f(i) for i in 1:n
function psum(f::F, n::Int, ntasks::Int) where {F}
    nt = min(ntasks, max(1, n ÷ 4096))
    nt <= 1 && return sum((f(i) for i in 1:n); init = 0)
    ch = cld(n, nt)
    parts = zeros(Int, nt)
    @sync for t in 1:nt
        @spawn begin
            local acc = 0
            for i in ((t - 1) * ch + 1):min(t * ch, n)
                acc += f(i)
            end
            parts[t] = acc
        end
    end
    return sum(parts)
end

function pminimum(keys::Vector{Float32}, alist::Vector{Int32}, ntasks::Int)
    n = length(alist)
    nt = min(ntasks, max(1, n ÷ 8192))
    ch = cld(n, nt)
    parts = fill(Inf32, nt)
    @sync for t in 1:nt
        @spawn begin
            local m = Inf32
            @inbounds for i in ((t - 1) * ch + 1):min(t * ch, n)
                m = min(m, keys[alist[i]])
            end
            parts[t] = m
        end
    end
    return minimum(parts)
end

# ------------------------------------------------------------ setup

function fstate(weights::AbstractVector, graph::AbstractGraph, ntasks::Int, poolfactor::Float64)
    n = Int(nv(graph))
    cnt = zeros(Int, n)
    pfor(n, ntasks, 4096) do t, v
        local c = 0
        for u in neighbors(graph, v)
            u != v && (c += 1)
        end
        cnt[v] = c
    end
    astart = Vector{Int}(undef, n)
    tot = 0
    for v in 1:n
        astart[v] = tot + 1; tot += cnt[v]
    end
    pool = Vector{Int32}(undef, max(1024, round(Int, poolfactor * tot)))
    pfor(n, ntasks, 4096) do t, v
        local k = astart[v]
        for u in neighbors(graph, v)
            u != v && (pool[k] = u; k += 1)
        end
        seg = view(pool, astart[v]:(astart[v] + cnt[v] - 1))
        issorted(seg) || sort!(seg)
    end
    w = Vector{Int}(undef, n)
    pfor(n, ntasks, 4096) do t, v
        w[v] = max(1, round(Int, WSCALE * weights[v]))
    end
    return FState(n, pool, Atomic{Int}(tot), astart, Int32.(cnt), fill(ALIVE, n), w, zeros(Int, n), zeros(Int, n), zeros(Int32, n))
end

# merge vertices with equal closed neighbourhoods into the first of their hash run
function fcompress!(st::FState, ntasks::Int)
    n = st.n; pool = st.pool; astart = st.astart; alen = st.alen
    key = Vector{UInt64}(undef, n)
    pfor(n, ntasks, 4096) do t, v
        local h = mix(v)
        for k in astart[v]:(astart[v] + alen[v] - 1)
            h += mix(pool[k])
        end
        key[v] = (h & 0xffffffff00000000) | UInt64(v)
    end
    sort!(key)
    head = Vector{Int32}(undef, n)
    hv = Int32(0); prev = typemax(UInt64)
    for i in 1:n
        h = key[i] & 0xffffffff00000000
        h != prev && (hv = Int32(key[i] & 0xffffffff); prev = h)
        head[i] = hv
    end
    par = zeros(Int32, n)
    pfor(n, ntasks, 4096) do t, i
        local x = Int32(key[i] & 0xffffffff); local y = head[i]
        if x != y && alen[x] == alen[y]
            local ok = bsearch(pool, astart[x], alen[x], y)
            local k = astart[x]
            while ok && k < astart[x] + alen[x]
                local u = pool[k]
                ok = u == y || bsearch(pool, astart[y], alen[y], u)
                k += 1
            end
            ok && (par[x] = y)
        end
    end
    for x in 1:n
        r = par[x]
        if r != 0
            st.stt[x] = ABSORBED; st.parent[x] = r; st.w[r] += st.w[x]
        end
    end
    return nothing
end

function finit!(st::FState, works::Vector{FWork}, alist::Vector{Int32}, needdef::Bool, ntasks::Int)
    pool = st.pool; astart = st.astart; alen = st.alen; stt = st.stt; w = st.w
    pfor(st.n, ntasks, 4096) do t, v
        local acc = 0
        for k in astart[v]:(astart[v] + alen[v] - 1)
            local u = pool[k]
            stt[u] == ALIVE && (acc += w[u])
        end
        st.wdeg[v] = acc
    end
    needdef || return nothing
    pfor(length(alist), ntasks, 64) do t, i
        local v = alist[i]; local wk = works[t]
        local s = stamp1!(wk); local mk = wk.mk
        for k in astart[v]:(astart[v] + alen[v] - 1)
            @inbounds mk[pool[k]] = s
        end
        local W = 0; local W2 = 0; local tri = 0
        for k in astart[v]:(astart[v] + alen[v] - 1)
            @inbounds a = pool[k]
            @inbounds stt[a] == ALIVE || continue
            @inbounds wa = w[a]
            W += wa; W2 += wa * wa
            for q in astart[a]:(astart[a] + alen[a] - 1)
                @inbounds b = pool[q]
                @inbounds if mk[b] == s && stt[b] == ALIVE
                    tri += wa * w[b]
                end
            end
        end
        st.def[v] = ((W * W - W2) >> 1) - (tri >> 1)
    end
    return nothing
end

# ------------------------------------------------------------ one elimination

function felim!(st::FState, wk::FWork, v::Int32, needdef::Bool, kout::Vector{Int32}, ko::Int)
    pool = st.pool; astart = st.astart; alen = st.alen; stt = st.stt
    w = st.w; def = st.def; wdeg = st.wdeg
    mk = wk.mk; mk2 = wk.mk2
    K = wk.K; empty!(K)
    @inbounds for k in astart[v]:(astart[v] + alen[v] - 1)
        x = pool[k]
        stt[x] == ALIVE && push!(K, x)
    end
    d = length(K)
    @inbounds wv = w[v]
    s1 = stamp1!(wk)
    WK = 0
    @inbounds for i in 1:d
        x = K[i]; mk[x] = s1; WK += w[x]; kout[ko + i] = x
    end
    length(wk.sw) < d && resize!(wk.sw, d)
    sw = wk.sw; pa = wk.pa; pb = wk.pb
    empty!(pa); empty!(pb)
    @inbounds for i in 1:d
        a = K[i]
        s2 = stamp2!(wk)
        swa = 0
        for k in astart[a]:(astart[a] + alen[a] - 1)
            b = pool[k]
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
        p = 1
        @inbounds while p <= np
            i = pa[p]; a = K[i]; wa = w[a]
            s2 = stamp2!(wk)
            for k in astart[a]:(astart[a] + alen[a] - 1)
                mk2[pool[k]] = s2
            end
            while p <= np && pa[p] == i
                b = K[pb[p]]; wb = w[b]
                wab = wa * wb; wout = 0
                for k in astart[b]:(astart[b] + alen[b] - 1)
                    c = pool[k]
                    if mk2[c] == s2 && c != v && stt[c] == ALIVE
                        atomadd!(def, c, -wab)
                        mk[c] != s1 && (wout += w[c])
                    end
                end
                if wout != 0
                    atomadd!(def, a, -wb * wout)
                    atomadd!(def, b, -wa * wout)
                end
                p += 1
            end
        end
        @inbounds for i in 1:d
            x = K[i]; swx = sw[i]
            Wn = WK - w[x] - swx
            Wo = wdeg[x] - wv - swx
            atomadd!(def, x, Wo * (Wn - wv))
        end
    end
    # new lists: (N(x) \ K \ v, live) then K \ x, at fresh pool space
    @inbounds for i in 1:d
        x = K[i]
        wdeg[x] += WK - w[x] - sw[i] - wv
        sx = astart[x]; lx = alen[x]
        c = 0
        for k in sx:(sx + lx - 1)
            e = pool[k]
            (e != v && stt[e] == ALIVE && mk[e] != s1) && (c += 1)
        end
        newlen = c + d - 1
        pos = atomic_add!(st.ptop, newlen) + 1
        o = pos
        for k in sx:(sx + lx - 1)
            e = pool[k]
            if e != v && stt[e] == ALIVE && mk[e] != s1
                pool[o] = e; o += 1
            end
        end
        for j in 1:d
            j == i && continue
            pool[o] = K[j]; o += 1
        end
        astart[x] = pos; alen[x] = Int32(newlen)
    end
    return d
end

# indistinguishable vertices of one clique: equal neighbourhoods outside it
function fdetect!(st::FState, wk::FWork, kout::Vector{Int32}, ko::Int, d::Int, par::Vector{Int32})
    pool = st.pool; astart = st.astart; alen = st.alen; stt = st.stt
    @inbounds for i in 1:d
        par[ko + i] = 0
    end
    d < 2 && return nothing
    s1 = stamp1!(wk)
    @inbounds for i in 1:d
        wk.mk[kout[ko + i]] = s1
    end
    hk = wk.hk; empty!(hk)
    @inbounds for i in 1:d
        x = kout[ko + i]
        h = zero(UInt64); c = Int32(0)
        for k in astart[x]:(astart[x] + alen[x] - 1)
            o = pool[k]
            if wk.mk[o] != s1 && stt[o] == ALIVE
                h += mix(o); c += Int32(1)
            end
        end
        push!(hk, (h, c, Int32(i)))
    end
    sort!(hk)
    a = 1
    @inbounds while a <= d
        b = a
        while b < d && hk[b + 1][1] == hk[a][1] && hk[b + 1][2] == hk[a][2]
            b += 1
        end
        if b > a
            x = kout[ko + hk[a][3]]
            s = stamp2!(wk)
            for k in astart[x]:(astart[x] + alen[x] - 1)
                wk.mk2[pool[k]] = s
            end
            for q in (a + 1):b
                y = kout[ko + hk[q][3]]
                ok = true
                for k in astart[y]:(astart[y] + alen[y] - 1)
                    o = pool[k]
                    if wk.mk[o] != s1 && stt[o] == ALIVE && wk.mk2[o] != s
                        ok = false; break
                    end
                end
                ok && (par[ko + hk[q][3]] = x)
            end
        end
        a = b + 1
    end
    return nothing
end

# ------------------------------------------------------------ driver

function fcheck(st::FState, alist)
    nb(v) = Set(u for u in st.pool[st.astart[v]:(st.astart[v] + st.alen[v] - 1)] if st.stt[u] == ALIVE)
    for v in alist
        N = nb(v); W = 0; W2 = 0; tri = 0
        for a in N
            W += st.w[a]; W2 += st.w[a]^2
            for b in nb(a)
                b in N && (tri += st.w[a] * st.w[b])
            end
        end
        @assert st.def[v] == ((W * W - W2) >> 1) - (tri >> 1) "deficiency mismatch at $v"
        @assert st.wdeg[v] == sum((st.w[a] for a in N); init = 0) "wdeg mismatch at $v"
    end
end

function frounds(weights::AbstractVector, graph::AbstractGraph, alg::RoundsAlg; stats = nothing, timing = nothing, poolfactor = 3.0, check::Bool = false)
    T0 = time()
    ntasks = alg.ntasks > 0 ? alg.ntasks : nthreads()
    sc = alg.score
    needdef = sc !== :md
    st = fstate(weights, graph, ntasks, poolfactor)
    n = st.n
    alg.compress && fcompress!(st, ntasks)
    works = [FWork(n) for _ in 1:ntasks]
    alist = pfilter!(Int32[], collect(Int32, 1:n), v -> st.stt[v] == ALIVE, ntasks)
    finit!(st, works, alist, needdef, ntasks)
    T1 = time()

    alist2 = Int32[]; cand = Int32[]; sel = Int32[]
    keys = Vector{Float32}(undef, n)
    best = fill(typemax(UInt64), n)
    koff = Int[]; dsel = Int[]
    kout = Int32[]; par = Int32[]
    order = Vector{Int32}(undef, n); nord = 0
    seed = UInt32(alg.seed % UInt32)
    unit = scoreunit(sc)
    nround = 0; small = 0; nrepack = 0
    tm = zeros(4)
    while !isempty(alist)
        nalive = length(alist)
        nalive <= alg.tail && break
        ta = time()
        let alist = alist
            pfor(nalive, ntasks, 8192) do t, i
                local v = alist[i]
                @inbounds keys[v] = sc === :md ? Float32(st.wdeg[v]) : sc === :mf ? Float32(st.def[v]) :
                    Float32(st.def[v]) / Float32(st.w[v] + st.wdeg[v])
            end
        end
        kmin = pminimum(keys, alist, ntasks)
        thr = isinf(alg.relax) ? Inf32 : max(kmin, Float32(kmin * (1 + alg.relax) + alg.delta * unit))
        pfilter!(cand, alist, v -> @inbounds(keys[v] <= thr), ntasks)
        nc = length(cand)
        let cand = cand
            pfor(nc, ntasks, 256) do t, r
                local v = cand[r]; local p = packpri(keys[v], v, seed)
                atommin!(best, v, p)
                for k in st.astart[v]:(st.astart[v] + st.alen[v] - 1)
                    atommin!(best, st.pool[k], p)
                end
            end
            pfilter!(sel, cand, v -> begin
                local p = packpri(keys[v], v, seed)
                local ok = best[v] == p
                if ok
                    for k in st.astart[v]:(st.astart[v] + st.alen[v] - 1)
                        if best[st.pool[k]] != p
                            ok = false; break
                        end
                    end
                end
                ok
            end, ntasks)
            pfor(nc, ntasks, 256) do t, r
                local v = cand[r]
                best[v] = typemax(UInt64)
                for k in st.astart[v]:(st.astart[v] + st.alen[v] - 1)
                    best[st.pool[k]] = typemax(UInt64)
                end
            end
        end
        ns = length(sel)
        @assert ns > 0
        # offsets of the cliques, the pool bound
        resize!(koff, ns + 1); koff[1] = 0
        for i in 1:ns
            koff[i + 1] = koff[i] + st.alen[sel[i]]
        end
        ktot = koff[ns + 1]
        small = (ns <= alg.tailsel && ktot >= 32 * ns) ? small + 1 : 0
        (small >= 2 && nalive <= alg.tailmax) && break
        nround += 1
        stats === nothing || push!(stats, (nalive, nc, ns))
        bound = let sel = sel
            psum(ns, ntasks) do i
                local v = sel[i]; local l = Int(st.alen[v]); local acc = 0
                for k in st.astart[v]:(st.astart[v] + l - 1)
                    acc += Int(st.alen[st.pool[k]]) + l
                end
                acc
            end
        end
        if st.ptop[] + bound > length(st.pool)
            used = psum(nalive, ntasks) do i
                Int(st.alen[alist[i]])
            end
            newpool = Vector{Int32}(undef, max(length(st.pool), 2 * (used + bound)))
            off = Vector{Int}(undef, nalive + 1); off[1] = 0
            for i in 1:nalive
                off[i + 1] = off[i] + st.alen[alist[i]]
            end
            let alist = alist
                pfor(nalive, ntasks, 1024) do t, i
                    local v = alist[i]
                    copyto!(newpool, off[i] + 1, st.pool, st.astart[v], st.alen[v])
                    st.astart[v] = off[i] + 1
                end
            end
            st.pool = newpool; st.ptop[] = used; nrepack += 1
        end
        length(kout) < ktot && resize!(kout, 2ktot)
        length(par) < ktot && resize!(par, 2ktot)
        resize!(dsel, ns)
        tb = time(); tm[1] += tb - ta
        let sel = sel
            pfor(ns, ntasks, 1) do t, i
                dsel[i] = felim!(st, works[t], sel[i], needdef, kout, koff[i])
            end
            tc = time(); tm[2] += tc - tb
            if alg.compress
                pfor(ns, ntasks, 1) do t, i
                    fdetect!(st, works[t], kout, koff[i], dsel[i], par)
                end
                pfor(ns, ntasks, 1) do t, i
                    for j in 1:dsel[i]
                        local r = par[koff[i] + j]
                        if r != 0
                            local x = kout[koff[i] + j]
                            st.stt[x] = ABSORBED; st.parent[x] = r
                            st.w[r] += st.w[x]; st.wdeg[r] -= st.w[x]
                        end
                    end
                end
            end
            tm[3] += time() - tc
        end
        td = time()
        for v in sel
            st.stt[v] = DEAD; order[nord += 1] = v
        end
        pfilter!(alist2, alist, v -> st.stt[v] == ALIVE, ntasks)
        alist, alist2 = alist2, alist
        tm[4] += time() - td
        check && needdef && fcheck(st, alist)
    end
    T2 = time()
    nal = length(alist)
    final = order[1:nord]
    if nal > 0
        loc = zeros(Int, n)
        for (i, v) in enumerate(alist)
            loc[v] = i
        end
        fadj = Vector{Vector{Int}}(undef, nal); ne2 = 0
        for (i, v) in enumerate(alist)
            lst = Int[]
            for k in st.astart[v]:(st.astart[v] + st.alen[v] - 1)
                j = loc[st.pool[k]]
                j > 0 && push!(lst, j)
            end
            sort!(lst); unique!(lst); fadj[i] = lst; ne2 += length(lst)
        end
        sub = SimpleGraph(ne2 ÷ 2, fadj)
        so, _ = permutation([st.w[v] / WSCALE for v in alist], sub, serialalg(alg))
        append!(final, alist[so])
    end
    # expand the supervariables
    head = zeros(Int32, n); nxt = zeros(Int32, n)
    for x in 1:n
        r = st.parent[x]
        if r != 0
            nxt[x] = head[r]; head[r] = x
        end
    end
    out = Vector{Int}(undef, n); no = 0
    stack = Int32[]
    for v in final
        push!(stack, v)
        while !isempty(stack)
            x = pop!(stack)
            out[no += 1] = x
            c = head[x]
            while c != 0
                push!(stack, c); c = nxt[c]
            end
        end
    end
    @assert no == n "emitted $no of $n"
    if timing !== nothing
        timing[:pre] = T1 - T0; timing[:loop] = T2 - T1; timing[:post] = time() - T2
        timing[:rounds] = nround; timing[:tail] = nal; timing[:repack] = nrepack
        timing[:sel] = tm[1]; timing[:elim] = tm[2]; timing[:detect] = tm[3]; timing[:compact] = tm[4]
    end
    return out, invperm(out)
end

end # module
