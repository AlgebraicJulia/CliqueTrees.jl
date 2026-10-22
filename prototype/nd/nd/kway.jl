# Top-k variant (ND_NOTES.md "Closing the AMD gap", item 3); included in module GND.
#
# The top d ND levels (2^d parts) come from ONE multilevel hierarchy instead of d recursive ones:
#   - the GPU coarsens the (twin-compressed) graph once, down to ~2^d * kwc0 vertices;
#   - on the coarsest graph the CPU computes a nested dissection of depth d (BFS-grown initial
#     separators, best of kwtrials, node FM), giving every vertex a tree code: heap numbering, root 1,
#     children 2x (A side) and 2x+1 (B side); a vertex with code x < 2^d lies in node x's separator, a
#     code >= 2^d is a leaf (one of the 2^d parts);
#   - uncoarsening on the CPU threads (the level graphs are downloaded while the CPU works): codes are
#     projected, then at every level the separators are refined top-down by depth: node x's region
#     (codes in x's subtree) is labelled A / B / S and refined by the same 1-sided node FM as the
#     hybrid (all nodes of a depth in parallel). A vertex that leaves S for side c is put in node c's
#     separator (always a valid nested dissection; the FM of the next depth moves it down);
#   - the codes are uploaded once; the ND driver turns them into A/B/S labels for its first d levels
#     (one kernel per level) instead of running the multilevel separator.

const OUT = 0x07                       # label of a vertex outside every region of the current depth
@inline cdepth(c::Integer) = 31 - leading_zeros(UInt32(c))

# labels of depth t from the codes; returns region offsets / vertex lists per node x = 2^t + i (i = 0 .. 2^t - 1)
function kw_depth!(lab::Vector{UInt8}, code::Vector{Int32}, t::Int)
    n = length(code); K = 1 << t
    cnt = zeros(Int, K + 1)
    @inbounds for v in 1:n
        c = code[v]; dc = cdepth(c)
        if dc < t
            lab[v] = OUT
            continue
        end
        anc = c >> (dc - t)
        lab[v] = c == anc ? SS : iseven(c >> (dc - t - 1)) ? SA : SB
        cnt[anc - K + 2] += 1
    end
    for i in 2:(K + 1)
        cnt[i] += cnt[i - 1]
    end
    regv = Vector{Int32}(undef, cnt[end]); pos = cnt[1:K] .+ 1
    @inbounds for v in 1:n
        c = code[v]; dc = cdepth(c)
        dc < t && continue
        i = (c >> (dc - t)) - K + 1
        regv[pos[i]] = v; pos[i] += 1
    end
    return cnt, regv
end

# after the FM of depth t: S -> x; a vertex now on side c (c = 2x or 2x+1) whose code is not in c's subtree
# (it left S, possibly after being pulled from the other side in the same pass) -> c's separator
function kw_update!(code::Vector{Int32}, lab::Vector{UInt8}, regv, a::Int, b::Int, x::Int32)
    t = cdepth(x)
    @inbounds for i in a:b
        v = regv[i]; l = lab[v]
        if l == SS
            code[v] = x
        else
            ch = l == SA ? 2x : 2x + Int32(1)
            c = code[v]; dc = cdepth(c)
            (dc > t && (c >> (dc - t - 1)) == ch) || (code[v] = ch)
        end
    end
    return
end

# one depth of refinement (all nodes of depth t in parallel); W: region weights recomputed here
function kw_refine_depth!(lab, code, ptr, adj, vw, t::Int, o, seed::UInt32)
    off, regv = kw_depth!(lab, code, t)
    K = 1 << t
    ws = hws(length(vw))
    nt = min(Threads.nthreads(), K)
    next = Threads.Atomic{Int}(1)
    mv = zeros(Int, nt)
    work = function (k)
        wk = ws.works[k]
        while true
            i = Threads.atomic_add!(next, 1)
            i > K && break
            a = off[i] + 1; b = off[i + 1]
            a > b && continue
            W = 0
            @inbounds for j in a:b
                W += vw[regv[j]]
            end
            ms = Int32(floor((1 + o.eps) / 2 * W))
            m, = cpu_fm1_part!(lab, ptr, adj, vw, view(regv, a:b), ms, o.cfmpass, o.cfmlimit, seed + UInt32(i), ws, wk)
            mv[k] += m
            kw_update!(code, lab, regv, a, b, Int32(K + i - 1))
        end
    end
    if nt <= 1
        work(1)
    else
        @sync for k in 1:nt
            Threads.@spawn work(k)
        end
    end
    return sum(mv)
end

# initial separator of every node of depth t on the coarsest graph: best of `trials` BFS-grown bisections + FM
function kw_init_depth!(lab, code, ptr, adj, vw, t::Int, o, seed::UInt32)
    off, regv = kw_depth!(lab, code, t)     # every region vertex has code x: label SS
    K = 1 << t
    ws = hws(length(vw))
    nt = min(Threads.nthreads(), K)
    next = Threads.Atomic{Int}(1)
    work = function (k)
        wk = ws.works[k]
        q = Int32[]; seen = Dict{Int32, Bool}()
        while true
            i = Threads.atomic_add!(next, 1)
            i > K && break
            a = off[i] + 1; b = off[i + 1]
            a > b && continue
            R = view(regv, a:b); nr = b - a + 1
            W = sum(Int(vw[v]) for v in R)
            ms = Int32(floor((1 + o.eps) / 2 * W))
            best = nothing; bestS = typemax(Int); bestD = typemax(Int)
            for trial in 1:o.kwtrials
                for v in R; lab[v] = SB; end
                # BFS from a pseudo-random region vertex until A reaches half the weight (restarts on disconnected regions)
                h = mix32(seed + UInt32(trial) * 0x9e3779b1 + UInt32(i) * 0x85ebca6b)
                wa = 0; start = Int(h % UInt32(nr)) + 1; scan = 0
                empty!(q)
                while wa * 2 < W
                    if isempty(q)
                        # next unvisited region vertex
                        s = 0
                        for j in 0:(nr - 1)
                            u = R[(start + j - 1) % nr + 1]
                            if lab[u] == SB
                                s = u; break
                            end
                        end
                        s == 0 && break
                        lab[s] = SA; wa += vw[s]; push!(q, s); scan = 1
                    end
                    scan > length(q) && (empty!(q); continue)
                    u = q[scan]; scan += 1
                    for e in ptr[u]:(ptr[u + 1] - 1)
                        x = adj[e]
                        if lab[x] == SB && wa * 2 < W
                            lab[x] = SA; wa += vw[x]; push!(q, x)
                        end
                    end
                end
                # separator: A vertices with a B neighbour
                for v in R
                    if lab[v] == SA
                        for e in ptr[v]:(ptr[v + 1] - 1)
                            if lab[adj[e]] == SB
                                lab[v] = SS; break
                            end
                        end
                    end
                end
                cpu_fm1_part!(lab, ptr, adj, vw, R, ms, o.cfmpass, o.cfmlimit, seed + UInt32(trial), ws, wk)
                wS = 0; wA = 0; wB = 0
                for v in R
                    l = lab[v]; l == SS ? (wS += vw[v]) : l == SA ? (wA += vw[v]) : (wB += vw[v])
                end
                D = abs(wA - wB)
                if wS < bestS || (wS == bestS && D < bestD)
                    bestS = wS; bestD = D; best = UInt8[lab[v] for v in R]
                end
            end
            x = Int32(K + i - 1)
            for (j, v) in enumerate(R)
                l = best[j]; lab[v] = l
                code[v] = l == SS ? x : l == SA ? 2x : 2x + Int32(1)
            end
        end
    end
    if nt <= 1
        work(1)
    else
        @sync for k in 1:nt
            Threads.@spawn work(k)
        end
    end
    return
end

"""
    kway_codes(G, o, st) -> CuVector{Int32}

Tree codes (see the file comment) of depth o.kway for every vertex of the device graph G (one part).
"""
function kway_codes(G, o, st)
    t0 = time()
    d = o.kway
    n = G.n
    poff = CuVector{Int32}(Int32[1, n + 1]); Wp = CuVector{Int32}(Int32[sum(Array(G.vw))])
    c0 = o.kwc0 << d
    oc = GNDOpts(o; C0 = c0)
    levels = coarsen_levels(G, poff, 1, Wp, oc, 0, st, typemax(Int), c0)
    CUDA.synchronize()
    t1 = time()
    L = length(levels)
    ta = time()
    arena_reserve!(ceil(Int, 2.5 * (3n + G.m + 2)))
    st[:kw_arena] = time() - ta
    ch = Channel{NTuple{5, Vector{Int32}}}(L)
    dlt = Threads.@spawn begin
        for j in L:-1:1
            put!(ch, dl_level(levels[j], j < L))
        end
    end
    seed = mix32(o.seed * 0x632be5ab + UInt32(n))
    code = Int32[]; lab = UInt8[]
    tinit = 0.0; tref = 0.0; tw = 0.0; mvs = 0
    for j in L:-1:1
        tw0 = time()
        ptrH, adjH, vwH, _, cmH = take!(ch)
        tw += time() - tw0
        nj = length(vwH)
        if j == L
            code = ones(Int32, nj); lab = Vector{UInt8}(undef, nj)
            ti = time()
            for t in 0:(d - 1)
                kw_init_depth!(lab, code, ptrH, adjH, vwH, t, o, seed + UInt32(t))
            end
            tinit = time() - ti
        else
            cf = Vector{Int32}(undef, nj)
            @inbounds for v in 1:nj
                cf[v] = code[cmH[v]]
            end
            code = cf; lab = Vector{UInt8}(undef, nj)
        end
        tr = time()
        for t in 0:(d - 1)
            mvs += kw_refine_depth!(lab, code, ptrH, adjH, vwH, t, o, seed + UInt32(16j + t))
        end
        tref += time() - tr
    end
    tpost = time()
    wait(dlt)
    codeD = CuVector{Int32}(code)
    st[:kw_post] = time() - tpost
    for j in 1:(L - 1)
        dfree!(levels[j + 1].g); CUDA.unsafe_free!(levels[j + 1].poff); CUDA.unsafe_free!(levels[j].cmap)
    end
    CUDA.unsafe_free!(poff); CUDA.unsafe_free!(Wp)
    st[:kw_coarsen] = t1 - t0; st[:kw_init] = tinit; st[:kw_ref] = tref; st[:kw_wait] = tw; st[:kw_levels] = L
    st[:kw_moves] = mvs; st[:kw_total] = time() - t0
    return codeD
end

# A/B/S labels of the current ND level from the codes: part p is tree node pnode[p]
function k_kwlab!(lab, code, gid, pid, pnode, n::Int32)
    v = gtid(); v > n && return
    @inbounds begin
        c = code[gid[v]]; x = pnode[pid[v]]
        if c == x
            lab[v] = SS
        else
            dc = Int32(31) - Int32(leading_zeros(reinterpret(UInt32, c))); t = Int32(31) - Int32(leading_zeros(reinterpret(UInt32, x)))
            ch = c >> (dc - t - Int32(1))
            lab[v] = iseven(ch) ? SA : SB
        end
    end
    return
end
