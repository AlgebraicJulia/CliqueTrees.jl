p='src/BlockAnneal.jl'; s=open(p).read()
a='''# exact evaluation with slice mask SM'''
b='''@inline function bmin(x::T, RED, tid, nth) where {T}
    @inbounds RED[tid] = x
    sync_threads()
    s = nth >> Int32(1)
    while s > Int32(0)
        if tid <= s
            @inbounds RED[tid] = min(RED[tid], RED[tid+s])
        end
        sync_threads()
        s >>= Int32(1)
    end
    r = @inbounds RED[1]
    sync_threads()
    r
end

# TreeSASlicer's leg scores, block-parallel: for every index, the number of maximum-size intermediates
# (size > mw - 0.99) whose legs contain it. Returns (best unsliced index to add: most frequent; best sliced
# index to remove: least frequent), 0 if none; ties broken at random.
@inline function slicescores(L, SM, CM, cw, K, W, nt, nn, mw, tid, nth, CNT, g, RU64, rs)
    i = tid
    while i <= W * Int32(64)
        @inbounds CNT[i, g] = Int32(0)
        i += nth
    end
    sync_threads()
    v = nt + tid
    while v <= nn
        sz = 0.0f0
        for w in Int32(1):W
            @inbounds sz += wset(L[w, v] & ~SM[w], w, CM, cw, K)
        end
        if sz > mw - 0.99f0
            for w in Int32(1):W
                @inbounds x = L[w, v]
                while x != UInt64(0)
                    b = Int32(trailing_zeros(x))
                    CUDA.atomic_add!(pointer(CNT, (g - Int32(1)) * W * Int32(64) + (w - Int32(1)) * Int32(64) + b + Int32(1)), Int32(1))
                    x &= x - UInt64(1)
                end
            end
        end
        v += nth
    end
    sync_threads()
    ka = UInt64(0); kr = typemax(UInt64)
    i = tid
    while i <= W * Int32(64)
        @inbounds c = CNT[i, g]
        @inbounds insl = ((SM[(i-Int32(1))>>6+Int32(1)] >> ((i - Int32(1)) & Int32(63))) & UInt64(1)) == UInt64(1)
        rs, r = xs(rs)
        key = (UInt64(c) << 40) | (((r >> 48) & UInt64(0xffff)) << 24) | UInt64(i)
        if insl
            kr = min(kr, key)
        elseif c > Int32(0)
            ka = max(ka, key)
        end
        i += nth
    end
    ka = bmax(ka, RU64, tid, nth)
    kr = bmin(kr, RU64, tid, nth)
    ia = Int32(ka & UInt64(0xffffff))
    ir = kr == typemax(UInt64) ? Int32(0) : Int32(kr & UInt64(0xffffff))
    ia, ir, rs
end

# exact evaluation with slice mask SM'''
assert a in s; s=s.replace(a,b,1)
rep=[
("    RF = CuDynamicSharedArray(Float32, nth, off); off += 4 * nth\n",
 "    RF = CuDynamicSharedArray(Float32, nth, off); off += 4 * nth\n    off = (off + 7) & ~7\n    RU64 = CuDynamicSharedArray(UInt64, nth, off); off += 8 * nth\n"),
("    b = 8nth + 4nth + 32 + 16W + 8W * K + 4K + 16\n", "    b = 8nth + 4nth + 8 + 8nth + 32 + 16W + 8W * K + 4K + 16\n"),
("function block_kernel!(Lg, CHg, SLg, STg, RNGg, BESTE, BESTCH, BESTS, NACC, NPROP, CMg, cwg, LBETA, RU, RL,",
 "function block_kernel!(Lg, CHg, SLg, STg, RNGg, BESTE, BESTCH, BESTS, NACC, NPROP, CMg, cwg, LBETA, RU, RL, CNT,"),
("nreconf::Int32, Wf::Float32) where {SH}", "nreconf::Int32, Wf::Float32, pheur::Float32) where {SH}"),
("""                for _ in Int32(1):nslice
                    # thread 1 proposes S' in SM2
                    if tid == Int32(1)
                        rs2 = rs
                        rs2, r2 = xs(rs2)
                        nS = Int32(0)
                        for w in Int32(1):W
                            @inbounds SM2[w] = SM[w]; @inbounds nS += Int32(count_ones(SM[w]))
                        end
                        op = nS == 0 ? Int32(0) : rint(r2, 3)
                        good = true
                        if op != Int32(0)""",
"""                for _ in Int32(1):nslice
                    ia = Int32(0); ir = Int32(0)
                    if pheur > 0.0f0
                        ia, ir, rs = slicescores(L, SM, CM, cw, K, W, nt, nn, mw, tid, nth, CNT, g, RU64, rs)
                    end
                    # thread 1 proposes S' in SM2
                    if tid == Int32(1)
                        rs2 = rs
                        rs2, r2 = xs(rs2)
                        nS = Int32(0)
                        for w in Int32(1):W
                            @inbounds SM2[w] = SM[w]; @inbounds nS += Int32(count_ones(SM[w]))
                        end
                        op = nS == 0 ? Int32(0) : rint(r2, 3)
                        good = true
                        rs2, r3 = xs(rs2)
                        if u01(r3) < pheur && ia > Int32(0)
                            # TreeSASlicer moves: add the most frequent leg of the widest nodes while over
                            # the cap; at the cap: remove the least useful slice, or swap it for the best leg
                            if mw > Wt + 1.0f-3 || nS == 0 || ir == Int32(0)
                                op = Int32(-1)
                            else
                                op = (r3 & 1) == 1 ? Int32(-2) : Int32(-3)
                            end
                            if op == Int32(-1) || op == Int32(-3)
                                @inbounds SM2[(ia-Int32(1))>>6+Int32(1)] |= UInt64(1) << ((ia - Int32(1)) & Int32(63))
                            end
                            if op == Int32(-2) || op == Int32(-3)
                                @inbounds SM2[(ir-Int32(1))>>6+Int32(1)] &= ~(UInt64(1) << ((ir - Int32(1)) & Int32(63)))
                            end
                        end
                        if op == Int32(1) || op == Int32(2)"""),
("""                        if op != Int32(1)
                            bv = Int32(0); bsz = -1.0f0""", """                        if op == Int32(0) || op == Int32(2)
                            bv = Int32(0); bsz = -1.0f0"""),
("    RL::CuArray{UInt64,3}\nend", "    RL::CuArray{UInt64,3}\n    CNT::CuMatrix{Int32}\nend"),
("CUDA.zeros(UInt64, W, 256, nch), CUDA.zeros(UInt64, W, 256, nch))", "CUDA.zeros(UInt64, W, 256, nch), CUDA.zeros(UInt64, W, 256, nch), CUDA.zeros(Int32, 64W, nch))"),
("A.cmask, A.cw, A.LBETA, A.RU, A.RL, Val(A.shared),", "A.cmask, A.cw, A.LBETA, A.RU, A.RL, A.CNT, Val(A.shared),"),
("Int32(nreconf), Float32(Wf))", "Int32(nreconf), Float32(Wf), Float32(pheur))"),
("Int32(nreconf), Float32(Wf);", "Int32(nreconf), Float32(Wf), Float32(pheur);"),
("evalevery = 64, nslice = 0, kmax = 8, nreconf = 0, Wf = Wt)", "evalevery = 64, nslice = 0, kmax = 8, nreconf = 0, Wf = Wt, pheur = 0.0)"),
("                 lam1 = lam, Wt0 = Wt, wfrac = 0.6)", "                 lam1 = lam, Wt0 = Wt, wfrac = 0.6, pheur = 0.0)"),
("        step!(A, rounds, beta, nothing; lam = lm, Wt = we, mix, evalevery, nslice, kmax, nreconf, Wf = Wt)",
 "        step!(A, rounds, beta, nothing; lam = lm, Wt = we, mix, evalevery, nslice, kmax, nreconf, Wf = Wt, pheur)"),
]
for a,b in rep:
    n=s.count(a); assert n>=1, a
    s=s.replace(a,b)
open(p,'w').write(s)
p='scripts/pipeline.jl'; t=open(p).read()
a='wfrac = gf("wfrac", 0.6), nslice,'
assert a in t
t=t.replace(a,'wfrac = gf("wfrac", 0.6), nslice, pheur = gf("pheur", 0.75),')
open(p,'w').write(t)
print("ok")
