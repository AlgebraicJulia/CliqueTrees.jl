# Summaries of the per-round phase profile of GRounds (rprof > 0; LATENCY_NOTES.md).
# pf = timing[:pf] (PFS × NPH × RC Int64), see the slot list in GRounds.jl.
const PHN = (:scan, :reb1, :reb2, :cand, :luby2, :sizes, :elim12, :ob, :whg, :elim34, :detect, :apply)
const BLKPH = (:detect, :apply)          # block-per-item phases

# µs per clock cycle from slots 9/10 (block durations in cycles and ns), else the attribute clock
function pfcyc2us(pf, R, khz)
    size(pf, 1) < 10 && return 1e3 / khz
    c = 0.0; t = 0.0
    for r in 1:R, k in 1:size(pf, 2)
        c += pf[9, k, r]; t += pf[10, k, r]
    end
    return c > 0 && t > 0 ? (t / c) / 1e3 : 1e3 / khz
end

# per phase: rounds run, wall total (ms), mean wall (µs), items, mean item (µs), mean of the
# per-round max item (µs), maxshare = Σ max item / Σ wall, balshare = Σ (Σ item / resident
# warps) / Σ wall, plus the round totals (wall from the first start to the last end) and the
# gaps between phases (launch / graph-node latency)
function pfsummary(pf::Array{Int64, 3}, nrounds::Integer, grids, khz::Integer; tpb = 256)
    R = min(Int(nrounds), size(pf, 3))
    cyc2us = pfcyc2us(pf, R, khz)           # one cycle in µs (measured clock rate)
    rows = NamedTuple[]
    for (k, nm) in enumerate(PHN)
        G = hasproperty(grids, nm) ? getproperty(grids, nm) : 0
        res = nm in BLKPH ? G : G * (tpb ÷ 32)
        nr = 0; wall = 0.0; items = 0; sumc = 0.0; maxs = 0.0; bal = 0.0; mxi = 0.0; maxsz = 0
        for r in 1:R
            t0 = pf[1, k, r]; t1 = pf[2, k, r]
            (t0 == typemax(Int64) || t1 == 0) && continue
            nr += 1
            w = (t1 - t0) / 1e3
            wall += w
            items += pf[3, k, r]
            sumc += pf[4, k, r] * cyc2us
            maxs += pf[5, k, r] * cyc2us
            mxi = max(mxi, pf[5, k, r] * cyc2us)
            maxsz = max(maxsz, pf[6, k, r])
            res > 0 && (bal += pf[4, k, r] * cyc2us / res)
        end
        nr == 0 && continue
        push!(rows, (phase = nm, rounds = nr, wall_ms = wall / 1e3, wall_us = wall / nr, items_per_round = items / nr,
            item_us = items > 0 ? sumc / items : 0.0, maxitem_us = maxs / nr, maxshare = wall > 0 ? maxs / wall : 0.0,
            balshare = wall > 0 ? bal / wall : 0.0, worst_item_us = mxi, max_size = maxsz))
    end
    # round wall and gaps
    rw = 0.0; pw = 0.0; nr = 0
    for r in 1:R
        s = typemax(Int64); e = 0; ps = 0
        for k in 1:length(PHN)
            t0 = pf[1, k, r]; t1 = pf[2, k, r]
            (t0 == typemax(Int64) || t1 == 0) && continue
            s = min(s, t0); e = max(e, t1); ps += t1 - t0
        end
        e == 0 && continue
        nr += 1; rw += (e - s) / 1e3; pw += ps / 1e3
    end
    return rows, (ghz = 1e-3 / cyc2us, rounds = nr, round_us = rw / max(nr, 1), phases_us = pw / max(nr, 1), gap_us = (rw - pw) / max(nr, 1))
end

# compact per-round matrix (Float32): round wall µs, then per phase (wall µs, max item µs, items)
function pfrounds(pf::Array{Int64, 3}, nrounds::Integer, khz::Integer)
    R = min(Int(nrounds), size(pf, 3)); P = length(PHN)
    M = zeros(Float32, R, 1 + 3P)
    cyc2us = pfcyc2us(pf, R, khz)
    for r in 1:R
        s = typemax(Int64); e = 0
        for k in 1:P
            t0 = pf[1, k, r]; t1 = pf[2, k, r]
            (t0 == typemax(Int64) || t1 == 0) && continue
            s = min(s, t0); e = max(e, t1)
            M[r, 1 + 3(k - 1) + 1] = (t1 - t0) / 1e3
            M[r, 1 + 3(k - 1) + 2] = pf[5, k, r] * cyc2us
            M[r, 1 + 3(k - 1) + 3] = pf[3, k, r]
        end
        e > 0 && (M[r, 1] = (e - s) / 1e3)
    end
    return M
end

function pfprint(io, name, cfg, rows, tot)
    println(io, "## $name $cfg  GHz=$(round(tot.ghz, digits=2)) rounds=$(tot.rounds) round=$(round(tot.round_us, digits=1))µs phases=$(round(tot.phases_us, digits=1))µs gaps=$(round(tot.gap_us, digits=1))µs")
    println(io, rpad("phase", 8), lpad("rounds", 8), lpad("wall_ms", 10), lpad("wall_us", 9), lpad("items", 9), lpad("item_us", 9),
        lpad("maxit_us", 9), lpad("maxshr", 8), lpad("balshr", 8), lpad("worst_us", 10), lpad("maxsz", 9))
    for r in rows
        println(io, rpad(string(r.phase), 8), lpad(r.rounds, 8), lpad(round(r.wall_ms, digits = 2), 10), lpad(round(r.wall_us, digits = 2), 9),
            lpad(round(r.items_per_round, digits = 1), 9), lpad(round(r.item_us, digits = 2), 9), lpad(round(r.maxitem_us, digits = 2), 9),
            lpad(round(r.maxshare, digits = 2), 8), lpad(round(r.balshare, digits = 3), 8), lpad(round(r.worst_item_us, digits = 1), 10),
            lpad(r.max_size, 9))
    end
end
