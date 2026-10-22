# Small-graph policy (ND_NOTES.md "Closing the AMD gap", item 5): pick GPU ND or a serial CPU ordering from
# predicted times so that the end-to-end time stays within `slack` x serial AMD.
# Times are modelled per machine from a warm-up calibration on grid graphs (no device constants):
#   serial alg a: t = c_a * (n + m)             (least squares through the origin)
#   GPU ND:       t = a0 + a1 * (n + m)          (fixed launch/latency floor + linear part)
# choice: GPU ND if t_gnd <= slack * t_amd; else the best-fill serial algorithm (MF1 > AMF > AMD) whose
# predicted time is within slack * t_amd. The size where GPU ND starts winning is n + m = a0 / (c_amd*slack - a1).
mutable struct PolicyCal
    c::Dict{String, Float64}
    a0::Float64
    a1::Float64
    xs::Vector{Float64}
    tg::Vector{Float64}
    ts::Dict{String, Vector{Float64}}
end
const PCAL = PolicyCal(Dict{String, Float64}(), NaN, NaN, Float64[], Float64[], Dict{String, Vector{Float64}}())
# race only below this size (n + m): twice the largest calibration size at which GPU ND was not yet within
# slack x AMD (racing big graphs costs GPU ND time: the serial thread's allocations trigger GC pauses)
function race_cap(; slack = 1.25)
    isempty(PCAL.xs) && return Inf
    lose = [x for (x, t, a) in zip(PCAL.xs, PCAL.tg, PCAL.ts["AMD"]) if t > slack * a]
    return isempty(lose) ? 0.0 : 2 * maximum(lose)
end

# least-squares fit t = a + b x with a, b >= 0
function fit_ab(xs, ts)
    n = length(xs)
    n == 0 && return NaN, NaN
    mx = sum(xs) / n; mt = sum(ts) / n
    sxx = sum((x - mx)^2 for x in xs); sxt = sum((x - mx) * (t - mt) for (x, t) in zip(xs, ts))
    b = sxx > 0 ? max(sxt / sxx, 0.0) : 0.0
    return max(mt - b * mx, 0.0), b
end

function policy_calibrate!(rungnd, runser; sizes = ((300, 300), (700, 700), (1000, 1000), (40, 40, 40), (80, 80, 80)), algs = ("AMD", "AMF", "MF1"))
    xs = Float64[]; tg = Float64[]; ts = Dict(a => Float64[] for a in algs)
    for d in sizes
        ptr, adj = GND.grid_csr(d)
        x = (length(ptr) - 1) + length(adj) / 2
        push!(xs, x)
        push!(tg, minimum(@elapsed(rungnd(ptr, adj)) for _ in 1:3))
        for a in algs
            push!(ts[a], minimum(@elapsed(runser(a, ptr, adj)) for _ in 1:2))
        end
    end
    for a in algs
        PCAL.c[a] = sum(xs .* ts[a]) / sum(xs .^ 2)
    end
    PCAL.xs = xs; PCAL.tg = tg; PCAL.ts = ts
    a0, a1 = fit_ab(xs, tg)
    PCAL.a0 = a0; PCAL.a1 = a1
    return PCAL
end

function policy_choose(n::Integer, m::Integer; slack = 1.25)
    x = n + m
    tamd = PCAL.c["AMD"] * x
    tg = PCAL.a0 + PCAL.a1 * x
    tg <= slack * tamd && return "gnd", tg, tamd
    for a in ("MF1", "AMF")
        haskey(PCAL.c, a) && PCAL.c[a] * x <= slack * tamd && return a, PCAL.c[a] * x, tamd
    end
    return "AMD", tamd, tamd
end
policy_crossover(; slack = 1.25) = (d = PCAL.c["AMD"] * slack - PCAL.a1; d > 0 ? PCAL.a0 / d : Inf)

# Race with a deadline: the serial algorithm (default AMF) runs on one CPU thread while GPU ND runs; once the
# serial order is ready at time ts, GPU ND may continue until slack * ts and is then abandoned at its next ND
# level boundary. Returns (perm, winner, tserial). The serial task, if it loses, finishes in the background.
function race_perm(rungnd, runser_, ptr, adj; slack = 1.25, serial = "AMF")
    T0 = time()
    abort = Threads.Atomic{Bool}(false)
    ser = Threads.@spawn begin
        p = runser_(serial, ptr, adj)
        (p, time() - T0)
    end
    done = Threads.Atomic{Bool}(false)
    watch = Threads.@spawn begin
        while !istaskdone(ser) && !done[]
            sleep(0.001)
        end
        if !done[]
            ts = fetch(ser)[2]
            while !done[] && time() - T0 < slack * ts
                sleep(0.001)
            end
            done[] || (abort[] = true)
        end
    end
    pg = rungnd(ptr, adj, abort)
    done[] = true
    if pg !== nothing
        return pg, "gnd", ser
    end
    p, ts = fetch(ser)
    return p, serial, ser
end
