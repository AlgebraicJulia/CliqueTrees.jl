#
# The rounds of the exact searches on the GPU (see bt/backend.jl).
#
#   - SepEnumGPU and SepDPGPU are sepenum and sepdp: the same lists of
#     separators in the same order, the same blocks, values, and caps,
#     and the same orderings. A set is a word (n <= 64) or a tuple of
#     W words (n <= 64W), up to NMAX vertices.
#   - PIDGPU decides the rounds by a positive-instance driven search
#     (Tamaki 2017), downward from a greedy ordering (see pidround!);
#     its orderings are not those of sepdp.
#
# A round of a search with the backend GPUSeptw asks for the GPU with a
# request (Req) in a queue, which a worker on an interactive thread serves,
# and runs on the CPU in the meantime (race). The first of the two to
# finish answers the round: the GPU stops the CPU round through the flag of
# its clock, and the CPU stops the GPU round through the flag stop of the
# request. A request is served once it is ripe: when the CPU time of its
# round is expected to exceed its GPU time (see ripe); of the ripe
# requests, the one with the largest expected gain goes first, and with
# it, for sepenum and sepdp, the other ripe requests whose sets have the
# same number of words: a batch, served by one launch of each kernel (see
# septwbatch), in which each request keeps its deadline and its stop
# (unless the backend says batch = false).
#
# The expected times come from a model learned from the rounds measured so
# far, by the size of the graph (see predict): of the CPU rounds, and of the
# GPU rounds of each engine, by the size of the batch: the seconds of a
# batch shared by its requests (with engine = :auto, a round asks the engine
# with the least expected time, see engine); before the first GPU round of
# a size, the cost of a GPU round on a small graph, measured on the device
# (cost!), so that every size is tried. The CPU time of a round is expected
# to be at least what it has taken so far, what the growth of the last two
# rounds of the search says, and what the rounds of the size take.
#
# Without the race, or without an interactive thread for the worker, a
# round asks for the GPU when it starts, if its expected CPU time exceeds its expected GPU
# time and the wait for the GPU, and then waits for it (without an
# interactive thread: runs on it, if it is free, and on the CPU if not).
#
# A round that fails on the GPU (out of memory, a stack or a bag too large
# for the device code) is run on the CPU, and so are the later rounds of
# the search.
#
module CUDAExt

using CUDA
using CliqueTrees
using CliqueTrees: GPUSeptw, BTStatus, BT_FOUND, BT_ABOVE, BT_TIMEOUT, bt_clk, bt_int, bt_next, sepenum, sepdp
using Graphs
using Random

include("bits.jl")
include("sepenum.jl")
include("sepdp.jl")
include("pid/pid.jl")

using .GPUBits: Bits
using .SepEnumGPU: SepEnumGPU
using .SepDPGPU: SepDPGPU
using .PIDGPU: PIDGPU

# the largest graph of SepEnumGPU and SepDPGPU
const NMAX = 512

# the device: held by the worker while it serves a request, by a round that
# runs on the GPU itself (no worker), and by the measurements
const LOCK = ReentrantLock()

# the cost of a GPU round on a small graph, for each engine (see cost!),
# and whether the device has failed for good
const COST = Dict{Symbol, Float64}()
const BROKEN = Ref(false)

# counts of the rounds: cpu (finished on the CPU), gpuwon (answered by the
# GPU), cpuwon (finished on the CPU while the GPU served it), known
# (answered by an earlier PID round), escalated (requests served),
# gpufound, gpuabove, gputimeout, gpufailed (their outcomes), expired (past
# the deadline in the queue), busy (the GPU was held, no worker), broken (1
# once the device has failed for good), fallback (searches and rounds kept
# on the CPU after a failure of the GPU); and the
# seconds of the rounds on the CPU (tcpu), on the GPU (tgpu), and of the
# requests in the queue (twait)
const STATS = Dict{Symbol, Float64}()
const STATSLOCK = Threads.SpinLock()

function count!(key::Symbol, x::Real = 1)
    lock(STATSLOCK) do
        STATS[key] = get(STATS, key, 0.0) + x
    end

    return
end

function resetstats!()
    lock(STATSLOCK) do
        empty!(STATS)
    end

    return
end

usable() = !BROKEN[] && !CLOSED[] && CUDA.functional()

ispositive(x::Real) = x > zero(x)

# ------------------------------------------------------------------ the model
# The seconds of the rounds, by engine (:cpu for the CPU rounds), by the
# size of the graph (bins of n vertices a factor 2^(1/4) apart), and by the
# size of the batch (bins of B requests a factor 2 apart): for a batch, the
# seconds of the batch over B, the GPU time that each of its requests
# costs. A bin keeps a running mean of the log seconds (over the last ten
# rounds or so), and the bin of single rounds counts the GPU rounds served
# and failed.
mutable struct Bin
    lg::Float64
    n::Int
    served::Int
    failed::Int
end

const MODEL = Dict{Tuple{Symbol, Int, Int}, Bin}()
const MODELLOCK = Threads.SpinLock()

sizebin(n::Integer) = round(Int, 4 * log2(max(n, 1)))
batchbin(B::Integer) = floor(Int, log2(max(B, 1)))

function resetmodel!()
    lock(MODELLOCK) do
        empty!(MODEL)
    end

    return
end

# A round of t seconds on a graph of n vertices, in a batch of B requests;
# censored: the round was stopped (by its deadline, or by the other side of
# the race), and would have taken more: it counts as 2t (a round that has
# run for t is expected to run as long again), if that is more than the
# mean, so that a size whose GPU rounds keep losing is tried only on longer
# and longer rounds.
function observe!(key::Symbol, n::Integer, t::Real; B::Integer = 1, censored::Bool = false)
    x = log(max(t, 1.0e-6)) + (censored ? log(2.0) : 0.0)

    lock(MODELLOCK) do
        b = get!(() -> Bin(0.0, 0, 0, 0), MODEL, (key, sizebin(n), batchbin(B)))
        censored && ispositive(b.n) && x <= b.lg && return
        b.n += 1; b.lg += (x - b.lg) / min(b.n, 10)
    end

    return
end

function served!(key::Symbol, n::Integer, failed::Bool)
    lock(MODELLOCK) do
        b = get!(() -> Bin(0.0, 0, 0, 0), MODEL, (key, sizebin(n), 0))
        b.served += 1; b.failed += failed
    end

    return
end

# The expected seconds of a round on a graph of n vertices in a batch of B
# requests: NaN if no batch of at most B requests of the size has been
# measured yet, and Inf if most of the GPU rounds of the size failed. A
# batch larger than those measured is expected to take as long as the
# largest of them, shared by more requests (so that larger batches are
# tried). (The times of other sizes say little: a size learns its own.)
function predict(key::Symbol, n::Integer; B::Integer = 1)
    lock(MODELLOCK) do
        sb = sizebin(n); bb = batchbin(B)
        b = get(MODEL, (key, sb, 0), nothing)
        !isnothing(b) && b.failed >= 2 && 2b.failed > b.served && return Inf

        for c in bb:-1:0
            b = get(MODEL, (key, sb, c), nothing)
            !isnothing(b) && ispositive(b.n) && return exp(b.lg) * exp2(c - bb)
        end

        return NaN
    end
end

# The expected GPU seconds of a round on a graph of n vertices on the
# engine e, in a batch of B requests (before any round of the size: the
# cost of a round on a small graph).
function gpuguess(e::Symbol, n::Integer, B::Integer = 1)
    x = predict(e, n; B)
    isnan(x) && (x = get(COST, e, NaN) * exp2(-batchbin(B)))
    return x
end

# ------------------------------------------------------------------ the queue
# The state of the rounds of a search: the backend, the number of vertices,
# whether a GPU round has failed (the rest on the CPU), the seconds of the
# last two CPU rounds and of all of them, the state of PIDGPU (made at the
# first PID round), and the last request.
mutable struct Esc
    backend::GPUSeptw
    n::Int
    off::Bool
    t1::Float64
    t2::Float64
    spent::Float64
    pid::Any
    req::Any
end

Esc(backend::GPUSeptw, n::Integer) = Esc(backend, n, false, 0.0, 0.0, 0.0, nothing, nothing)

# A request for the GPU: the round of a search at the width k, with the
# deadline dl, asked at the time t0, when the CPU had spent spent seconds
# on the search; est and g are the expected CPU and GPU seconds of the
# round, on the GPU engine engine. The task sets stop when it no longer needs an answer. The worker
# sets started when it takes the request, fin when it is done with it
# (the answer, or :failed, in result; then done is notified), and then, if
# the GPU answered, cpu: the flag of the clock of the CPU round.
mutable struct Req
    esc::Esc
    engine::Symbol
    weights::AbstractVector
    graph::AbstractGraph
    k::Real
    dl::Float64
    t0::Float64
    spent::Float64
    est::Float64
    g::Float64
    ts::Float64
    result::Any
    stop::Threads.Atomic{Bool}
    cpu::Threads.Atomic{Bool}
    started::Threads.Atomic{Bool}
    fin::Threads.Atomic{Bool}
    done::Base.Event
end

function Req(esc::Esc, engine::Symbol, weights::AbstractVector, graph::AbstractGraph, k::Real, dl::Real, t0::Float64, est::Float64, g::Float64)
    return Req(esc, engine, weights, graph, k, Float64(dl), t0, esc.spent, est, g, NaN, nothing,
        Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(false), Threads.Atomic{Bool}(false), Base.Event())
end

const QUEUE = Req[]
# (spin locks, here and for the statistics and the model: the worker must
# not wait for a task to wake it, see work)
const QLOCK = Threads.SpinLock()
const CURRENT = Req[]
const WORKER = Ref{Union{Nothing, Task}}(nothing)

# Whether the GPU is closed (see quiesce!): no round starts on it.
const CLOSED = Ref(false)

# Close the GPU: stop the requests in the queue and in service, and wait,
# at most timeout seconds, until the device is free; then hold it (LOCK),
# so that no round starts on it. Returns whether the device is held (then
# resume!(true) opens it again). The extension closes the GPU when the
# process exits: the CUDA driver destroys its context after the hooks of
# atexit, and a round still on the device (on the worker, which runs on
# while the main task exits) would fail with "context is destroyed".
function quiesce!(; timeout::Real = 10.0)
    CLOSED[] = true

    lock(QLOCK) do
        foreach(r -> r.stop[] = true, QUEUE)
        foreach(r -> r.stop[] = true, CURRENT)
        purge!(time())
    end

    t = time() + timeout

    while !trylock(LOCK)
        time() > t && return false
        Libc.systemsleep(0.001); yield()
    end

    return true
end

function resume!(held::Bool)
    held && unlock(LOCK)
    CLOSED[] = false
    return
end

function __init__()
    atexit() do
        try
            quiesce!()
        catch
        end
    end

    return
end

# Start the worker, on an interactive thread (the tasks of the searches do
# not yield, and would hold it up on a thread of their own). The main task
# may run on an interactive thread (thread 1, as from Julia 1.12) and search
# there, so the worker needs another one. False if there is none.
function worker!()
    spare = Threads.nthreads(:interactive) - (Threads.threadpool(1) === :interactive)
    spare > 0 || return false

    lock(QLOCK) do
        if isnothing(WORKER[]) || istaskdone(WORKER[])
            WORKER[] = Threads.@spawn :interactive work()
        end
    end

    return true
end

# (a request to a closed GPU is finished at once, without an answer)
function submit!(r::Req)
    lock(QLOCK) do
        CLOSED[] ? finish!(r) : push!(QUEUE, r)
    end

    return
end

# The class of a request: the requests of a class can share a batch. Those
# of sepenum and sepdp by the number of words of their sets; PID rounds are
# not batched (class 0).
class(r::Req) = r.engine === :septw && r.esc.backend.batch ? nwords(settype(r.esc.n)) : 0

# The number of requests in the queue of the class c (the caller holds
# QLOCK).
function inclass(c::Integer)
    m = 0

    for r in QUEUE
        !r.stop[] && class(r) == c && (m += 1)
    end

    return m
end

queued(c::Integer) = lock(() -> inclass(c), QLOCK)

# The expected CPU seconds of the round of r, at the time t.
cpuguess(r::Req, t::Float64) = max(t - r.t0, r.est)

# Whether the GPU should serve r at the time t: if the round waits for it;
# with a budget, if the CPU has spent it on the search; otherwise, if the
# CPU time of the round is expected to exceed its GPU time.
function ripe(r::Req, t::Float64)
    b = r.esc.backend
    b.race || return true
    isnan(b.budget) || return r.spent + t - r.t0 >= b.budget
    return cpuguess(r, t) >= r.g
end

# The expected seconds until the GPU is free, at the time t: the rest of the
# batch in service, and the ripe requests in the queue.
function backlog(t::Float64)
    lock(QLOCK) do
        x = 0.0

        for r in CURRENT
            x += r.g
        end

        isempty(CURRENT) || (x = max(0.0, x - (t - CURRENT[1].ts)))

        for r in QUEUE
            !r.stop[] && t <= r.dl && ripe(r, t) && (x += r.g)
        end

        return x
    end
end

answered(r::Req) = r.fin[] && r.result isa Tuple && r.result[3] != BT_TIMEOUT

function finish!(r::Req)
    r.fin[] && return
    r.fin[] = true; notify(r.done)
    answered(r) && (r.cpu[] = true)
    return
end

# Drop the requests that are stopped or past their deadlines from the queue,
# and say whether one of the others is ripe at the time t (the caller holds
# QLOCK).
function purge!(t::Float64)
    i = 1; any = false

    while i <= length(QUEUE)
        q = QUEUE[i]

        if q.stop[] || t > q.dl
            q.stop[] || count!(:expired)
            deleteat!(QUEUE, i); finish!(q)
            continue
        end

        any |= ripe(q, t); i += 1
    end

    return any
end

# The next batch, at the time t (the caller holds QLOCK): the ripe request
# with the largest expected gain (CPU seconds less GPU seconds), and the
# other ripe requests of its class, the largest gains first, as many as the
# memory of the device takes (nmax(W) vertices for sets of W words). The
# GPU seconds of a request are those of its share of a batch of all the
# ripe requests of its class (with the seconds of a batch of all the
# requests of the class in the queue, at first: more requests make each
# cheaper).
function select!(t::Float64, nmax)
    purge!(t)

    for q in QUEUE
        q.g = gpuguess(q.engine, q.esc.n, iszero(class(q)) ? 1 : inclass(class(q)))
    end

    best = 0; gain = -Inf

    for (i, q) in enumerate(QUEUE)
        if ripe(q, t)
            x = cpuguess(q, t) - q.g
            x > gain && (best = i; gain = x)
        end
    end

    iszero(best) && return Req[]
    c = class(QUEUE[best])

    if iszero(c)
        batch = [QUEUE[best]]
    else
        cand = [q for q in QUEUE if class(q) == c && ripe(q, t)]

        # (with the size of the batch of the ripe requests, fewer may be ripe)
        for q in cand
            q.g = gpuguess(q.engine, q.esc.n, length(cand))
        end

        batch = [q for q in cand if q === QUEUE[best] || ripe(q, t)]
        sort!(batch; by = q -> q === QUEUE[best] ? Inf : cpuguess(q, t) - q.g, rev = true)
        m = 0; j = 0

        while j < length(batch) && (iszero(j) || m + batch[j + 1].esc.n <= nmax(c))
            j += 1; m += batch[j].esc.n
        end

        resize!(batch, j)

        for q in batch
            q.g = gpuguess(q.engine, q.esc.n, j)
        end
    end

    filter!(q -> !any(r -> r === q, batch), QUEUE)

    for q in batch
        q.started[] = true; q.ts = t
    end

    append!(empty!(CURRENT), batch)
    return batch
end

# The most vertices of a batch of sepenum with sets of W words, with free
# bytes of device memory: the stacks of its threads (one for each vertex,
# see SepEnumGPU.gpu_sepenum) take at most a quarter of it.
batchmax(free::Integer, W::Integer) = max(1, (free ÷ 4) ÷ (8 * 1024 * W^2))

# The worker: serve the next batch, while there is one; requests that are
# stopped or past their deadlines leave the queue. The worker polls: it
# sleeps with the thread (the timers of sleep need the event loop, which a
# busy main thread may not run), and yields after, since it may be on the
# thread of the main task, which waits for it and cannot move. The batch
# is chosen once the device is free. The worker ends when the GPU is
# closed (see quiesce!).
function work()
    while !CLOSED[]
        any = lock(() -> purge!(time()), QLOCK)

        # (a request ripens as its CPU round goes on)
        if !any
            Libc.systemsleep(0.0005); yield()
            continue
        end

        while !trylock(LOCK)
            CLOSED[] && return
            Libc.systemsleep(0.001); yield()
        end

        if CLOSED[]
            unlock(LOCK)
            return
        end

        batch = Req[]

        try
            free = CUDA.free_memory()
            batch = lock(() -> select!(time(), W -> batchmax(free, W)), QLOCK)
            isempty(batch) || run!(batch)
        finally
            unlock(LOCK)

            lock(QLOCK) do
                empty!(CURRENT)
            end

            foreach(finish!, batch)
        end
    end
end

run!(r::Req) = run!([r])

# Serve the batch of requests on the GPU (the caller holds LOCK): the
# result of each is the answer of its round, or :failed; it is finished as
# soon as its answer is known.
function run!(batch::Vector{Req})
    ts = time(); B = length(batch); engine = batch[1].engine
    count!(:batches); count!(:escalated, B); count!(engine, B)
    B > 1 && count!(:batched, B)

    lock(STATSLOCK) do
        STATS[:maxbatch] = max(get(STATS, :maxbatch, 0.0), B)
    end

    for r in batch
        count!(:twait, ts - r.t0)
    end

    function done(r::Req, result)
        r.result = result

        if result isa Tuple
            status = result[3]
            count!(status == BT_FOUND ? :gpufound : status == BT_ABOVE ? :gpuabove : :gputimeout)
        else
            count!(:gpufailed)
        end

        finish!(r)
        return
    end

    gpuround!(batch, done)
    t = time() - ts; count!(:tgpu, t)

    for r in batch
        esc = r.esc; result = r.result
        result isa Tuple && observe!(engine, esc.n, t / B; B, censored = result[3] == BT_TIMEOUT)
        served!(engine, esc.n, result === :failed)
        result === :failed && (esc.off = true)
    end

    return batch[1].result
end

# A batch of rounds on the GPU (the caller holds LOCK); done(r, result) is
# called with the result of each request r, (perm, next, status) or :failed,
# as soon as it is known.
function gpuround!(batch::Vector{Req}, done)
    try
        if batch[1].engine === :pid
            r = only(batch)
            done(r, pidround!(r.esc, r.weights, r.graph, r.k, r.dl, r.stop))
        else
            n = maximum(r -> r.esc.n, batch); T = settype(n)
            cut(i) = batch[i].stop[] || time() > batch[i].dl
            clks = [(; deadline = r.dl, tick = Ref(0), out = r.stop) for r in batch]

            septwbatch(T, [r.weights for r in batch], [r.graph for r in batch], [r.k for r in batch], cut, clks;
                done = (i, x) -> done(batch[i], answer(batch[i].weights, batch[i].graph, batch[i].k, x)))
        end
    catch err
        err isa InterruptException && rethrow()
        failed!(err)

        for r in batch
            r.fin[] || done(r, :failed)
        end
    end

    return
end

# After a failure of the device (called in the catch block, for the
# backtrace): out of memory is not fatal, but an error of a kernel spoils
# the context, and the GPU is not used again. The first failure of the
# device is logged as an error, with its backtrace, and the statistics say
# that the device is broken (broken = 1; fallback counts the searches and
# rounds that stayed on the CPU because of it).
function failed!(err, bt = catch_backtrace())
    # (a round stopped by the exit of the process)
    if CLOSED[]
        @debug "septw: GPU round failed while the GPU closes" exception = (err, bt)
        return
    end

    if err isa CUDA.OutOfGPUMemoryError || err isa ErrorException
        @debug "septw: GPU round failed" exception = (err, bt)
    else
        broken!(err, bt)
    end

    try
        GC.gc(false); CUDA.reclaim()
    catch err2
        broken!(err2, catch_backtrace())
    end

    return
end

function broken!(err, bt)
    BROKEN[] && return
    BROKEN[] = true; count!(:broken)
    @error "septw: the GPU failed for good (thread $(Threads.threadid()), pool $(Threads.threadpool())); the exact searches stay on the CPU" exception = (err, bt)
    return
end

# ------------------------------------------------------------------ the rounds
function CliqueTrees.bt_esc(backend::GPUSeptw, weights::AbstractVector, graph::AbstractGraph, dl::Real)
    n = nv(graph)
    n < max(backend.nmin, 2) && return nothing
    BROKEN[] && (count!(:fallback); return nothing)
    usable() || return nothing
    any(e -> !isnan(cost!(e)), engines(backend, n)) || return nothing
    return Esc(backend, n)
end

# The engines that may take a graph of n vertices: with :auto, PID, and
# sepenum and sepdp up to NMAX vertices.
function engines(backend::GPUSeptw, n::Integer)
    backend.engine === :auto && return n <= NMAX ? (:septw, :pid) : (:pid,)
    backend.engine === :septw && n > NMAX && return ()
    return (backend.engine,)
end

# The engine with the least expected GPU seconds for a graph of n
# vertices, and these seconds (an engine not yet tried on the size is
# expected to take its cost on a small graph, so that it is tried). The
# rounds of sepenum and sepdp would share a batch with those of their class
# in the queue.
function engine(backend::GPUSeptw, n::Integer)
    best = :none; g = Inf

    for e in engines(backend, n)
        x = gpuguess(e, n, e === :septw && backend.batch ? 1 + queued(nwords(settype(n))) : 1)
        !isnan(x) && x < g && (best = e; g = x)
    end

    return best, g
end

# The growth of the last two CPU rounds of a search: the expected seconds
# of the next.
growth(esc::Esc) = ispositive(esc.t1) ? max(2.0, ispositive(esc.t2) ? esc.t1 / esc.t2 : 2.0) * esc.t1 : 0.0

# A round answered by the earlier PID rounds of the search: the bound ub is
# at most k, or it is the width.
function known(esc::Esc, k)
    p = esc.pid
    isnothing(p) && return nothing
    p.ub[] <= k && return p.order[], p.ub[], BT_FOUND
    p.proved[] && return Int[], p.ub[], BT_ABOVE
    return nothing
end

function CliqueTrees.bt_round!(esc::Esc, weights::AbstractVector{W}, graph::AbstractGraph, k::W, dl::Real, nt::Integer) where {W}
    # (the GPU may still be on the last request of the search, which the
    # CPU answered first)
    last = esc.req; free = isnothing(last) || last.fin[]

    if free
        result = known(esc, k)
        isnothing(result) || (count!(:known); return result)
    end

    ts = time(); b = esc.backend; req = nothing

    e, g = engine(b, esc.n)
    (esc.off || BROKEN[]) && count!(:fallback)

    if free && !esc.off && !BROKEN[] && !CLOSED[] && e !== :none
        c = predict(:cpu, esc.n); isnan(c) && (c = 0.0)
        req = esc.req = Req(esc, e, weights, graph, k, dl, ts, max(c, growth(esc)), g)

        if b.race && worker!()
            submit!(req)
        else
            # ask for the GPU now, or not at all
            ask = isnan(b.budget) ? req.est >= g + backlog(ts) : esc.spent >= b.budget

            if ask
                if worker!()
                    submit!(req); wait(req.done); result = req.result
                elseif trylock(LOCK)
                    try
                        result = run!(req)
                    finally
                        unlock(LOCK); finish!(req)
                    end
                else
                    count!(:busy); result = nothing
                end

                if result isa Tuple && result[3] != BT_TIMEOUT
                    count!(:gpuwon)
                    return result
                end

                time() > dl && return Int[], k, BT_TIMEOUT
            end

            req = nothing
        end
    end

    # the round on the CPU: the GPU stops it if it answers first
    clk = (; deadline = Float64(dl), tick = Ref(0), out = isnothing(req) ? Threads.Atomic{Bool}(false) : req.cpu)
    list, status = sepenum(weights, graph, k, clk, nt)

    if status != BT_TIMEOUT
        perm, _, mink, status = sepdp(weights, graph, list, k, clk, nt)
    end

    t = time() - ts; esc.spent += t; count!(:tcpu, t)

    if status != BT_TIMEOUT
        if !isnothing(req)
            req.stop[] = true; req.started[] && count!(:cpuwon)
        end

        esc.t2 = esc.t1; esc.t1 = t; count!(:cpu); observe!(:cpu, esc.n, t)
        status == BT_ABOVE && return perm, bt_next(weights, graph, k, mink), BT_ABOVE
        return perm, mink, status
    end

    # (the CPU round stops at the deadline too, which sets the same flag)
    if !isnothing(req) && answered(req)
        count!(:gpuwon)
        return req.result
    end

    isnothing(req) || (req.stop[] = true)
    observe!(:cpu, esc.n, t; censored = true)
    return Int[], k, BT_TIMEOUT
end

# the type of a set of n vertices on the device
settype(n::Integer) = n <= 64 ? UInt64 : Bits{cld(n, 64) <= 2 ? 2 : cld(n, 64) <= 4 ? 4 : 8}

# A round of sepenum and sepdp on the GPU, or sepdp alone on the
# separators sep that the CPU has listed.
function septwround(weights::AbstractVector{W}, graph::AbstractGraph, k::W, dl::Real, sep) where {W}
    dl = Float64(dl); seps = isnothing(sep) ? nothing : [sep]
    x = only(septwbatch(settype(nv(graph)), [weights], [graph], [k], _ -> time() > dl, [bt_clk(dl)]; seps))
    return answer(weights, graph, k, x)
end

# The answer of a round from the result x of septwbatch.
function answer(weights::AbstractVector{W}, graph::AbstractGraph, k::W, x) where {W}
    isnothing(x) && return Int[], k, BT_TIMEOUT
    x === :failed && return :failed
    perm, mink, found = x; mink = towidth(W, mink)
    found === :found && return perm, mink, BT_FOUND
    return perm, bt_next(weights, graph, k, mink), BT_ABOVE
end

# Rounds of sepenum and sepdp on the GPU for a batch of graphs whose sets
# have the type T: one launch of each kernel of sepenum for all of them,
# and one of each kernel of sepdp (one for each size of the blocks) for
# those that are left. The result of the graph i, at the width ks[i], is
# (perm, mink, :found | :above), where mink is the width of perm or the
# bound of sepdp; nothing if the round was stopped, once cut(i) holds or
# its clock clks[i] runs out; or :failed if it overflowed the device code.
# The results are those of sepenum and sepdp on each graph alone, and
# done(i, result) is called as soon as the result of the graph i is
# known. With seps, sepdp alone, on these separators.
function septwbatch(::Type{T}, weights, graphs, ks, cut, clks; seps = nothing, done = (i, x) -> nothing) where {T}
    G = length(graphs); res = Vector{Any}(nothing, G)
    wgt = [Float64[w[v] for v in 1:nv(g)] for (w, g) in zip(weights, graphs)]
    put!(i, x) = (res[i] = x; done(i, x); nothing)

    if isnothing(seps)
        ts = time()
        batch = SepEnumGPU.prepare(T, graphs, wgt, Float64[k for k in ks])
        lists, ovf, tk, info = SepEnumGPU.gpu_sepenum(batch; split = 3, cut)
        seps = Vector{Any}(nothing, G)

        for i in 1:G
            if info.cut[i]
                put!(i, nothing)
            elseif ovf[i]
                put!(i, :failed)
            else
                seps[i] = SepEnumGPU.sepgraph(lists[i], nv(graphs[i]))
            end
        end

        count!(:tenum, time() - ts); count!(:tenumk, tk)
    end

    # (the graphs answered by sepenum, or stopped, leave the batch)
    live = Int[]

    for i in 1:G
        if !isnothing(seps[i])
            cut(i) ? put!(i, nothing) : push!(live, i)
        end
    end

    isempty(live) && return res
    ts = time()

    _, info = SepDPGPU.gpu_sepdp(T, wgt[live], graphs[live], seps[live], Float64[ks[i] for i in live]; cut = j -> cut(live[j]), clks = clks[live],
        done = (j, x) -> put!(live[j], isnothing(x) || x === :failed ? x : (x[1], x[3], x[4])))

    count!(:tdp, time() - ts); count!(:tdpprep, info.tprep); count!(:tdpup, info.tupload); count!(:tdpk, info.tgpu); count!(:tdpfin, info.tfinish)
    return res
end

towidth(::Type{W}, x::Float64) where {W} = isfinite(x) ? convert(W, x) : typemax(W)
towidth(::Type{W}, x::Float64) where {W <: AbstractFloat} = convert(W, x)

# A round decided by PIDGPU, on the graph relabeled in reverse BFS order
# (the order of the search; see PIDGPU.vorder). The search goes down from
# an upper bound ub, the width of a greedy ordering (MF) at first: it asks
# for a decomposition lighter than ub, which, if there is one, lowers ub to
# its width. A positive instance is cheap for PIDGPU and a negative one is
# not, and only the last question is negative: it proves that ub is the
# width. The round at k is then answered from ub: found if ub <= k, above
# (with the next width ub) if the width ub is proved and above k. The
# bound and the ordering are kept for the next rounds of the search. The
# round stops at the deadline dl, or when stop[] is set.
function pidround!(esc::Esc, weights::AbstractVector{W}, graph::AbstractGraph, k::W, dl::Real, stop::Threads.Atomic{Bool} = Threads.Atomic{Bool}(false)) where {W}
    n = nv(graph)

    if isnothing(esc.pid)
        # (the orderings and widths on the CPU are computed on a graph and
        # weights of fixed types, which the warm-up compiles)
        adj = [Int[x for x in neighbors(graph, v) if x != v] for v in 1:n]
        wv = W[weights[v] for v in 1:n]; sg = SimpleGraph(n)

        for v in 1:n, x in adj[v]
            x > v && add_edge!(sg, v, x)
        end

        perm = PIDGPU.vorder(adj, :rbfs); pg = PIDGPU.relabel(adj, Float64.(wv), perm)
        order, _ = permutation(wv, sg, MF())
        esc.pid = (; perm, pg, gg = PIDGPU.GPUGraph(pg), order = Ref(collect(order)),
            ub = Ref(convert(W, treewidth(wv, sg, order))), proved = Ref(false), wv, sg)
    end

    (; perm, pg, gg, order, ub, proved, wv, sg) = esc.pid

    while true
        ub[] <= k && return order[], ub[], BT_FOUND
        proved[] && return Int[], ub[], BT_ABOVE
        (stop[] || time() > dl) && return Int[], k, BT_TIMEOUT

        r = PIDGPU.gpu_decide(gg, Float64(CliqueTrees.rtw_below(ub[])); tol = 0.0, deadline = Float64(dl), stop)
        get(r.stats, :timeout, false) && return Int[], k, BT_TIMEOUT

        if r.yes
            new = perm[PIDGPU.gpu_order(pg, r)]; width = convert(W, treewidth(wv, sg, new))

            if width < ub[]
                order[] = new; ub[] = width
            else
                proved[] = true
            end
        else
            proved[] = true
        end
    end
end

# The cost of a GPU round on a small graph (a grid of 30 vertices, unit
# weights, at the width 5): compiled first, then the least of a few runs.
function cost!(engine::Symbol)
    c = lock(MODELLOCK) do
        get(COST, engine, nothing)
    end

    isnothing(c) || return c

    lock(LOCK) do
        c = NaN

        try
            # (a new state for each run: the PID rounds keep their bounds)
            g = BipartiteGraph(Graphs.grid([5, 6])); w = ones(Float64, nv(g))
            once() = engine === :pid ? pidround!(Esc(GPUSeptw(; engine), nv(g)), w, g, 5.0, Inf) : septwround(w, g, 5.0, Inf, nothing)
            once(); c = minimum(@elapsed(once()) for _ in 1:5)
        catch err
            err isa InterruptException && rethrow()
            failed!(err)
        end

        lock(MODELLOCK) do
            COST[engine] = c
        end

        return c
    end
end

# Compile the device code for every set type up to nmax vertices, and
# measure the cost of a round.
function CliqueTrees.septwwarmup!(backend::GPUSeptw; nmax::Integer = NMAX)
    usable() || return backend

    if backend.engine === :auto
        for engine in (:septw, :pid)
            CliqueTrees.septwwarmup!(GPUSeptw(; engine); nmax)
        end

        return backend
    end

    cost!(backend.engine)

    lock(LOCK) do
        nwrd = backend.engine === :pid ? (1:cld(nmax, 64)) : unique(nwords(settype(64w)) for w in 1:cld(min(nmax, NMAX), 64))

        for w in nwrd
            # a cycle with a chord, with two weights; for PID, a grid of five
            # rows (every step of its search runs: joins, both answers, and
            # the orderings)
            if backend.engine === :pid
                g = Graphs.grid([5, cld(64(w - 1) + 20, 5)])
            else
                g = cycle_graph(64(w - 1) + 10); add_edge!(g, 1, nv(g) ÷ 2)
            end

            n = nv(g); g = BipartiteGraph(g); wgt = Float64[isodd(v) ? 1.0 : 2.0 for v in 1:n]

            try
                if backend.engine === :pid
                    esc = Esc(backend, n); pidround!(esc, wgt, g, 0.0, Inf)
                    pidround!(Esc(backend, n), ones(Int, n), g, 0, Inf)

                    # a positive answer, and its ordering
                    p = esc.pid; r = PIDGPU.gpu_decide(p.gg, Float64(p.ub[]); tol = 0.0, deadline = Inf, stop = Threads.Atomic{Bool}(false))
                    r.yes && PIDGPU.gpu_order(p.pg, r)

                    # a denser graph, as the local graphs of HBT (the paths of
                    # larger rounds: the buffers grow), for a second
                    m = 64w - 4; h = erdos_renyi(m, 0.3; rng = Random.Xoshiro(w))

                    for v in 1:(m - 1)
                        add_edge!(h, v, v + 1)
                    end

                    pidround!(Esc(backend, m), Float64[isodd(v) ? 1.0 : 2.0 for v in 1:m], BipartiteGraph(h), 0.0, time() + 1.0)
                else
                    septwround(wgt, g, 3.0, Inf, nothing)
                end
            catch err
                err isa InterruptException && rethrow()
                failed!(err)
            end
        end
    end

    return backend
end

nwords(::Type{UInt64}) = 1
nwords(::Type{Bits{W}}) where {W} = W

end
