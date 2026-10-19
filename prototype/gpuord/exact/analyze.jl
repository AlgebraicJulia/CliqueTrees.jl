using Serialization, Printf
for f in filter(x -> startswith(x, "cap10") && endswith(x, ".jls"), readdir())
    d = deserialize(f)
    for (nm, v) in d
        recs = v.recs; isempty(recs) && continue
        tt = sum(r.t for r in recs)
        tto = sum(r.t for r in recs if r.result == :timeout; init = 0.0)
        ts = sort([r.t for r in recs]; rev = true)
        top10 = sum(ts[1:min(end, cld(length(ts), 10))])
        @printf("%-36s calls %5d  total %.1fs  timeouts %d (%.0f%% of time)  top10%% calls = %.0f%% of time  median %.2f ms  rounds/call %.1f\n",
            nm, length(recs), tt, count(r -> r.result == :timeout, recs), 100tto / tt, 100top10 / tt, 1000ts[cld(end, 2)], sum(get(r.ph, :rounds, 0.0) for r in recs) / length(recs))
        for (lo, hi) in ((0, 16), (17, 24), (25, 32), (33, 48), (49, 64), (65, 128))
            rs = [r for r in recs if lo <= r.kn <= hi]
            isempty(rs) && continue
            @printf("     kernel nv %3d-%3d: %5d calls  %5.1f%% of time  mean %.2f ms  timeouts %d\n", lo, hi, length(rs), 100sum(r.t for r in rs) / tt, 1000sum(r.t for r in rs) / length(rs), count(r -> r.result == :timeout, rs))
        end
    end
end
