using Serialization, Printf
for f in filter(x -> startswith(x, "cap10") && endswith(x, ".jls"), readdir())
    for (nm, v) in deserialize(f)
        recs = sort(v.recs; by = r -> -r.t)
        tt = sum(r.t for r in recs; init = 0.0)
        g(r, k) = get(r.ph, k, 0.0)
        S(k, rs) = sum(g(r, k) for r in rs; init = 0.0)
        top = recs[1:min(end, cld(length(recs), 10))]
        @printf("%-36s all: septw %.0f%% (sepenum %.0f%% sepdp %.0f%%) kernel %.0f%% minimal %.0f%% lb %.0f%% other %.0f%% | top10%%: kn med %d, seps/round med %.0f, rounds med %.0f\n", nm,
            100S(:septw, recs)/tt, 100S(:sepenum, recs)/tt, 100S(:sepdp, recs)/tt, 100S(:kernel, recs)/tt, 100S(:minimal, recs)/tt, 100S(:lb, recs)/tt,
            100(tt - S(:septw, recs) - S(:kernel, recs) - S(:minimal, recs) - S(:lb, recs))/tt,
            sort([r.kn for r in top])[cld(end,2)], sort([g(r,:nsep)/max(1,g(r,:rounds)) for r in top])[cld(end,2)], sort([g(r,:rounds) for r in top])[cld(end,2)])
    end
end
