"""Per-stage log10 C of method (b) from run_gpu.jl logs: seeds (best raw induced tree), after mass
search (best raw), after reconfiguration of the top 8, after ILS, final; plus throughput.
usage: python -I stages.py <log> [<log> ...]"""
import re
import statistics
import sys

pat = re.compile(r"mass (\d+) cand in (\d+) gens, ([0-9.e+]+) cand/s; raw log10C seeds ([0-9.]+) -> ([0-9.]+)  reconf K10 on \d+: (\d+) rounds (\d+) chunks -> ([0-9.]+)  ILS (\d+) steps (\d+) impr -> ([0-9.]+)  K12: \d+ rounds -> ([0-9.]+)  total ([0-9.]+) s")
for f in sys.argv[1:]:
    rows = []
    lines = open(f).read().splitlines()
    for i, l in enumerate(lines):
        m = pat.search(l)
        if m and i + 1 < len(lines) and "B=" in lines[i + 1]:
            g = m.groups()
            rows.append(dict(cand=int(g[0]), cps=float(g[2]), seed=float(g[3]), mass=float(g[4]), chunks=int(g[6]), reconf=float(g[7]),
                             ils_steps=int(g[8]), ils=float(g[10]), final=float(g[11]), t=float(g[12])))
    med = lambda k: statistics.median(r[k] for r in rows)
    print(f"{f}: {len(rows)} networks")
    print(f"  median log10C drop: mass search {med('seed') - 0 - med('mass') if False else statistics.median(r['seed'] - r['mass'] for r in rows):.3f}, "
          f"reconf {statistics.median(r['mass'] - r['reconf'] for r in rows):.3f}, ILS {statistics.median(r['reconf'] - r['ils'] for r in rows):.3f}, "
          f"K12 polish {statistics.median(r['ils'] - r['final'] for r in rows):.4f}; total seed->final {statistics.median(r['seed'] - r['final'] for r in rows):.3f}")
    print(f"  mean drops: mass {statistics.mean(r['seed'] - r['mass'] for r in rows):.3f} reconf {statistics.mean(r['mass'] - r['reconf'] for r in rows):.3f} "
          f"ILS {statistics.mean(r['reconf'] - r['ils'] for r in rows):.3f}")
    cps = sorted(r["cps"] for r in rows)
    print(f"  mass scoring: median {statistics.median(cps):.3g} cand/s (min {cps[0]:.3g}, max {cps[-1]:.3g}); candidates per network median {med('cand'):.3g}; "
          f"GPU stage median {med('t'):.2f} s; DP chunks per network median {med('chunks'):.3g}")
