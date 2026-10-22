"""Task list for the frontier benchmark job (one task per line, longest first).
line: <est seconds> <J|P> <name> <method> <budget> <seed> <targets>
J = our ordering (orders.jl, then eval_orders.py on its output), P = cotengra baseline (baselines.py)."""
import sys

AMP = [f"syc53_m{m}_amp" for m in (12, 14, 16, 18, 20)]
MARG = [f"syc53_m{m}_marg6" for m in (12, 14, 16, 18, 20)] + ["syc53_m20_open6huang"]
BIG = ["syc53_m20_marg21"]
RAW = [f"syc53_m{m}_raw" for m in (12, 14, 16, 18, 20)]
BEYOND = ["syc70_m24_amp", "syc67_m32_amp"]
LONG = {"syc53_m20_amp", "syc53_m18_amp", "syc53_m20_marg6", "syc53_m20_marg21"} | set(BEYOND)

targets = {}
for n in AMP:
    targets[n] = ([27, 29, 30, 32], [27, 30])
for n in MARG:
    targets[n] = ([29, 32], [29])
for n in BIG:
    targets[n] = ([30, 32], [30])
for n in RAW:
    targets[n] = ([27, 30], [])
for n in BEYOND:
    targets[n] = ([30, 32], [30])

tasks = []
budgets = [1, 10, 100, 1000]
for n, (tg, hs) in targets.items():
    bs = budgets + ([10000] if n in LONG else [])
    tgs = ",".join(map(str, tg))
    tasks.append((60, "J", n, "bw", 0, 1, tgs))
    tasks.append((300, "J", n, "nd", 0, 1, tgs))
    tasks.append((30, "P", n, "greedy", 0, 1, tgs))
    for T in bs:
        post = 120 if T < 1000 else 300
        tasks.append((T + 90 + post, "J", n, "hbt", T, 1, tgs))
        tasks.append((T + 300 + post, "J", n, "hbtnd", T, 1, tgs))
        tasks.append((T + post, "P", n, "rgreedy", T, 1, tgs))
        tasks.append((T + post, "P", n, "hyper", T, 1, tgs))
        if T >= 100:
            for h in hs:
                tasks.append((T, "P", n, f"hslice{h}", T, 1, tgs))
tasks.sort(key=lambda t: -t[0])
# fixed-slicing experiments, queued last: they wait for the T=1000 hyper+slicing tree of the
# same job, then order the cut network with our methods (budgets 10,100,1000 s)
for n, (tg, hs) in targets.items():
    for h in hs:
        tasks.append((2500, "P", n, f"cuthslice{h}", 1000, 1, str(h)))
with open(sys.argv[1], "w") as f:
    for t in tasks:
        f.write(" ".join(map(str, t)) + "\n")
tot = sum(t[0] for t in tasks)
print(len(tasks), "tasks,", round(tot / 3600, 1), "core-hours est; longest", tasks[0][0], "s")
