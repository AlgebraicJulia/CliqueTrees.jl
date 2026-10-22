"""Stronger cotengra baselines on the B networks.  python -I costStrong.py <shard> <nshards> <outdir> [ncpu]
For every network of queueB (that has resB/trees/<net>.json) and every config below: HyperOptimizer(parallel=ncpu,
slicing_reconf_opts={'target_size': 2**30} ALWAYS on), then (if the result is still wider than 2^30) slice_and_reconfigure,
(if unsliced) ftn.refine like costB.  Trees are re-costed with cotengra (contraction_cost, contraction_width, multiplicity).
Writes <outdir>/<net>.json (same layout as resB/trees/<net>.json, trees = ours60 + ctg_hyper60 copied + the strong ones) and
appends rows to <outdir>/strong.tsv: net tag log10C W nslices search_s."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cotengra as ctg
import ftn, tnc

CONFIGS = [  # tag, minimize, max_time, kind
    ("sf60", "flops", 60, "hyper"), ("sc60", "combo", 60, "hyper"), ("sc256_60", "combo-256", 60, "hyper"),
    ("sr60", "combo", 60, "reusable"),
    ("sc600", "combo", 600, "hyper"), ("sc256_600", "combo-256", 600, "hyper"), ("sf600", "flops", 600, "hyper"),
]
TARGET = 30

def main():
    shard, nsh, outdir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    ncpu = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
    only = set(os.environ.get("ONLY", "").split(",")) - {""}
    cfgs = [c for c in CONFIGS if c[0] in set(os.environ.get("CFGS", ",".join(c[0] for c in CONFIGS)).split(","))]
    src = "/blue/fairbanksj/itaykadosh/gpu/exactness/resB/trees"
    nets = [l.strip() for l in open(os.path.join(HERE, "..", "queueB.txt")) if l.strip()]
    nets = [n for n in nets if os.path.exists(f"{src}/{n}.json") and (not only or n in only)][shard::nsh]
    os.makedirs(outdir, exist_ok=True)
    for net in nets:
        jf = f"{outdir}/{net}.json"
        if os.path.exists(jf):
            js = json.load(open(jf))
        else:
            js = json.load(open(f"{src}/{net}.json"))
            js["trees"] = [t for t in js["trees"] if t["tag"] in ("ours60", "ctg_hyper60")]
        have = {t["tag"] for t in js["trees"]}
        inputs, output, size = tnc.load(net)
        inv = {ctg.get_symbol(i): i for i in range(1, len(size) + 2)}
        for tag, mini, T, kind in cfgs:
            if tag in have:
                continue
            t0 = time.time()
            kw = dict(slicing_reconf_opts={"target_size": 2 ** TARGET})
            cls = ctg.ReusableHyperOptimizer if kind == "reusable" else ctg.HyperOptimizer
            try:
                opt = cls(max_time=T, parallel=ncpu, minimize=mini, progbar=False, max_repeats=10 ** 6, **kw)
                h = opt.search(inputs, output, size)
                if h.sliced_inds == () or len(h.sliced_inds) == 0:
                    if h.contraction_width() > TARGET:
                        h = h.slice_and_reconfigure(target_size=2 ** TARGET, minimize="flops")
                    else:
                        h = ftn.refine(h)
                elif h.contraction_width() > TARGET:
                    h = h.slice_and_reconfigure(target_size=2 ** TARGET, minimize="flops")
            except Exception as e:
                print(net, tag, "FAILED", repr(e)[:200], flush=True)
                continue
            s = time.time() - t0
            d = dict(tag=tag, path=[list(p) for p in h.get_path()], sliced=[inv[x] for x in h.sliced_inds],
                     log10C=h.contraction_cost(log=10), W=h.contraction_width(), search_s=s)
            js["trees"].append(d)
            json.dump(js, open(jf + ".tmp", "w")); os.replace(jf + ".tmp", jf)
            with open(f"{outdir}/strong.tsv", "a") as f:
                f.write(f"{net}\t{tag}\t{d['log10C']:.6f}\t{d['W']:.3f}\t{h.multiplicity}\t{s:.2f}\n")
            print(net, tag, f"{d['log10C']:.3f} W={d['W']:.1f} nsl={h.multiplicity} {s:.0f}s", flush=True)

if __name__ == "__main__":
    main()
