"""cotengra re-costing of the selected trees (consistency with the exactness study).

    python -I recost.py <net> <pooldir> <gpudir> <out.tsv>

Items: every TD named in a SEL line of <net>.pool (stages bucket = ftn.tree_from_order greedy, br = its
branchtree+branchrefine ssa, br+ctg = ftn.refine(br), as costA.py), the GPU mass-search winner (gms:
bucket, bucket+ctg) and the GPU-polished trees (pol:<name>: raw, +ctg).
Columns: net item stage log10C W maxop seconds
"""
import os
import sys
import time

sys.path.insert(0, os.environ.get("EXACT_SRC", "/blue/fairbanksj/itaykadosh/gpu/exactness/src"))
import cotengra as ctg  # noqa: E402
import ftn  # noqa: E402
import tnc  # noqa: E402


def pairs(s):
    v = [int(x) for x in s.split()]
    return [(v[k], v[k + 1]) for k in range(0, len(v), 2)]


def main():
    net, pooldir, gpudir, outp = sys.argv[1:5]
    lines = open(os.path.join(pooldir, net + ".pool")).read().split("\n")
    orders, ssa, want = {}, {}, set()
    i = 0
    while i < len(lines):
        f = lines[i].split("\t")
        if f[0] == "TD":
            orders[f[1]] = [int(x) for x in lines[i + 1].split()]; i += 1
        elif f[0] == "SSA":
            ssa[f[1]] = pairs(lines[i + 1]); i += 1
        elif f[0] == "SEL":
            want.add(f[5]); want.add(f[7])
        i += 1
    gms, pol = None, []
    gp = os.path.join(gpudir, net + ".gpu")
    if os.path.exists(gp):
        gl = open(gp).read().split("\n")
        for k, l in enumerate(gl):
            f = l.split("\t")
            if f[0] == "GMS":
                gms = [int(x) for x in gl[k + 1].split()]
            elif f[0] == "POL":
                pol.append((f[1], pairs(gl[k + 1])))
    inputs, output, size = tnc.load(net)
    out = open(outp, "w")

    def row(item, stage, tree, secs):
        st = ftn.stats(tree)
        out.write(f"{net}\t{item}\t{stage}\t{st['cost']:.6f}\t{st['width']:.4f}\t{st['maxop']:.4f}\t{secs:.3f}\n")
        out.flush()

    for tag in sorted(want):
        t0 = time.time()
        row("td:" + tag, "bucket", ftn.tree_from_order(inputs, output, size, orders[tag], optimize="greedy"), time.time() - t0)
        if tag in ssa:
            br = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=ssa[tag])
            row("td:" + tag, "br", br, 0.0)
            t0 = time.time(); r = ftn.refine(br)
            row("td:" + tag, "br+ctg", r, time.time() - t0)
    if gms is not None:
        b = ftn.tree_from_order(inputs, output, size, gms, optimize="greedy")
        row("gms", "bucket", b, 0.0)
        t0 = time.time(); row("gms", "bucket+ctg", ftn.refine(b), time.time() - t0)
    for name, p in pol:
        t = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=p)
        row("pol:" + name, "raw", t, 0.0)
        t0 = time.time(); row("pol:" + name, "+ctg", ftn.refine(t), time.time() - t0)
    out.write(f"#END\t{net}\n")
    out.close()


if __name__ == "__main__":
    main()
