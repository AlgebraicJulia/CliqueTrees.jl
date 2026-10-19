"""Independent cotengra re-costing of every tree in one or more directories.
usage: python verify.py <out tsv> <tree dir> [<tree dir> ...]
Trees are grouped by instance (file name prefix before '__'); each is re-costed with cotengra
(ContractionTree.from_path + remove_ind_ for sliced indices) and compared with the number the producer
stored (gpu_E / host_log2C for the GPU annealer, treesa_tc is NOT comparable when sliced, ctg_log2C for cotengra).
Columns: file instance ctg_log2C ctg_W nsliced stored_gpu_E stored_host abs_diff status"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tnio  # noqa: E402


def main():
    out = sys.argv[1]
    files = []
    for d in sys.argv[2:]:
        files += sorted(glob.glob(os.path.join(d, "*.tree")))
    done = set()
    if os.path.exists(out):
        done = {l.split("\t")[0] for l in open(out)}
    cache = {}
    nbad = 0
    with open(out, "a") as fo:
        for f in files:
            b = os.path.basename(f)
            if b in done:
                continue
            name = os.environ.get("TNNAME") or b.split("__")[0]
            if name not in cache:
                cache.clear()
                cache[name] = tnio.load(name)
            inputs, output, size, _ = cache[name]
            nt, ssa, sliced, meta = tnio.read_tree(f)
            tree = tnio.ctg_tree(inputs, output, size, ssa, sliced)
            st = tnio.tree_stats(tree)
            g = float(meta.get("gpu_E", "nan")); h = float(meta.get("host_log2C", meta.get("ctg_log2C", "nan")))
            ref = g if g == g else h
            d = abs(st["log2C"] - ref) if ref == ref else float("nan")
            ok = (d != d) or d < 1e-3
            nbad += not ok
            fo.write(f"{b}\t{name}\t{st['log2C']:.6f}\t{st['W']:.3f}\t{st['nsliced']}\t{g:.6f}\t{h:.6f}\t{d:.2e}\t{'OK' if ok else 'MISMATCH'}\n")
            fo.flush()
    print("verified", len(files) - len(done), "trees,", nbad, "mismatches")


if __name__ == "__main__":
    main()
