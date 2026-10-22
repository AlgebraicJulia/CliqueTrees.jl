"""Independent cotengra re-costing of .tree files (tnio format).
usage: python -I verify.py <out tsv> <tree files or dirs...>
Each tree: cotengra ContractionTree.from_path(ssa) -> contraction_cost(log=2), contraction_width;
compared with the producer's stored '# log2C' and '# W' metadata.
Columns: file name ctg_log2C ctg_log10C ctg_W stored_log2C stored_W |dC| |dW| status"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tnio  # noqa: E402


def main():
    out = sys.argv[1]
    files = []
    for d in sys.argv[2:]:
        files += sorted(glob.glob(os.path.join(d, "*.tree"))) if os.path.isdir(d) else [d]
    cache = {}
    nbad = 0
    with open(out, "a") as fo:
        for f in files:
            b = os.path.basename(f)
            name = b.split("__")[0]
            if name not in cache:
                cache.clear()
                cache[name] = tnio.load(name)
            inputs, output, size, _ = cache[name]
            nt, ssa, sliced, meta = tnio.read_tree(f)
            assert nt == len(inputs), (f, nt, len(inputs))
            tree = tnio.ctg_tree(inputs, output, size, ssa, sliced)
            c2 = tree.contraction_cost(log=2)
            wd = tree.contraction_width(log=2)
            sc = float(meta.get("log2C", "nan")); sw = float(meta.get("W", "nan"))
            dc = abs(c2 - sc); dw = abs(wd - sw)
            ok = dc < 1e-3 and dw < 1e-3
            noref = sc != sc
            nbad += (not ok) and not noref
            st = "NOREF" if noref else ("OK" if ok else "MISMATCH")
            fo.write(f"{b}\t{name}\t{c2:.6f}\t{c2 * 0.30102999566:.4f}\t{wd:.3f}\t{sc:.6f}\t{sw:.3f}\t{dc:.2e}\t{dw:.2e}\t{st}\n")
            fo.flush()
    print("verified", len(files), "trees,", nbad, "mismatches")


if __name__ == "__main__":
    main()
