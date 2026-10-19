"""Independent CPU re-costing of annealer output trees with cotengra.
usage: python recost.py <name> <tree files...>
For each tree: builds cotengra ContractionTree.from_path(ssa_path), slices the listed indices with
remove_ind_, and compares cotengra's contraction_cost (log2) / contraction_width with the GPU energy
(gpu_E) and the Julia host re-costing (host_log2C, host_W) stored in the tree metadata.
Prints one TSV row per tree: file, ctg_log2C, gpu_E, host_log2C, |ctg-gpu|, ctg_W, host_W, nsliced, ok
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tnio  # noqa: E402


def main():
    name = sys.argv[1]
    inputs, output, size, _ = tnio.load(name)
    bad = 0
    for p in sys.argv[2:]:
        nt, ssa, sliced, meta = tnio.read_tree(p)
        assert nt == len(inputs), (nt, len(inputs))
        tree = tnio.ctg_tree(inputs, output, size, ssa, sliced)
        st = tnio.tree_stats(tree)
        ge = float(meta.get("gpu_E", "nan")); hc = float(meta.get("host_log2C", "nan"))
        hw = float(meta.get("host_W", "nan"))
        d = abs(st["log2C"] - ge) if ge == ge else 0.0
        ok = d < 1e-3 and abs(st["log2C"] - hc) < 1e-6 and (hw != hw or abs(st["W"] - hw) < 1e-6)
        bad += not ok
        print(f"{os.path.basename(p)}\t{st['log2C']:.6f}\t{ge:.6f}\t{hc:.6f}\t{d:.2e}\t{st['W']:.3f}\t{hw:.3f}\t{st['nsliced']}\t{'OK' if ok else 'MISMATCH'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
