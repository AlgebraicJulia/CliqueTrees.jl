"""cotengra's C of the tree induced by each ordering (frontier ftn.tree_from_order: open indices last,
bucket elimination, each bucket binarized by 'auto-hq'), for the corr_orders.jl pools.
usage: python -I corr_ctg.py <dir> <name> [<name> ...]   -> <dir>/<name>.ctg.tsv (k, ctg_log2C, ctg_W, seconds)"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/itaykadosh/Projects/GATAS/tn-work/frontier/scripts")
import tnio  # noqa: E402
import ftn  # noqa: E402


def main(d, names):
    for name in names:
        inputs, output, size, _ = tnio.load(name)
        out = os.path.join(d, name + ".ctg.tsv")
        with open(os.path.join(d, name + ".orders")) as f, open(out, "w") as fo:
            fo.write("k\tctg_log2C\tctg_W\tsec\n")
            for k, line in enumerate(f, 1):
                order = [int(x) for x in line.split()]
                t0 = time.time()
                tree = ftn.tree_from_order(inputs, output, size, order)
                fo.write(f"{k}\t{tree.contraction_cost(log=2):.6f}\t{tree.contraction_width(log=2):.3f}\t{time.time() - t0:.3f}\n")
                fo.flush()
        print(name, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
