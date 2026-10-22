"""Turn our elimination orderings (.ord files from orders.jl) into cotengra contraction trees and
measure them raw, refined and sliced (ftn.post).
usage: python eval_orders.py <ord file> <outdir> <targets comma>
Writes <outdir>/<ord basename>.tsv (same columns as baselines.py; column 7 = our bag width)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftn  # noqa: E402


def main(path, outdir, targets):
    with open(path) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        order = [int(x) for x in f.readline().split()]
    name, method, budget, seed, secs, bag = hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]
    inputs, output, size = ftn.load(name)
    t0 = time.time()
    tree = ftn.tree_from_order(inputs, output, size, order)
    tb = time.time() - t0
    rows, _ = ftn.post(tree, targets)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, os.path.basename(path)[:-4] + ".tsv")
    with open(out, "w") as f:
        for stage, st, dt in rows:
            f.write("\t".join(map(str, (name, method, float(budget), seed, round(float(secs), 3), -1, bag, stage,
                                         round(st["cost"], 4), round(st["width"], 3), round(st["maxop"], 3),
                                         st["nsliced"], round(dt + (tb if stage == "raw" else 0), 3)))) + "\n")
    print(os.path.basename(out), "bag", bag, {k: round(v, 2) for k, v in rows[0][1].items()},
          "refined", round(rows[1][1]["cost"], 2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")])
