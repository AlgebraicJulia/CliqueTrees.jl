"""Fixed-slicing experiment: remove a given set S of sliced indices from a frontier network
(one slice: every tensor loses the indices of S) and write it as a new instance <name>_cut<|S|>.
Total cost of any tree of the cut network = 2^|S| x its cost (all slices have the same structure).
The sliced set comes from a cotengra hyper+slicing run (<res>/<tag>.path.json, 'sliced_inds').
usage: python make_cut.py <name> <path.json> <out frontier dir>   (network read from $FRONTIER_DIR)"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftn  # noqa: E402
from gen_frontier import write_instance  # noqa: E402


def main(name, pathjson, fdir):
    inputs, output, size = ftn.load(name)
    S = set(json.load(open(pathjson))["sliced_inds"])
    assert S <= set(size), "sliced index not in network"
    cin = [tuple(x for x in t if x not in S) for t in inputs]
    cout = tuple(x for x in output if x not in S)
    csize = {k: v for k, v in size.items() if k not in S}
    cname = f"{name}_cut{len(S)}"
    params = f"cut_of={name};sliced={len(S)};from={os.path.basename(pathjson)}"
    st = write_instance(fdir, cname, "cut", cin, cout, csize, params)
    print(cname, st)


if __name__ == "__main__":
    main(*sys.argv[1:4])
