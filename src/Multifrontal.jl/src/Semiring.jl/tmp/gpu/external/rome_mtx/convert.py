#!/usr/bin/env python3
"""Convert a 'coordinate real general' MTX listing both arc directions (symmetric weights)
into a 'coordinate real symmetric' MTX with only the lower triangle (i > j).
ROME's MtxReader adds both i->j and j->i for every entry, so the general file would
create duplicate edges (ROME aborts on multigraphs). Weights are copied verbatim."""
import sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    lines = [l for l in f if not l.startswith('%')]
n, m, nnz = map(int, lines[0].split())
w = {}
for l in lines[1:1 + nnz]:
    i, j, v = l.split()
    w[(int(i), int(j))] = v
for (i, j), v in w.items():
    assert w[(j, i)] == v, "asymmetric weight"
low = sorted(((i, j, v) for (i, j), v in w.items() if i > j), key=lambda t: (t[1], t[0]))
with open(dst, 'w') as f:
    f.write("%%MatrixMarket matrix coordinate real symmetric\n")
    f.write(f"% converted from {src}: lower triangle of a symmetric arc list, weights unchanged\n")
    f.write(f"{n} {m} {len(low)}\n")
    for i, j, v in low:
        f.write(f"{i} {j} {v}\n")
print(dst, n, len(low))
