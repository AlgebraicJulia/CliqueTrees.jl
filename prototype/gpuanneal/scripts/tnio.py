"""I/O shared by the gpuanneal scripts.

Networks come from .tnc files (tncorpus/scripts/tncwrite.py format). Index v (1-based line-graph vertex)
gets cotengra label sym(v), exactly as in frontier/scripts/ftn.py. Tensors with no index (all dims 1) are
dropped (scalars), as ftn.load does; tensor ids below are positions in the kept list (0-based).

Tree file (.tree), text:
  line 1: "TREE <ntensors> <nsliced>"
  line 2: ssa path, flattened pairs "a b a b ..." (cotengra ssa convention, 0-based)
  line 3: sliced index ids (1-based line-graph vertices), may be empty
Optional further lines "# key value" carry metadata (e.g. the GPU's own cost numbers).
"""
import math
import os

import numpy as np
import cotengra as ctg

SCR = "/path/to/tn-work"
CORPUS = os.path.join(SCR, "tncorpus")
FRONTIER = os.path.join(SCR, "frontier")
MAGIC = b"TNCORP1\0"


def sym(v):
    return ctg.get_symbol(int(v))


def find_tnc(name):
    for p in (os.path.join(FRONTIER, "data", name + ".tnc"),):
        if os.path.exists(p):
            return p
    with open(os.path.join(CORPUS, "manifest.tsv")) as f:
        for line in f:
            r = line.rstrip("\n").split("\t")
            if r[0] == name:
                return os.path.join(CORPUS, r[11])
    raise FileNotFoundError(name)


def read_tnc(path):
    with open(path, "rb") as f:
        assert f.read(8) == MAGIC
        n, nt, nnz, nopen, hnnz = np.fromfile(f, "<i8", 5)
        dims = np.fromfile(f, "<i8", n)
        np.fromfile(f, "<i8", n + 1)
        np.fromfile(f, "<i4", nnz)
        openv = np.fromfile(f, "<i4", nopen)
        hptr = np.fromfile(f, "<i8", nt + 1)
        hidx = np.fromfile(f, "<i4", hnnz)
    tens = [hidx[hptr[t] - 1:hptr[t + 1] - 1].tolist() for t in range(nt)]
    return dims.tolist(), tens, openv.tolist()


def load(name):
    """-> (inputs, output, size) in cotengra labels, plus raw (dims, tensors(1-based ids), open)."""
    dims, tens, openv = read_tnc(find_tnc(name))
    tens = [t for t in tens if t]
    inputs = [tuple(sym(x) for x in t) for t in tens]
    output = tuple(sym(x) for x in openv)
    size = {sym(v + 1): int(d) for v, d in enumerate(dims)}
    return inputs, output, size, (dims, tens, openv)


def write_tree(path, nt, ssa, sliced=(), meta=None):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"TREE {nt} {len(sliced)}\n")
        f.write(" ".join(f"{a} {b}" for a, b in ssa) + "\n")
        f.write(" ".join(str(int(x)) for x in sliced) + "\n")
        for k, v in (meta or {}).items():
            f.write(f"# {k} {v}\n")
    os.replace(tmp, path)


def read_tree(path):
    with open(path) as f:
        hdr = f.readline().split()
        nt = int(hdr[1])
        flat = [int(x) for x in f.readline().split()]
        sliced = [int(x) for x in f.readline().split()]
        meta = {}
        for line in f:
            if line.startswith("# "):
                k, _, v = line[2:].rstrip("\n").partition(" ")
                meta[k] = v
    ssa = list(zip(flat[0::2], flat[1::2]))
    return nt, ssa, sliced, meta


def ctg_tree(inputs, output, size, ssa, sliced=()):
    """Independent re-costing: build a cotengra ContractionTree from the ssa path and slice it."""
    tree = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=ssa)
    for v in sliced:
        tree.remove_ind_(sym(v))
    return tree


def tree_stats(tree):
    """log2 total cost C (multiply-adds summed over all contractions and all slices), log2 of the
    largest intermediate per slice (W), log2 of the largest single contraction, #sliced."""
    return dict(
        log2C=tree.contraction_cost(log=2),
        W=tree.contraction_width(log=2),
        maxop=max(math.log2(tree.get_flops(nd)) for nd, _, _ in tree.traverse()),
        nsliced=len(tree.sliced_inds),
    )
