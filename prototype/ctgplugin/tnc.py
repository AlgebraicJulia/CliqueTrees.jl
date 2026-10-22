"""Reader for the TNCORP1 .tnc corpus files (same as tncorpus/scripts/ctg_bench.py) -> cotengra symbols."""
import csv
import os
import numpy as np
import cotengra as ctg

CORPUS = os.environ.get("TNCORPUS_DIR", "/home/itaykadosh/Projects/GATAS/tn-work/tncorpus")


def read_tnc(path):
    with open(path, "rb") as f:
        assert f.read(8) == b"TNCORP1\0"
        n, nt, nnz, nopen, hnnz = np.fromfile(f, "<i8", 5)
        dims = np.fromfile(f, "<i8", n)
        np.fromfile(f, "<i8", n + 1); np.fromfile(f, "<i4", nnz)
        openv = np.fromfile(f, "<i4", nopen)
        hptr = np.fromfile(f, "<i8", nt + 1); hidx = np.fromfile(f, "<i4", hnnz)
    inputs = [tuple(int(x) for x in hidx[hptr[t] - 1:hptr[t + 1] - 1]) for t in range(nt)]
    size = {i + 1: int(d) for i, d in enumerate(dims)}
    return inputs, tuple(int(x) for x in openv), size


def manifest(corpus=CORPUS):
    with open(os.path.join(corpus, "manifest.tsv")) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def einsum_names(corpus=CORPUS, paper=True):
    rows = [r for r in manifest(corpus) if r["source"] == "einsum" and (not paper or "paper=1" in r["params"])]
    return [r["name"] for r in sorted(rows, key=lambda r: (int(r["n"]), r["name"]))]


def load(name, corpus=CORPUS):
    """(inputs, output, size_dict) with one cotengra symbol per index (same labelling as ftn.load);
    scalar tensors (no index of dim > 1) are dropped, as in the draft protocol."""
    r = next(r for r in manifest(corpus) if r["name"] == name)
    inputs, output, size = read_tnc(os.path.join(corpus, r["file"]))
    sym = lambda v: ctg.get_symbol(int(v))
    return ([tuple(sym(x) for x in t) for t in inputs if t], tuple(sym(x) for x in output),
            {sym(k): v for k, v in size.items()})
