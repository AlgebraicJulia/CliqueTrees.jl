"""Diagnose the relerr==1.0 trees of experiment B: magnitude of the c128 / c64 result (numpy, CPU).
python -I diagB.py <pkldir> net1,net2,... [--maxc 11.0]"""
import pickle, sys, math, hashlib, time
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from contractB import scales
pk, nets = sys.argv[1], sys.argv[2].split(",")
for net in nets:
    d = pickle.load(open(f"{pk}/{net}.pkl", "rb"))
    inputs, output, size = d["inputs"], d["output"], d["size"]
    seed = int(hashlib.md5(net.encode()).hexdigest()[:8], 16)
    sc = scales(inputs, output, size)
    rs = np.random.default_rng(seed)
    arr = [((rs.standard_normal([size[i] for i in t]) + 1j * rs.standard_normal([size[i] for i in t])) / math.sqrt(2) * sc[j])
           for j, t in enumerate(inputs)]
    trees = sorted(d["trees"], key=lambda x: x[1].contraction_cost())
    tag, tree = trees[0][0], trees[0][1]
    n = len(inputs)
    print(net, "n_tensors", n, "out_inds", len(output), "tree", tag, "log10C", round(tree.contraction_cost(log=10), 2),
          "nslices", tree.multiplicity, flush=True)
    t0 = time.time()
    r = np.asarray(tree.contract(arr, backend="numpy")).ravel()
    nr = np.linalg.norm(r)
    print("  c128 |result| =", nr, "log10 =", math.log10(nr) if nr > 0 else None, "(%.0fs)" % (time.time() - t0), flush=True)
    a64 = [x.astype(np.complex64) for x in arr]
    r64 = np.asarray(tree.contract(a64, backend="numpy")).ravel().astype(np.complex128)
    print("  c64  |result| =", np.linalg.norm(r64), " relerr c64 vs c128 =", np.linalg.norm(r64 - r) / nr, flush=True)
    s = nr ** (-1.0 / n)
    a64 = [(x * s).astype(np.complex64) for x in arr]
    r64 = np.asarray(tree.contract(a64, backend="numpy")).ravel().astype(np.complex128)
    print("  after global rescale s=%.4f: c64 |result| = %.3e, relerr vs c128*s^n = %.3e" % (s, np.linalg.norm(r64), np.linalg.norm(r64 - r * s ** n) / (nr * s ** n)), flush=True)
