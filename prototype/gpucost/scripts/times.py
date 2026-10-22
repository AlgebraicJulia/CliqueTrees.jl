"""Median realized wall seconds per method (tree metadata 't', gpu.tsv total incl. seed orderings)."""
import glob, os, statistics, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tnio
G = "/home/itaykadosh/Projects/GATAS/tn-work/gpucost/results/"
T = collections.defaultdict(list)
for f in glob.glob(G + "base/*.tree") + glob.glob(G + "treesa/*.tree"):
    b = os.path.basename(f); m = b.split("__")[1][:-5]
    T[m].append(float(tnio.read_tree(f)[3]["t"]))
for B in (1, 10):
    for l in open(G + f"gpu{B}/gpu.tsv"):
        r = l.split("\t"); T[f"gpuB{B}"].append(float(r[5]))
for m in sorted(T):
    v = T[m]; print(f"{m:20s} n={len(v):3d} median {statistics.median(v):7.2f} s  max {max(v):7.2f} s")
