"""CPU baseline: cotengra parallel_temper_tree (PT over contraction trees with integrated slicing).
usage: python ctg_pt.py <name> <seed .tree> <budget s> <parallel (0 = in-process, n = n worker processes)>
                        <num_trees> <out prefix> [Wt] [numiter] [rseed]
Starts every replica from the seed tree, temperatures tstart=1 -> tfinal=0.01 (cotengra defaults),
minimize='flops', slice_mode='drift', parallel_slice_mode='temperature' (defaults) with target_size 2**Wt.
max_time = budget caps the run; numiter (default 50, the cotengra default) sets the schedule length.
Writes <prefix>.tree (ssa path + sliced ids) and prints the cotengra cost and wall time."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cotengra as ctg
from cotengra.pathfinders.path_simulated_annealing import parallel_temper_tree
import tnio

name, seedf, budget, par, ntrees, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
Wt = float(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] != "Inf" else None
numiter = int(sys.argv[8]) if len(sys.argv) > 8 else 50
rseed = int(sys.argv[9]) if len(sys.argv) > 9 else 1
inputs, output, size, _ = tnio.load(name)
nt, ssa, sl0, _ = tnio.read_tree(seedf)
seed = tnio.ctg_tree(inputs, output, size, ssa, sl0)
kw = dict(num_trees=ntrees, numiter=numiter, minimize="flops", max_time=budget, seed=rseed,
          parallel=(par if par > 0 else False), progbar=False)
if Wt:
    kw["target_size"] = 2 ** int(Wt)
t0 = time.time()
tree = parallel_temper_tree(seed, **kw)
ts = time.time() - t0
if Wt and tree.contraction_width() > Wt + 1e-9:     # make sure the reported tree respects the cap
    tree = tree.slice(target_size=2 ** int(Wt), minimize="flops")
st = tnio.tree_stats(tree)
inv = {tnio.sym(v + 1): v + 1 for v in range(len(size))}
sl = [inv[x] for x in tree.sliced_inds]
tnio.write_tree(out + ".tree", len(inputs), [tuple(map(int, p)) for p in tree.get_ssa_path()], sl,
                {"t": f"{ts:.3f}", "ctg_log2C": f"{st['log2C']:.6f}", "method": "ctg_parallel_temper_tree",
                 "num_trees": ntrees, "parallel": par, "numiter": numiter, "Wt": Wt or "Inf"})
print(f"{name}\tctgPT\tp{par}\tn{ntrees}\tni{numiter}\tW{Wt}\tbudget{budget}\t{ts:.1f}s\t{st['log2C']:.4f}\t{st['W']:.1f}\t{st['nsliced']}", flush=True)
