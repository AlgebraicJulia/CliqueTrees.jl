"""Check a cut network against the sliced cotengra tree it came from: rebuild the tree on the
cut network (same tensor order) and report its stats; write its induced ordering.
usage: python cutcheck.py <cut name> <path.json> <out.ord>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ftn, cotengra as ctg
name, pj, out = sys.argv[1:4]
i, o, s = ftn.load(name)
p = json.load(open(pj))
t = ctg.ContractionTree.from_path(i, o, s, ssa_path=p["ssa_path"])
order, seen = [], set()
for node, l, r in t.traverse():
    gone = (set(t.get_legs(l)) | set(t.get_legs(r))) - set(t.get_legs(node))
    for ix in sorted(gone):
        if ix not in seen: seen.add(ix); order.append(ix)
order += [ix for ix in s if ix not in seen]
inv = {ctg.get_symbol(k): k for k in range(1, len(s) + 2)}
st = ftn.stats(t)
with open(out, "w") as f:
    f.write("\t".join(map(str, (name, "ctgslice", 0, 0, 0, st["maxop"], len(s), 0))) + "\n")
    f.write(" ".join(str(inv[x]) for x in order) + "\n")
print(name, "per-slice tree", st)
