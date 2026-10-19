"""Convert a sliced set between cotengra symbols (path.json 'sliced_inds') and vertex ids.
usage: python sconv.py tovert <path.json> <out.txt> | python sconv.py tojson <in.txt> <out.json>"""
import json, sys
import cotengra as ctg
mode, a, b = sys.argv[1:4]
if mode == "tovert":
    inv = {ctg.get_symbol(i): i for i in range(1, 200000)}
    S = json.load(open(a))["sliced_inds"]
    open(b, "w").write("".join(f"{inv[x]}\n" for x in S))
else:
    S = [int(x) for x in open(a).read().split()]
    json.dump({"ssa_path": [], "sliced_inds": [ctg.get_symbol(v) for v in S]}, open(b, "w"))
