"""Parser for the .td files written by tdgen.jl."""


def parse(path):
    """-> dict(net=..., exact=(width, lower, certified, seconds) or None,
               orders=[dict(tag, seconds, width, lcost, bt_cost, bt_width, bt_peak, br_cost, br_width, br_peak,
                            br_seconds, order, ssa_bt, ssa_br)],
               ours=[dict(tag, seconds, source, cost, width, peak, tdwidth, tdlower, certified, ssa)])"""
    with open(path) as f:
        lines = f.read().split("\n")
    out = dict(net=None, exact=None, orders=[], ours=[], complete=False)
    i = 0

    def pairs(s):
        v = [int(x) for x in s.split()]
        return [(v[k], v[k + 1]) for k in range(0, len(v), 2)]

    while i < len(lines):
        f = lines[i].split("\t")
        if f[0] == "NET":
            out["net"] = dict(name=f[1], n=int(f[2]), ntensors=int(f[3]), nopen=int(f[4]))
        elif f[0] == "EXACT":
            out["exact"] = dict(width=float(f[1]), lower=float(f[2]), certified=f[3] == "1", seconds=float(f[4]))
        elif f[0] == "ORDER":
            keys = ("seconds", "width", "lcost", "bt_cost", "bt_width", "bt_peak", "br_cost", "br_width", "br_peak",
                    "br_seconds")
            d = dict(tag=f[1], **{k: float(x) for k, x in zip(keys, f[2:])})
            d["order"] = [int(x) for x in lines[i + 1].split()]
            d["ssa_bt"] = pairs(lines[i + 2])
            d["ssa_br"] = pairs(lines[i + 3])
            out["orders"].append(d)
            i += 3
        elif f[0] == "OURS":
            d = dict(tag=f[1], seconds=float(f[2]), source=f[3], cost=float(f[4]), width=float(f[5]), peak=float(f[6]),
                     tdwidth=float(f[7]), tdlower=float(f[8]), certified=f[9] == "1")
            d["ssa"] = pairs(lines[i + 1])
            out["ours"].append(d)
            i += 1
        elif f[0] == "END":
            out["complete"] = True
        i += 1
    return out
