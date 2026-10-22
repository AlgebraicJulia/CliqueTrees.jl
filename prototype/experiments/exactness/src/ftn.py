"""Shared helpers: load a frontier network, build / measure / refine / slice cotengra trees.

Cost conventions (cotengra 0.8.2, as in Gray & Kourtis 2021 and the cotengra notebooks):
  cost  = tree.contraction_cost()  = number of scalar multiply-adds summed over all pairwise
          contractions and over all slices (the "C" of Gray & Kourtis; real FLOPs = 2C for real,
          ~8C for complex dtype; opt_einsum's "FLOPs" progress number is 2C)
  width = tree.contraction_width() = log2 of the largest intermediate (elements), per slice
  maxop = log2 of the largest single pairwise contraction (product of the dims of all indices
          involved): the tree-side analogue of a bag width
"""
import json
import math
import os
import time

import cotengra as ctg

FDIR = os.environ.get("FRONTIER_DIR", "")


def sym(v):
    return ctg.get_symbol(int(v))


def load(name, fdir=None):
    fdir = fdir or os.environ.get("FRONTIER_DIR", FDIR)
    with open(os.path.join(fdir, "data", name + ".json")) as f:
        d = json.load(f)
    # labels: one unicode symbol per index (cotengrust's optimal needs single characters);
    # label sym(v) = line-graph vertex v of the .tnc
    inputs = [tuple(sym(x) for x in t) for t in d["inputs"] if t]   # scalar tensors carry no index
    output = tuple(sym(x) for x in d["output"])
    size = {sym(int(k)): int(v) for k, v in d["size_dict"].items()}
    return inputs, output, size


def tree_from_order(inputs, output, size, order, optimize="auto-hq"):
    """Elimination ordering of line-graph vertices (= index labels) -> contraction tree.

    Bucket elimination (cotengra ContractionTree.from_path(edge_path=...), the same bridge
    cotengra uses for its own QuickBB / FlowCutter tree-decomposition optimizers): for each
    index in order, contract every current tensor carrying it; each multi-way merge is
    binarized by `optimize` ('auto-hq': exact dynamic programming for small merges, greedy
    otherwise). Open indices are never eliminated: the remaining tensors are contracted last.
    """
    outs = set(output)
    if outs:
        order = open_last(inputs, output, [sym(v) for v in order])
    else:
        order = [sym(v) for v in order]
    ep = [v for v in order if v not in outs]
    return ctg.ContractionTree.from_path(inputs, output, size, edge_path=ep, optimize=optimize, autocomplete=True)


def open_last(inputs, output, order):
    """A perfect elimination ordering of the SAME filled graph that ends with the open indices.

    Dropping the open indices from an ordering is not enough: an open index that the ordering
    eliminates early is absent from the later bags, yet it stays on the tensors until the end, so
    the intermediates outgrow the bags. The open indices form a clique of the line graph, hence of
    the filled graph; maximum cardinality search on the (chordal) filled graph started from that
    clique visits it first, and its reverse is a PEO (no new fill, same bags) that eliminates the
    open indices last -- the re-rooting of draft Prop. 5 / Algorithm 1 step 1."""
    adj = {v: set() for v in order}
    for t in inputs:
        for a in t:
            adj[a].update(t)
    outs = list(dict.fromkeys(output))
    for a in outs:
        adj[a].update(outs)
    for v in adj:
        adj[v].discard(v)
    pos = {v: i for i, v in enumerate(order)}
    # symbolic elimination: filled graph
    fill = {v: set(adj[v]) for v in order}
    for v in order:
        later = [u for u in fill[v] if pos[u] > pos[v]]
        for u in later:
            fill[u].update(w for w in later if w != u)
    # MCS with buckets, open clique first
    import heapq
    card = {v: 0 for v in order}
    seen = set()
    visit = []
    heap = []
    prio_open = set(outs)
    for v in order:
        heapq.heappush(heap, (0, 0 if v in prio_open else 1, pos[v], v))
    while heap:
        c, o, _, v = heapq.heappop(heap)
        if v in seen or -c != card[v]:
            continue
        seen.add(v); visit.append(v)
        for u in fill[v]:
            if u not in seen:
                card[u] += 1
                heapq.heappush(heap, (-card[u], 0 if u in prio_open else 1, pos[u], u))
    assert len(visit) == len(order)
    peo = visit[::-1]
    assert set(peo[-len(outs):]) == set(outs), "open clique not last"
    return peo


def stats(tree):
    mo = max((tree.get_flops(nd) for nd, _, _ in tree.traverse()), default=1)
    return dict(
        cost=tree.contraction_cost(log=10),
        width=tree.contraction_width(),
        maxop=math.log2(mo),
        nslices=tree.multiplicity,
        nsliced=len(tree.sliced_inds),
    )


def refine(tree, seed=0):
    """cotengra subtree reconfiguration (subtrees of 8 leaves re-solved optimally for flops),
    repeated until a pass no longer lowers the cost (at most 8 passes)."""
    t = tree.copy()
    prev = t.contraction_cost()
    for _ in range(8):
        t.subtree_reconfigure_(subtree_size=8, minimize="flops", seed=seed)
        c = t.contraction_cost()
        if c >= prev * (1 - 1e-9):
            break
        prev = c
    return t


def slice_only(tree, target_log2, seed=0):
    return tree.slice(target_size=2 ** target_log2, minimize="flops", seed=seed)


def slice_reconf(tree, target_log2):
    return tree.slice_and_reconfigure(target_size=2 ** target_log2, minimize="flops")


def post(tree, targets):
    """raw / refined / sliced measurements of one tree. Returns (rows, refined tree), rows =
    list of (stage, stats, seconds). Slicing is tried four ways -- {raw, refined} tree x
    {tree.slice (best of 16 greedy slicings), tree.slice_and_reconfigure (slice one index at a
    time, subtree-reconfiguring in between)} -- and stage 'best_slice<t>' is the cheapest."""
    rows = [("raw", stats(tree), 0.0)]
    t0 = time.time(); r = refine(tree); rows.append(("refined", stats(r), time.time() - t0))
    for tg in targets:
        best = None
        for base, bt in (("raw", tree), ("refined", r)):
            for how, fn in (("slice", slice_only), ("slicereconf", slice_reconf)):
                t0 = time.time()
                s = fn(bt, tg) if bt.contraction_width() > tg else bt
                st = stats(s)
                rows.append((f"{base}+{how}{tg}", st, time.time() - t0))
                if best is None or st["cost"] < best["cost"]:
                    best = st
        rows.append((f"best_slice{tg}", best, 0.0))
    return rows, r
