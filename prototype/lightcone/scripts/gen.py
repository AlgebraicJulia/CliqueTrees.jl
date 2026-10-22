"""Per setting: build QAOA MaxCut circuit, light-cone TN of every ZZ term (quimb local_expectation rehearse='tn'),
dedup by structural hash, save distinct networks. usage: gen.py kind N p [cap_seconds] [maxterms]"""
import sys, time, pickle, random, collections
import numpy as np, networkx as nx
import quimb as qu, quimb.tensor as qtn

def make_graph(kind, N, seed=1):
    if kind == "reg3":
        return nx.random_regular_graph(3, N, seed=seed)
    G = nx.fast_gnp_random_graph(N, 5.0 / (N - 1), seed=seed)   # avg degree 5
    return G

def struct_hash(inputs, size):
    H = nx.Graph()
    for i, t in enumerate(inputs):
        H.add_node(("t", i), a=f"t{len(t)}")
    for ix in {x for t in inputs for x in t}:
        H.add_node(("i", ix), a=f"i{size[ix]}")
    for i, t in enumerate(inputs):
        for ix in t:
            H.add_edge(("t", i), ("i", ix))
    return nx.weisfeiler_lehman_graph_hash(H, node_attr="a", iterations=6)

def main(kind, N, p, cap=150.0, maxterms=100000):
    G = make_graph(kind, N)
    edges = [tuple(e) for e in G.edges]
    terms = {e: 1.0 for e in edges}
    circ = qtn.circ_qaoa(terms, p, [0.3 + .1 * k for k in range(p)], [0.4 - .05 * k for k in range(p)])
    ZZ = qu.pauli("Z") & qu.pauli("Z")
    rnd = random.Random(0)
    order = edges[:]; rnd.shuffle(order)
    t0 = time.time(); done = 0
    uniq = {}; counts = collections.Counter(); ext_t = 0.0
    for e in order:
        if done >= maxterms or (time.time() - t0 > cap and done >= 20): break
        t1 = time.time()
        tn = circ.local_expectation(ZZ, e, rehearse="tn")
        ext_t += time.time() - t1
        inputs, output, size = tn.get_inputs_output_size_dict()
        inputs = [tuple(t) for t in inputs]
        # drop scalars (carry no index)
        h = struct_hash([t for t in inputs if t], size)
        counts[h] += 1; done += 1
        if h not in uniq:
            uniq[h] = (inputs, size)
    res = dict(kind=kind, N=N, p=p, nedges=len(edges), nsampled=done, ndistinct=len(uniq), counts=dict(counts),
               extract_time=ext_t, per_term=ext_t / done, nets=uniq)
    pickle.dump(res, open(f"data/{kind}_N{N}_p{p}.pkl", "wb"))
    print(f"{kind} N={N} p={p}: terms={len(edges)} sampled={done} distinct={len(uniq)} extract/term={ext_t/done:.3f}s total_wall={time.time()-t0:.1f}", flush=True)

if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), int(a[3]), float(a[4]) if len(a) > 4 else 150.0, int(a[5]) if len(a) > 5 else 100000)
