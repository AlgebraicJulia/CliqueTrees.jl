"""TensorOrder / QTensor ordering-step baselines (CPU). One row per (network, method, budget, seed); columns identical to
einsumbench/worker.py (network method budget seed cfg status wall search post logC width maxop pf_logC pf_width greedy_s extra).
usage: python td_worker.py <queue> <outdir> --tag w0 [--cfg c1]
queue line: <source: einsum|frontier> <network> <method> <budget s> <seed>
Methods (all use the line graph: one vertex per tensor index, a clique per tensor; unweighted, i.e. index dims ignored):
  to-flow    TensorOrder 'line-Flow'   : FlowCutter (PACE'17 build) -> TD stream, kill at budget -> TensorOrder tree extraction
  to-tamaki  TensorOrder 'line-Tamaki' : Tamaki heuristic (TCS-Meiji PACE'17 tw.heuristic.MainDecomposer) -> same
  to-htd     TensorOrder 'line-htd'    : htd_main --opt width --iterations 0 --strategy challenge --preprocessing full -> same
  qt-tamaki  QTensor 'tamaki_<T>'      : tw-heuristic run for T s then SIGTERM -> TD -> qtree get_peo_from_tree PEO -> bucket elimination
  qt-greedy  QTensor default (OrderingOptimizer): min-degree greedy elimination on the line graph -> bucket elimination
TensorOrder's selection: of the TDs a solver prints, the one with the smallest max rank (first on ties); its contraction tree
is built by extract_contraction_tree_line (src/contraction_methods/line_graph_method.py, re-implemented here verbatim in
structure: post-order over the TD rooted at bag 1; at each bag join children's trees, then every not-yet-used tensor whose
index set is contained in the bag, left-deep). Bucket elimination for QTensor: cotengra ContractionTree.from_path(edge_path=...)
with 'greedy' binarisation of each bucket (QTensor contracts each bucket with one einsum; pairwise-binarised cost can differ).
logC etc. are computed with cotengra 0.8.2 exactly as in einsumbench (log10 of multiply-adds; width = log2 largest intermediate).
Deviations from the originals (documented): solver print threshold -p 100000 instead of 100; duplicate line-graph edges removed;
only the min-width TD is converted (not every printed TD); Java heap -Xmx12g not 25-30g.
"""
import argparse, fcntl, heapq, math, os, signal, subprocess, sys, tempfile, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("queue"); ap.add_argument("outdir")
ap.add_argument("--tag", default=str(os.getpid())); ap.add_argument("--cfg", default="c1")
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("CTGPLUGIN", os.path.join(HERE, "..", "ctgplugin")))
sys.path.insert(0, os.environ.get("FTN_DIR", os.path.join(HERE, "..", "frontier", "scripts")))
import cotengra as ctg  # noqa: E402
import tnc  # noqa: E402
import ftn  # noqa: E402

SOLV = os.environ.get("TO_SOLVERS", os.path.join(HERE, "..", "src", "TensorOrder", "solvers"))
JAVA = os.environ.get("JAVA_BIN", "/apps/java/jdk-14/bin/java")
JFLAGS = ["-Xmx12g", "-Xms512m", "-Xss1g"]
PTH = "100000"


def pop(path):
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path) as f:
            lines = f.readlines()
        if not lines:
            return None
        with open(path, "w") as f:
            f.writelines(lines[1:])
        return lines[0].split()


def load(src, name):
    return tnc.load(name) if src == "einsum" else ftn.load(name)


def line_graph(inputs):
    idx = {}
    for t in inputs:
        for i in t:
            idx.setdefault(i, len(idx) + 1)
    edges = set()
    for t in inputs:
        vs = sorted({idx[i] for i in t})
        for p in range(len(vs)):
            for q in range(p + 1, len(vs)):
                edges.add((vs[p], vs[q]))
    return idx, edges


def write_gr(path, n, edges):
    with open(path, "w") as f:
        f.write(f"p tw {n} {len(edges)}\n")
        f.writelines(f"{u} {v}\n" for u, v in sorted(edges))


def parse_td_blocks(lines):
    """PACE .td text -> list of (time, nbags, maxbag, bags{id:[v]}, edges[(a,b)])"""
    out, cur = [], None
    for ln in lines:
        if ln.startswith("s td"):
            _, _, nb, w, nv = ln.split()
            cur = dict(nb=int(nb), w=int(w), bags={}, edges=[], t=time.time())
        elif cur is not None and ln.startswith("b "):
            p = ln.split(); cur["bags"][int(p[1])] = [int(x) for x in p[2:]]
        elif cur is not None and ln.strip() == "=":
            out.append(cur); cur = None
        elif cur is not None and ln[:1].isdigit():
            p = ln.split(); cur["edges"].append((int(p[0]), int(p[1])))
    if cur is not None and cur["bags"] and len(cur["bags"]) == cur["nb"]:   # final block without '=' (Tamaki on SIGTERM)
        out.append(cur)
    return out


def solver_cmd(method, seed, gr):
    if method == "to-flow":
        return [f"{SOLV}/flow-cutter-pace17/flow_cutter_pace17", "-s", str(seed), "-p", PTH], None
    if method == "to-htd":
        return [f"{SOLV}/htd-master/build/bin/htd_main", "-s", str(seed), "--opt", "width", "--iterations", "0",
                "--strategy", "challenge", "--print-progress", "--preprocessing", "full"], None
    if method == "to-tamaki":
        return [JAVA] + JFLAGS + ["tw.heuristic.MainDecomposer", "-s", str(seed), "-p", PTH], f"{SOLV}/TCS-Meiji"
    if method == "qt-tamaki":    # no -p: the best TD is printed when the process is terminated (QTensor's run_heuristic_solver)
        return [JAVA] + JFLAGS + ["tw.heuristic.MainDecomposer", "-s", str(seed)], f"{SOLV}/TCS-Meiji"
    raise ValueError(method)


def run_solver(method, T, seed, gr):
    cmd, cwd = solver_cmd(method, seed, gr)
    lines = []
    t0 = time.time()
    with open(gr) as fin:
        p = subprocess.Popen(cmd, cwd=cwd, stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
                             start_new_session=True)
    rd = threading.Thread(target=lambda: lines.extend(iter(p.stdout.readline, "")), daemon=True); rd.start()
    try:
        p.wait(timeout=T)
    except subprocess.TimeoutExpired:
        if method == "qt-tamaki":
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        else:
            os.killpg(p.pid, signal.SIGKILL)
    rd.join(timeout=5)
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return parse_td_blocks(lines), time.time() - t0


def extract_to(inputs, td, idx):
    """TensorOrder extract_contraction_tree_line -> ssa path (cotengra), indices as in `inputs`."""
    n = len(inputs)
    tsets = [frozenset(idx[i] for i in t) for t in inputs]
    bags = {k: frozenset(v) for k, v in td["bags"].items()}
    nbr = {k: [] for k in bags}
    for u, v in td["edges"]:
        nbr[u].append(v); nbr[v].append(u)
    root = 1 if 1 in bags else min(bags)
    # postorder call order exactly as TreeDecomposition.traverse_postorder
    order, parent, st = [], {}, [(root, None, False)]
    results_children = {}
    while st:
        node, par, expanded = st.pop()
        if node != root and len(nbr[node]) == 1:
            order.append(node); results_children[node] = []
        elif expanded:
            order.append(node); results_children[node] = [c for c in nbr[node] if c != par]
        else:
            st.append((node, par, True))
            for c in nbr[node]:
                if c != par:
                    st.append((c, node, False))
    pos = {b: i for i, b in enumerate(order)}
    bags_of = {}
    for b in order:
        for v in bags[b]:
            bags_of.setdefault(v, []).append(b)
    assigned = {b: [] for b in order}
    last = None
    for t, ts in enumerate(tsets):
        if not ts:
            continue
        v0 = min(ts, key=lambda v: len(bags_of.get(v, ())))
        cand = sorted(bags_of.get(v0, ()), key=pos.get)
        b = next((b for b in cand if ts <= bags[b]), None)
        assigned[b if b is not None else order[-1]].append(t)
    ssa, nxt = [], [n]
    res = {}

    def join(x, y):
        if x is None:
            return y
        if y is None:
            return x
        ssa.append((x, y)); nxt[0] += 1
        return nxt[0] - 1
    for b in order:
        acc = None
        for c in results_children[b]:
            acc = join(acc, res.pop(c))
        for t in assigned[b]:
            acc = join(acc, t)
        res[b] = acc
    return ssa


def peo_from_td(td, idx_n):
    """qtree.graph_model.clique_trees.get_peo_from_tree (clique_vertices=[]): repeatedly remove the first leaf (insertion order of
    bags) that is not the root clique (first bag), appending its vertices that are not in its parent bag; then the root's vertices."""
    bags = {k: frozenset(v) for k, v in td["bags"].items()}
    nbr = {k: set() for k in bags}
    for u, v in td["edges"]:
        nbr[u].add(v); nbr[v].add(u)
    keys = list(bags)
    root = keys[0]
    alive = dict.fromkeys(keys)
    peo = []
    leafheap = [k for k in keys if len(nbr[k]) <= 1 and k != root]
    posn = {k: i for i, k in enumerate(keys)}
    h = [(posn[k], k) for k in leafheap]; heapq.heapify(h)
    while h:
        _, node = heapq.heappop(h)
        if node not in alive or len(nbr[node]) > 1:
            continue
        if not nbr[node]:
            break
        par = next(iter(nbr[node]))
        peo.extend(sorted(bags[node] - bags[par]))
        nbr[par].discard(node); del alive[node]
        if len(nbr[par]) <= 1 and par != root:
            heapq.heappush(h, (posn[par], par))
    peo.extend(sorted(bags[root]))
    seen = set(peo)
    peo.extend(v for v in range(1, idx_n + 1) if v not in seen)    # vertices in no bag (isolated)
    return peo


def greedy_mindeg(n, edges):
    adj = [set() for _ in range(n + 1)]
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    h = [(len(adj[v]), v) for v in range(1, n + 1)]; heapq.heapify(h)
    done, peo = [False] * (n + 1), []
    while h:
        d, v = heapq.heappop(h)
        if done[v] or d != len(adj[v]):
            continue
        done[v] = True; peo.append(v)
        nb = list(adj[v])
        for x in nb:
            adj[x].discard(v)
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                adj[nb[i]].add(nb[j]); adj[nb[j]].add(nb[i])
        for x in nb:
            heapq.heappush(h, (len(adj[x]), x))
    return peo


def stats(tree):
    mo = max((tree.get_flops(nd) for nd, _, _ in tree.traverse()), default=1)
    return tree.contraction_cost(log=10), tree.contraction_width(), math.log2(max(mo, 1))


def finish(tree, inputs, output, size):
    c, w, mo = stats(tree)
    g0 = time.time()
    gt = ctg.array_contract_tree(inputs, output, size, optimize="greedy")
    gs = time.time() - g0
    gc, gw, _ = stats(gt)
    pc, pw = (gc, gw) if gc < c else (c, w)
    return dict(logc=c, width=w, maxop=mo, pf_logc=pc, pf_width=pw, greedy=gs)


def bucket_tree(inputs, output, size, order_syms):
    outs = set(output)
    ep = [v for v in order_syms if v not in outs]
    return ctg.ContractionTree.from_path(inputs, output, size, edge_path=ep, optimize="greedy", autocomplete=True)


def run(method, T, seed, inputs, output, size):
    nan = math.nan
    t0 = time.time()
    idx, edges = line_graph(inputs)
    inv = {v: k for k, v in idx.items()}
    n = len(idx)
    if method == "qt-greedy":
        peo = greedy_mindeg(n, edges)
        dt = time.time() - t0
        tree = bucket_tree(inputs, output, size, [inv[v] for v in peo])
        return "ok", dict(wall=time.time() - t0, search=dt, post=time.time() - t0 - dt, extra=f"nv={n}", **finish(tree, inputs, output, size))
    with tempfile.NamedTemporaryFile("w", suffix=".gr", delete=False, dir=os.environ.get("TMPDIR")) as f:
        gr = f.name
    try:
        write_gr(gr, n, edges)
        tds, sdt = run_solver(method, T, seed, gr)
    finally:
        os.unlink(gr)
    search = time.time() - t0
    if not tds:
        return "no_td", dict(wall=search, search=search, post=0.0, logc=nan, width=nan, maxop=nan, pf_logc=nan, pf_width=nan, greedy=0.0, extra=f"nv={n}")
    td = tds[-1] if method == "qt-tamaki" else min(tds, key=lambda d: max(len(b) for b in d["bags"].values()))   # first min (min returns first)
    p0 = time.time()
    if method.startswith("to-"):
        ssa = extract_to(inputs, td, idx)
        tree = ctg.ContractionTree.from_path(inputs, output, size, ssa_path=ssa, autocomplete=True)
    else:
        peo = peo_from_td(td, n)
        tree = bucket_tree(inputs, output, size, [inv[v] for v in peo])
    post = time.time() - p0
    return "ok", dict(wall=time.time() - t0, search=search, post=post, extra=f"nv={n};ntd={len(tds)};tdw={td['w']-1};tdt={td['t']-t0:.2f}",
                      **finish(tree, inputs, output, size))


os.makedirs(os.path.join(a.outdir, "res"), exist_ok=True)
out = open(os.path.join(a.outdir, "res", f"td_{a.cfg}_{a.tag}.tsv"), "a")
cache = {}
nan = math.nan
while True:
    t = pop(a.queue)
    if t is None:
        break
    src, name, method, T, seed = t[0], t[1], t[2], float(t[3]), int(t[4])
    try:
        if name not in cache:
            cache.clear(); cache[name] = load(src, name)
        inputs, output, size = cache[name]
        st, v = run(method, T, seed, inputs, output, size)
    except Exception as e:  # noqa: BLE001
        st, v = ("error:" + repr(e))[:120].replace("\t", " "), dict(wall=nan, search=nan, post=nan, logc=nan, width=nan, maxop=nan, pf_logc=nan, pf_width=nan, greedy=nan, extra="")
    out.write("\t".join(map(str, (name, method, T, seed, a.cfg, st, round(v["wall"], 4), round(v["search"], 4), round(v["post"], 4),
              round(v["logc"], 4), round(v["width"], 3), round(v["maxop"], 3), round(v["pf_logc"], 4), round(v["pf_width"], 3),
              round(v["greedy"], 4), v["extra"]))) + "\n"); out.flush()
    print(time.strftime("%T"), name, method, T, st, "wall=%.2f logC=%.3f W=%.2f" % (v["wall"], v["logc"], v["width"]), flush=True)
