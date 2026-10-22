"""cotengra-compatible contraction-path optimizer backed by the CliqueTrees.jl pipeline.

    import cotengra as ctg
    from ctg_cliquetrees import CliqueTreesOptimizer

    opt = CliqueTreesOptimizer(method='hbtnd', time=1.0, threads=8)
    tree = ctg.array_contract_tree(inputs, output, size_dict, optimize=opt)   # or opt.search(...)
    ssa  = opt.ssa_path(inputs, output, size_dict)                            # SSA path
    path = opt(inputs, output, size_dict)                                     # opt_einsum linear path
    np.einsum / opt_einsum.contract(eq, *arrays, optimize=opt)                # also works

Pipeline: the network is turned into its (weighted) line graph in a persistent Julia subprocess
(JSON lines over stdin/stdout, so JIT is paid once per process), one of our elimination-ordering
methods runs there, and the ordering comes back and is turned into a contraction tree by bucket
elimination (cotengra ContractionTree.from_path(edge_path=...), subtrees binarized with
'auto-hq'), then optionally refined by cotengra subtree reconfiguration (the protocol of draft
Sec. 5.3: both sides get their own local refinement). This is the conversion of
tn-work/frontier/scripts/ftn.py, copied here so the plugin is self-contained.

Methods: 'hbtnd' (HBT initialised by nested dissection, default), 'hbt', 'nd', 'bw' (greedy-ish
width heuristics, ms), 'treesa' (OMEinsumContractionOrders TreeSA, time-bounded by calibration).
`time` is the search budget handed to HBT; the wall time of a call also includes the line graph,
the preprocessing pipeline, the tree conversion and the refinement (see `last_info`).

portfolio=True returns the cheaper (by cotengra's total cost C) of our tree and cotengra's greedy
tree, so the result is never worse than greedy, at the cost of a few ms.

slicing: None, or a target log2 size; the tree is sliced (cotengra slice_and_reconfigure) and the
sliced indices are available as `opt.last_tree.sliced_inds` (a path alone cannot carry slicing).
"""
import atexit
import json
import os
import select
import shutil
import subprocess
import threading
import time as _time

import cotengra as ctg
from cotengra.oe import PathOptimizer

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ("hbtnd", "hbt", "nd", "bw", "treesa")
_SERVERS = {}


class JuliaServer:
    """One persistent `julia -t threads server.jl` process speaking JSON lines."""

    def __init__(self, threads, julia=None, env=None, log=None):
        self.threads = threads
        julia = julia or os.environ.get("CTG_JULIA") or shutil.which("julia") or os.path.expanduser("~/.juliaup/bin/julia")
        env = env or os.environ.get("CTG_JLENV") or os.path.join(HERE, "jlenv")
        self.cmd = [julia, f"-t{threads}", f"--project={env}", os.path.join(HERE, "server.jl")]
        self.log = log
        self.proc = None
        self.warmed = set()
        self.lock = threading.Lock()

    def start(self):
        err = open(self.log, "a") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                     text=True, bufsize=1, start_new_session=True)
        self.warmed = set()

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def kill(self):
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc.wait()
            self.proc = None

    def request(self, req, timeout):
        """Send one request, wait up to `timeout` s for the reply line. On timeout the server is killed
        (restarted by the next call) and TimeoutError raised."""
        with self.lock:
            if not self.alive():
                self.start()
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            r, _, _ = select.select([self.proc.stdout], [], [], timeout)
            if not r:
                self.kill()
                raise TimeoutError(f"julia server gave no reply in {timeout:.0f} s")
            line = self.proc.stdout.readline()
            if not line:
                self.kill()
                raise RuntimeError("julia server died")
            rep = json.loads(line)
            if "error" in rep:
                raise RuntimeError("julia server: " + rep["error"])
            return rep

    def warm(self, methods, timeout=900):
        todo = [m for m in methods if m not in self.warmed]
        if todo:
            self.request({"cmd": "warm", "methods": todo}, timeout)
            self.warmed.update(todo)


def get_server(threads, **kw):
    s = _SERVERS.get(threads)
    if s is None:
        s = _SERVERS[threads] = JuliaServer(threads, **kw)
    return s


@atexit.register
def _shutdown():
    for s in _SERVERS.values():
        s.kill()


# ---------------------------------------------------------------------------- ordering -> tree
def open_last(inputs, output, order):
    """A perfect elimination ordering of the SAME filled graph that ends with the open indices (the
    re-rooting of draft Prop. 5): symbolic elimination, then maximum cardinality search started from
    the open clique, reversed. (Copied from frontier/scripts/ftn.py.)"""
    import heapq
    adj = {v: set() for v in order}
    for t in inputs:
        for a in t:
            if a in adj:
                adj[a].update(x for x in t if x in adj)
    outs = [a for a in dict.fromkeys(output) if a in adj]
    for a in outs:
        adj[a].update(outs)
    for v in adj:
        adj[v].discard(v)
    pos = {v: i for i, v in enumerate(order)}
    fill = {v: set(adj[v]) for v in order}
    for v in order:
        later = [u for u in fill[v] if pos[u] > pos[v]]
        for u in later:
            fill[u].update(w for w in later if w != u)
    card = {v: 0 for v in order}
    seen, visit, heap = set(), [], []
    prio = set(outs)
    for v in order:
        heapq.heappush(heap, (0, 0 if v in prio else 1, pos[v], v))
    while heap:
        c, o, _, v = heapq.heappop(heap)
        if v in seen or -c != card[v]:
            continue
        seen.add(v); visit.append(v)
        for u in fill[v]:
            if u not in seen:
                card[u] += 1
                heapq.heappush(heap, (-card[u], 0 if u in prio else 1, pos[u], u))
    peo = visit[::-1]
    if outs:
        assert set(peo[-len(outs):]) == set(outs), "open clique not last"
    return peo


def tree_from_order(inputs, output, size, order, optimize="greedy"):
    """Elimination ordering of index labels -> ContractionTree (bucket elimination; same as ftn.py)."""
    outs = set(output)
    order = open_last(inputs, output, list(order)) if outs else list(order)
    ep = [v for v in order if v not in outs]
    return ctg.ContractionTree.from_path(inputs, output, size, edge_path=ep, optimize=optimize, autocomplete=True)


def refine(tree, seed=0, max_time=None, chunk=64):
    """cotengra subtree reconfiguration (subtrees of 8 leaves, minimize flops), in chunks of `chunk`
    reconfigurations, until a chunk stops improving, 8 full-pass equivalents are done, or `max_time`
    seconds have elapsed (checked between chunks; None = no cap). Own time cap so refinement cannot
    dominate the wall time of a short budget."""
    t = tree.copy()
    t0 = _time.time()
    prev = t.contraction_cost()
    for k in range(max(1, 8 * max(1, t.N // chunk))):
        if max_time is not None and _time.time() - t0 >= max_time:
            break
        t.subtree_reconfigure_(subtree_size=8, minimize="flops", seed=seed + k, maxiter=chunk)
        c = t.contraction_cost()
        if c >= prev * (1 - 1e-9):
            break
        prev = c
    return t


# ---------------------------------------------------------------------------- the optimizer
class CliqueTreesOptimizer(PathOptimizer):
    """See the module docstring. Parameters:

    method     'hbtnd' | 'hbt' | 'nd' | 'bw' | 'treesa'
    time       search budget in seconds for hbt / hbtnd / treesa (ignored by nd, bw)
    threads    Julia threads (the server is started with `julia -t threads`; one server per value)
    slicing    None or target log2 size of the largest intermediate
    portfolio  return the cheaper of (our tree, cotengra greedy)
    refine     cotengra subtree reconfiguration of our tree (draft Sec. 5.3)
    refine_time  cap (s) on the refinement; None -> max(0.1 * time, 0.05)
    convert    cotengra optimizer used to binarize the buckets in the ordering->tree conversion
               ('greedy' default: ~ms; 'auto-hq' was 4-7 s on some networks and did cotengra's own search)
    seed       HBT seed
    timeout    hard cap in seconds on the Julia call (default 3 * time + 120); on timeout the server
               is restarted and, with portfolio=True, the greedy tree is returned (else TimeoutError)
    """

    def __init__(self, method="hbtnd", time=1.0, threads=8, slicing=None, portfolio=True, refine=True,
                 refine_time=None, convert="greedy", seed=1, timeout=None, julia=None, jlenv=None, log=None):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method, self.time, self.threads = method, float(time), int(threads)
        self.slicing, self.portfolio, self.refine, self.seed = slicing, portfolio, refine, seed
        self.refine_time = max(0.1 * self.time, 0.05) if refine_time is None else float(refine_time)
        self.convert = convert
        self.timeout = timeout if timeout is not None else 3 * self.time + 120
        self._srv_kw = dict(julia=julia, env=jlenv, log=log)
        self.last_tree = None
        self.last_info = {}

    @property
    def server(self):
        return get_server(self.threads, **self._srv_kw)

    def warm(self):
        """Pay the Julia JIT now (first call otherwise)."""
        self.server.warm([self.method])

    def close(self):
        self.server.kill()

    # -- core
    def _julia_tree(self, inputs, output, size_dict):
        labels = {}
        for t in inputs:
            for a in t:
                labels.setdefault(a, len(labels) + 1)
        for a in output:
            labels.setdefault(a, len(labels) + 1)
        inv = {v: k for k, v in labels.items()}
        dims = [int(size_dict[inv[i]]) for i in range(1, len(labels) + 1)]
        req = {"cmd": "treesa" if self.method == "treesa" else "order", "method": self.method, "time": self.time,
               "threads": self.threads, "seed": self.seed, "dims": dims,
               "inputs": [[labels[a] for a in t] for t in inputs], "output": [labels[a] for a in output]}
        srv = self.server
        srv.warm([self.method])
        t0 = _time.time()
        rep = srv.request(req, self.timeout)
        tj = _time.time() - t0
        t1 = _time.time()
        if self.method == "treesa":
            tree = ctg.ContractionTree.from_path(inputs, output, size_dict, ssa_path=[tuple(p) for p in rep["ssa"]])
            info = dict(julia=tj, order=None)
        else:
            order = [inv[i] for i in rep["order"]]
            tree = tree_from_order(inputs, output, size_dict, order, optimize=self.convert)
            info = dict(julia=tj, julia_seconds=rep["seconds"], bagwidth=rep["bagwidth"], order=order)
        info["convert_only"] = _time.time() - t1
        if self.refine:
            r0 = _time.time()
            tree = refine(tree, seed=self.seed, max_time=self.refine_time)
            info["refine_s"] = _time.time() - r0
        info["convert"] = _time.time() - t1     # total post-processing after the Julia call
        return tree, info

    def search(self, inputs, output, size_dict, **kwargs):
        """-> cotengra ContractionTree (what ctg.array_contract_tree calls)."""
        inputs = [tuple(t) for t in inputs]
        output = tuple(output)
        t0 = _time.time()
        info = {}
        tree = None
        try:
            tree, info = self._julia_tree(inputs, output, size_dict)
        except (TimeoutError, RuntimeError) as e:
            if not self.portfolio:
                raise
            info = dict(error=repr(e))
        if self.portfolio:
            g0 = _time.time()
            gt = ctg.array_contract_tree(inputs, output, size_dict, optimize="greedy")
            info["greedy_seconds"] = _time.time() - g0
            if tree is None or gt.contraction_cost() < tree.contraction_cost():
                info["chose"] = "greedy"
                tree = gt
            else:
                info["chose"] = self.method
        if self.slicing is not None and tree.contraction_width() > self.slicing:
            tree = tree.slice_and_reconfigure(target_size=2 ** self.slicing, minimize="flops")
        info["seconds"] = _time.time() - t0
        self.last_tree, self.last_info = tree, info
        return tree

    def ssa_path(self, inputs, output, size_dict):
        return [tuple(p) for p in self.search(inputs, output, size_dict).get_ssa_path()]

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        # opt_einsum / np.einsum contract: linear path format
        return self.search(inputs, output, size_dict).get_path()

    # PathOptimizer subclasses are called once per contraction; no state is shared between calls
    def _check_args_against_first_call(self, *a, **k):
        pass
