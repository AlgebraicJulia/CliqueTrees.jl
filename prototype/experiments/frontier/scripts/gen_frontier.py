"""Frontier instances: Google Sycamore 53-qubit RCS circuits (the published
circuit_n53_m{m}_s0_e0_pABCDCDAB.qsim files, identical in cotengra/examples and
alibaba/acqdp/benchmark) turned into tensor networks by quimb exactly as the literature does.

Every network is written twice:
  data/<name>.tnc   corpus binary format (scripts/tncwrite.py of tncorpus), read by TNCorpus.load
  data/<name>.json  {"inputs", "output", "size_dict"} with the SAME integer index labels 1..n
                    as the .tnc vertices (so a Julia ordering of vertices is a cotengra edge path)
and one manifest line (manifest_parts/<group>.tsv).

Variants (quimb 1.15.0, Circuit.from_qsim_file defaults: gate_opts contract='auto-split-gate'):
  amp     single amplitude <b|C|0>, b = seeded random bitstring, quimb amplitude_tn default
          simplification 'ADCRS' (atol 1e-12, equalize_norms) -- Gray & Kourtis 2021 setting
  raw     the same amplitude with no simplification (simplify_sequence='')
  margK   2^K amplitudes: the K qubits circ.calc_qubit_ordering()[-K:] open, the other 53-K
          fixed to a seeded random bitstring, then 'ADCRS' (atol 1e-6) -- exactly
          Circuit.compute_marginal(rehearse='tn') as used by cotengra's notebooks reproducing
          Huang et al. 2005.06787 (K=6) and Pan & Zhang 2103.03074 (K=21)
  openQ   2^|Q| amplitudes with an explicit open qubit set Q (from the literature)

usage: python -I gen_frontier.py <qsim_dir> <out_dir> <spec> [<spec> ...]
spec:  amp:m  raw:m  marg:m:K  open:m:q1,q2,...   (m = cycles)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.environ.get("TNCORPUS_DIR", "/path/to/tn-work/tncorpus"), "scripts"))
from tncwrite import manifest_line, write_tn  # noqa: E402

SEED = 2026  # bitstring seed (structure does not depend on the bit values)


def circuit(qsim_dir, m):
    import quimb.tensor as qtn
    f = os.path.join(qsim_dir, f"circuit_n53_m{m}_s0_e0_pABCDCDAB.qsim")
    return qtn.Circuit.from_qsim_file(f)


def bits(n, seed):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(["0", "1"], size=n))


def tn_triplet(tn, output_inds):
    inputs = [tuple(t.inds) for t in tn]
    size = {}
    for t in tn:
        for ix, d in zip(t.inds, t.shape):
            size[ix] = int(d)
    return inputs, tuple(output_inds), size


def build(qsim_dir, spec):
    kind, *rest = spec.split(":")
    if kind == "ampfile":
        # ampfile:<qsim file>:<name>: single amplitude of an arbitrary qsim circuit, as 'amp'
        import quimb.tensor as qtn
        circ = qtn.Circuit.from_qsim_file(rest[0])
        b = bits(circ.N, SEED)
        tn = circ.amplitude_tn(b, simplify_sequence="ADCRS")
        return rest[1], tn_triplet(tn, ()), f"gen=qsimfile;file={os.path.basename(rest[0])};N={circ.N};kind=amp;b_seed={SEED};simplify=ADCRS;atol=1e-12"
    if kind == "ampcirq":
        # ampcirq:<cirq json>:<name>: single amplitude of a cirq-JSON circuit (Morvan et al. 2024
        # Zenodo 11663206); gates applied as raw unitaries, measurements dropped, then as 'amp'
        import cirq
        import quimb.tensor as qtn
        c = cirq.read_json(rest[0])
        qs = sorted(c.all_qubits()); idx = {q: i for i, q in enumerate(qs)}
        circ = qtn.Circuit(len(qs))
        nm = 0
        for op in c.all_operations():
            if cirq.is_measurement(op):
                nm += 1; continue
            circ.apply_gate_raw(cirq.unitary(op), [idx[q] for q in op.qubits])
        b = bits(circ.N, SEED)
        tn = circ.amplitude_tn(b, simplify_sequence="ADCRS")
        return rest[1], tn_triplet(tn, ()), (f"gen=cirqjson;file={os.path.basename(rest[0])};N={circ.N};moments={len(c)};"
                                            f"dropped_measurements={nm};kind=amp;b_seed={SEED};simplify=ADCRS;atol=1e-12")
    m = int(rest[0])
    circ = circuit(qsim_dir, m)
    N = circ.N
    if kind == "amp":
        b = bits(N, SEED + m)
        tn = circ.amplitude_tn(b, simplify_sequence="ADCRS")
        return f"syc53_m{m}_amp", tn_triplet(tn, ()), f"gen=qsim;n=53;m={m};s=0;e=0;p=ABCDCDAB;kind=amp;b_seed={SEED + m};simplify=ADCRS;atol=1e-12"
    if kind == "raw":
        b = bits(N, SEED + m)
        tn = circ.amplitude_tn(b, simplify_sequence="")
        return f"syc53_m{m}_raw", tn_triplet(tn, ()), f"gen=qsim;n=53;m={m};s=0;e=0;p=ABCDCDAB;kind=amp;b_seed={SEED + m};simplify=none"
    if kind in ("marg", "open"):
        if kind == "marg":
            K = int(rest[1])
            where = tuple(sorted(circ.calc_qubit_ordering()[-K:]))
            name = f"syc53_m{m}_marg{K}"
        else:
            where = tuple(sorted(int(q) for q in rest[1].split(",")))
            K = len(where)
            name = f"syc53_m{m}_open{K}" + (rest[2] if len(rest) > 2 else "")
        fixq = [q for q in range(N) if q not in where]
        b = bits(len(fixq), SEED + 100 * m + K)
        fix = dict(zip(fixq, b))
        tn = circ.compute_marginal(where=where, fix=fix, rehearse="tn")
        out = [circ.ket_site_ind(i) for i in where]
        out = [ix for ix in out if ix in tn.ind_map]
        return name, tn_triplet(tn, out), (f"gen=qsim;n=53;m={m};s=0;e=0;p=ABCDCDAB;kind={kind};K={K};"
                                           f"open={'-'.join(map(str, where))};fix_seed={SEED + 100 * m + K};simplify=ADCRS;atol=1e-6")
    raise ValueError(spec)


def main(argv):
    qsim_dir, out_dir, specs = argv[0], argv[1], argv[2:]
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "manifest_parts"), exist_ok=True)
    for spec in specs:
        t0 = time.time()
        name, (inputs, output, size), params = build(qsim_dir, spec)
        fam = name.split("_")[0]
        st = write_instance(out_dir, name, fam, inputs, output, size, params)
        print(f"{name}: n={st['n']} m={st['m']} nt={st['ntensors']} open={st['nopen']} wsum={st['wsum']:.0f} ({time.time() - t0:.1f}s)", flush=True)


def write_instance(out_dir, name, fam, inputs, output, size, params):
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "manifest_parts"), exist_ok=True)
    rel = f"data/{name}.tnc"
    st = write_tn(os.path.join(out_dir, rel), inputs, output, size)
    # relabel exactly as write_tn does (first appearance order over tensors, dims > 1 only)
    lab = {}
    for t in inputs:
        for x in t:
            if size[x] > 1 and x not in lab:
                lab[x] = len(lab) + 1
    for x in output:
        if size[x] > 1 and x not in lab:
            lab[x] = len(lab) + 1
    jin = [sorted({lab[x] for x in t if size[x] > 1}) for t in inputs]
    jout = sorted({lab[x] for x in output if size[x] > 1})
    jsz = {str(v): size[k] for k, v in lab.items()}
    with open(os.path.join(out_dir, "data", name + ".json"), "w") as f:
        json.dump({"inputs": jin, "output": jout, "size_dict": jsz, "params": params}, f)
    with open(os.path.join(out_dir, "manifest_parts", name + ".tsv"), "w") as f:
        f.write(manifest_line(name, fam, "frontier", st, params, rel))
    return st


if __name__ == "__main__":
    main(sys.argv[1:])
