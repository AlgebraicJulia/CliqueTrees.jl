#!/usr/bin/env python3 -I
"""Benchmark driver.  usage: run.py SOLVER SET TIMEOUT PAR [OUTDIR]
SOLVER: tamaki17 | tamaki16 | uplow22 | septw    SET: exact | bonus
Writes OUTDIR/<SOLVER>_<SET>.tsv: inst status secs width valid
status: OK (finished, optimal claimed) | TIMEOUT | FAIL.  For timeouts, width is the best
upper bound found if the solver left a td file (uplow22), else '-'.
"""
import sys, os, subprocess, time, shutil, tempfile, re
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
JAVA = os.environ.get("JAVA", f"{HERE}/jdk/bin/java"); VAL = f"{HERE}/src/td-validate/td-validate"
JULIA = os.environ.get("JULIA", os.path.expanduser("~/.juliaup/bin/julia"))
MTENV = os.environ.get("MTENV", os.path.expanduser("~/Projects/GATAS/tn-work/mtenv"))
JV = ["-XX:+UseSerialGC", "-Xmx" + os.environ.get("JXMX", "8g"), "-Xss10m"]

def td_width(path):
    try:
        for l in open(path):
            if l.startswith("s td"): return int(l.split()[3]) - 1
    except OSError: pass
    return None

def cmd(solver, gr, td, T):
    if solver == "tamaki17":
        return [JAVA, *JV, "-XX:ActiveProcessorCount=1", "-cp", f"{HERE}/src/PACE2017-TrackA", "tw.exact.MainDecomposer"], gr, td
    if solver == "tamaki16":
        return [f"{HERE}/src/treewidth-exact/tw-exact"], gr, td
    if solver == "uplow22":  # needs two threads (upper and lower bound)
        return [JAVA, *JV, "-XX:ActiveProcessorCount=2", "-cp", f"{HERE}/build/tw2022", "io.github.twalgor.main.UpLow", gr, td, td + ".mnr"], None, None
    if solver == "septw":
        return [JULIA, "-t1", f"--project={MTENV}", f"{HERE}/ctrun.jl", str(T), gr, td], None, None

def one(a):
    solver, setname, name, T, tmp = a
    gr = f"{ROOT}/data/plain/{setname}/{name}.gr"; td = f"{tmp}/{name}.td"
    c, fin, fout = cmd(solver, gr, td, T)
    inp = open(fin) if fin else None
    out = open(fout, "w") if fout else subprocess.PIPE
    t0 = time.monotonic(); status = None; secs = None
    try:
        r = subprocess.run(c, stdin=inp, stdout=out, stderr=subprocess.DEVNULL, timeout=T + (90 if solver == "septw" else 0), text=True)
        secs = time.monotonic() - t0
        if solver == "septw":
            f = r.stdout.split()
            status = {"FOUND": "OK", "TIMEOUT": "TIMEOUT"}.get(f[0], "FAIL") if f else "FAIL"; secs = float(f[2]) if f else secs
        else:
            status = "OK" if r.returncode == 0 and secs <= T else ("TIMEOUT" if secs > T else "FAIL")
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"; secs = T
    if status == "OK" and solver == "septw" and secs > T: status = "TIMEOUT"
    for h in (inp, out):
        if h not in (None, subprocess.PIPE): h.close()
    w = td_width(td) if os.path.exists(td) and os.path.getsize(td) else None
    valid = "-"
    if w is not None and (status == "OK" or solver == "uplow22"):
        p = subprocess.run([VAL, gr, td], capture_output=True, text=True); v = (p.stdout + p.stderr).strip()
        valid = "y" if v == "valid" else "n"
        if valid == "n": status = "FAIL" if status == "OK" else status
    if status == "OK" and solver == "uplow22" and False: pass
    return name, status, secs, (w if w is not None else "-"), valid

def main():
    solver, setname, T, par = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    outdir = sys.argv[5] if len(sys.argv) > 5 else f"{ROOT}/results"
    os.makedirs(outdir, exist_ok=True)
    names = sorted(f[:-3] for f in os.listdir(f"{ROOT}/data/plain/{setname}") if f.endswith(".gr"))
    if os.environ.get("ONLY"): names = [n for n in names if n in os.environ["ONLY"].split(",")]
    tmp = tempfile.mkdtemp(prefix=f"pace_{solver}_")
    path = f"{outdir}/{solver}_{setname}.tsv"
    with open(path, "w") as o, ThreadPoolExecutor(par) as ex:
        o.write("inst\tstatus\tsecs\twidth\tvalid\n")
        for name, st, s, w, v in ex.map(one, [(solver, setname, n, T, tmp) for n in names]):
            o.write(f"{name}\t{st}\t{s:.2f}\t{w}\t{v}\n"); o.flush()
    shutil.rmtree(tmp, ignore_errors=True)

main()
