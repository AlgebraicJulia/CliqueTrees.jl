# gpusep: GPU septw backend for CliqueTrees (HBT / PIDBT / RTW)

Worktree: ~/Projects/GATAS/CliqueTrees.jl-gsep, branch `itay/gpu-septw` (base `itay/mt-hbt` @ 5cfa336). Not pushed.
Envs: `env/` (CUDA 6.0, Julia 1.12, CliqueTrees dev'ed from gsep; copied from the gpuord scratch env), `testenv/` (the test deps, gsep).
Run: `julia --project=env ...`; tests: `julia --project=testenv -t N runtests4.jl` (septw, sepenum, sepdp, rtw), GPU test: `julia --project=env -t N -e 'using Test; include(".../test/gpu.jl")'`.

## Design

Core (no CUDA dependency), `src/bt/backend.jl`:
- `abstract type SeptwBackend`; `CPUSeptw()` (default) and `GPUSeptw(; engine = :septw | :pid, budget = NaN, delta = 2.0, nmin = 0)`.
- `CliqueTrees.septwbackend!(b)` sets the global backend `SEPTW[]`, `septwbackend()` reads it, `septwwarmup!(b; nmax)` compiles device code and measures the GPU round cost.
- Every round of every exact search goes through `bt_round!(esc, w, g, k, dl, nt) -> (perm, next, status)` with `esc = bt_esc(SEPTW[], w, g, dl)` made once per search. Callers: `septw_run` (so `septw`, `PIDBT`, `HBTExact` = HBT's `hbt_exa`, RTW's `rtw_fin`/`hbt_safe`) and `rtw_probe_run` (RTW batch probes). One switch covers HBT, PIDBT and RTW.
- A round returns the next width itself (CPU: `min(mink, k + wmin)` as before, PID: exact nextK). CPU behaviour unchanged (tests pass, see below).

Extension `ext/CUDAExt/` (weakdep CUDA = "6"):
- `bits.jl` (GPUBits: UInt64 / Bits{W}), `sepenum.jl` (SepEnumGPU = gpuord EnumW), `sepdp.jl` (SepDPGPU = gpuord DPW), `pid/` (PIDGPU = gpupid @e6518cb src, wrapped in a module, ENV knobs removed), `CUDAExt.jl` (policy).
- Policy (`bt_round!(::Esc, ...)`):
  1. CPU rounds while the atom's CPU budget lasts. Budget = `delta` x cost of one GPU round on a 30-vertex grid, measured on the device (`cost!`); ski-rental argument: an atom the CPU finishes within the budget never pays for the GPU, one it does not pays at most delta x GPU fixed cost extra.
  2. Work estimate: a round is not started on the CPU if the growth of the last two CPU rounds says it would end after the budget (same rule as rtw_probe) -> escalate at once.
  3. A round whose CPU sepdp ran out of budget continues with GPU sepdp on the CPU's separator list (bit-identical, so mixing is exact). All later rounds of the atom stay on the GPU.
  4. GPU is a shared resource: one search at a time (ReentrantLock, trylock). Busy -> that round on the CPU with the full deadline.
  5. Failures never crash: GPU OOM / ErrorException ("bag of more than 128 vertices", single-pair overflow) / sepenum-sepdp stack overflow -> the round reruns on CPU, the atom stays on CPU (`esc.off`). Any other device error marks the GPU broken for the process (`BROKEN`), warn once.
  6. Not escalated: atoms with n > 512 for engine :septw (bitset limit), n < nmin, no functional device.
- Engines: `:septw` = GPU sepenum + sepdp, bit-identical to CPU (same separators/order, blocks, caps, perm, k, status). `:pid` = one GPU PID-BT decision per round (Tamaki ESA 2017) at K = k: YES -> ordering (width rechecked on CPU with treewidth), NO -> exact nextK. Same widths/status where both finish (lo a lower bound); orderings differ; with a target lo > tw only "found, width <= lo" is comparable.
- Stats: `EXT.STATS` (`EXT = Base.get_extension(CliqueTrees, :CUDAExt)`): cpu, escalated, gpufound/gpuabove/gputimeout, gpufailed, busy, tcpu, tgpu. `EXT.resetstats!()`.
- Launch config: from SM count x occupancy (`active_blocks`, `launch_configuration`) and free/total memory; no per-card constants. Remaining fixed knobs inherited from the prototypes: threads = 64 per block, maxresident = 2^15 persistent threads, stack words per thread, PID maxbuf = 6% of total memory. None aimed at a card, but they are constants (open).

## Commits (itay/gpu-septw, not pushed)

- a9bbf7d Run the rounds of septw and of the RTW probes through a backend (core: backend.jl, septw_run/rtw_probe_run call bt_round!)
- 45a8e03 Add the GPU rounds of the exact searches as an extension of CUDA.jl (ext/CUDAExt, Project weakdep CUDA="6", test/gpu.jl, runtests)
- 3a77080 Search downward from a greedy ordering in the PID rounds, and stop the GPU sepdp at the deadline
- f6194f0 Set the abort flag of the GPU sepenum and sepdp with a memset (bug inherited from the gpuord prototype: copyto! of the flag waited for the running kernels, so a GPU round ignored its deadline; reproduced on wmc_2023_152__2 at k=15, hung > 5 min with a 5 s deadline; now aborts at ~6.5 s incl. compile)
- 2252263 Measure the cost of a PID round on a new state (cost! re-used a cached PID state -> 6e-7 s)

## Verification (local, laptop RTX 5060 Laptop GPU 8 GB, 26 SMs)

- CPU tests (testenv, no CUDA loaded) septw/sepenum/sepdp/rtw: t1 46288/46288 pass (tests_t1.log, at 2252263; core unchanged since 3a77080 apart from docstrings); t8 46285/46285 pass (tests_t8.log; the count differs by 3 from t1 also without GPU code, from the time-limited rtw tests).
- GPU test test/gpu.jl (env with CUDA): t1 354/354, t8 354/354 (gputest_t1.log, gputest_t8.log, after f6194f0). Covers: GPU sepenum+sepdp == CPU septw exactly (perm, invp, k, status) on unit/int/float weights, n 12-70 (W=1 and W=2), lo = 0 / tw / tw+1 / cap below tw; GPU sepdp on CPU separator lists == CPU sepdp; PID engine: same width/status, valid orderings, target and cap cases; PIDBT, HBT, RTW with GPUSeptw(budget=0) give the CPU width and escalate; busy GPU -> CPU, same result; n > 512 not escalated; budget Inf never escalates. Skips cleanly without CUDA (HASCUDA / CUDA.functional()).
- Not covered by a test: OOM fallback (seen organically: PID OOM on 8 GB -> "gpu-failed>cpu", correct result, e.g. wmc_2023_152__2, mc_2020_arjun_102__1), deadline compliance (checked by hand, see f6194f0), Bits{4}/Bits{8} in the test (only in the atom runs).
- Earlier dbg_cases.log: n=90 sparse random graphs, all match (GPU septw 3-10x slower than CPU there).

## Results (local)

- table_atoms_local.md (atoms_local.tsv): 12 gpupid atoms n 63-173, 20 s each, CPU 8 threads. Widths and certificates agree everywhere both solve. Highlights: lm_batch_likelihood_sentence_5_14c__1 CPU septw 19.7 s, RTW 5.4 s, esc-pid 0.25 s; str_nw_mera_closed_120__1 CPU septw timeout, RTW 2.9 s, esc-pid 0.54 s; mc_2023_002__13 esc-pid 0.09 s vs CPU 0.19 s. esc-septw (GPU sepenum+sepdp) never beats 8-thread CPU on this laptop and times out on 7/12. esc-pid with the measured budget (escalate at once) loses on easy atoms (5-12 s vs 0.1-2 s) and OOMs on 3 (falls back). esc-pid:2 (2 s CPU budget) = best column: CPU speed on easy atoms + PID wins on the two hard ones; only rnd_mixed_11__6 is solved by RTW alone (PID OOM on 8 GB).
- table_hbt_local.md (hbt_local.tsv): HBT 60 s, 8 threads, equal wall time. gm_Promedas_33: cpu 38/37/37 vs esc-pid 37/36/36 (3 seeds, 1 bit better each); str_ctg_regular_03: mixed; str_ctg_lattice_04 (1 seed): escalated worse (40.09/41.26 vs 39.34). Exact-round share of HBT time (rounds, summed over tasks / wall x 8): 45-73% (CPU-only runs 53-72%) -- consistent with the profile's 56% average, not 78%.
- GPU contention in HBT: the GPU lock is held 53-60 s of 60 s, and 2-24k escalated rounds found it busy and ran on CPU. One GPU serialises 8 HBT tasks.

## HPG (submitted, not waited for)

- /blue/fairbanksj/itaykadosh/gpu/gpusep/: CliqueTrees.jl = rsync of the worktree at 2252263 (CliqueTrees.jl.commit), env = gpupid env with the CliqueTrees path changed and CUDAExt/CUDA added by hand to the [deps.CliqueTrees] extensions/weakdeps of the Manifest (no Pkg.resolve: the HPG registry is stale; without this the extension does not load), depot, common.sh, atoms -> gpupid/atoms. Precompiled on the login node; `Base.get_extension(CliqueTrees, :CUDAExt)` loads.
- Job 46702458 `hpg/atoms.sbatch cert_100_600.txt 120 b200` (B200, 16 cores): 81 atoms n 100-600, 120 s per run. GPU side esc-septw,esc-pid,esc-pid:2 -> results/atoms_b200_gpu.tsv; CPU side cpu,rtw (8 threads) -> results/atoms_b200_cpu.tsv; logs/atoms-46702458{,-gpu,-cpu}.{out,log}.
- Job 46702459 `hpg/hbt.sbatch hpg/showcase.txt 300 b200`: gm_Promedas_33, str_ctg_lattice_04, qc_maxcut_n18_r17_p100, wmc_2021_145, str_ctg_regular_03, rnd_mixed_12 (control), HBT 300 s, 8 threads: esc-septw,esc-pid vs cpu (seed 1) -> results/hbt_b200.tsv; logs/hbt-46702459*.
- Both put the GPU side and the CPU side in two processes on disjoint 8-core halves of the same node. Risk: the GPU side is CPU-first, so GPU utilisation can be low (HPG idle-GPU policy).
- Note: the ssh ControlMaster expired once mid-session (keyboard-interactive) and came back later.

## Open / next

- Escalation budget: delta x (cost of a GPU round on a 30-vertex grid) = 2.5 ms (septw) / 8.5 ms (pid) on the laptop: far too eager, because the GPU per-round cost on real atoms is not its fixed cost and the GPU is not faster per round on this card. A 2 s budget worked better locally. Needs a cost model per atom size (e.g. measured CPU-vs-GPU round time ratio by set width, learned online) instead of the fixed-latency ski-rental rule. Unverified on B200.
- GPU contention in HBT: batch the rounds of several HBT tasks into one GPU launch (gpuord README next step), or a queue instead of trylock.
- End-to-end cost of the septw engine is CPU prep + final ordering around the kernels (coordinator: B200 kernels 14.3 s vs 102.6 s 1-thread, but wall only 1.3x vs t1 and 0.9x vs t16, gpuord jobs 46598180/46598182, tn-work/harvest_oct19/SUMMARY.md). Next: move sepdp prep (blocks, relabel, trie) to the GPU or overlap it across atoms; report vs t16/t32.
- PID deadline: checked only between join chunks/rounds; overshoots seen up to +2.6 s locally (+16 s before the downward search). Add a device-side abort flag like sepenum/sepdp.
- Backlog (coordinator): GPU PID-BT dedup per join chunk instead of per round (bufOA/bufON/bufP pre-dedup buffers dominate the 42 B200 OOMs, gpupid/oom_headroom.md); results must stay bit-identical; add verbose per-round counts (I/O/P, bytes) to the certification runner. Not started.
- Fixed constants inherited from the prototypes (threads=64/block, maxresident=2^15, per-thread stack words, PID maxbuf=6% of total memory): generic, not card-specific, but not derived from the device either.
- Float weights: the PID decides with K = rtw_below(ub) (relative 1e-10, as RTW) and tol 0; widths agree with CPU to ~1e-14 relative, not bit-for-bit.
