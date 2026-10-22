# Baselines for the TN contraction-ordering paper (Oct 20, 2026)

Everything runs on HPG under `/blue/fairbanksj/itaykadosh/gpu/baselines/tn/` (code: `code/` = `cutn/`, `code_td/` = `td/`). Related-work text: `../paper/RELATED_TD_TN.md`.

## 1. cuTensorNet path finder (`cutn/`, job name itay-cutn, one B200)
- venv `venv_cutn` (python 3.12): cuquantum-python-cu12 26.9.0, cutensornet-cu12 2.14.0, cupy-cuda12x 14.2 (+ctk wheels). Driver 580.178.04 / CUDA 13.2: the cu12 wheels work.
- `cutn_worker.py`: cuTensorNet has NO time budget; effort = `OptimizerOptions(samples=S, threads=K)`. The worker runs the ladder S=1,2,4,... (x4 when fast) until a run exceeds maxT, plus the DEFAULT `OptimizerOptions()` row. Budget T of the analysis = best row with search_s <= T (harvest.py).
  Network is built with random float32 operands only to satisfy the API; only the path matters. Scoring: path + sliced modes rebuilt in cotengra 0.8.2 -> logC (log10 multiply-adds over all slices, same units as einsumbench/frontier), width, maxop, nslices.
- Unsliced: `SlicerOptions(disable_slicing=True)`. If the unsliced workspace exceeds the device memory cuTensorNet raises MemoryLimitExceeded; the row is then retried with default slicing and flagged `fallback_sliced=1` in `extra`.
- Width cap W (frontier amp 27/30, marg6/open6huang 29, marg21 30, syc70/67 30): `memory_limit = 2^W*4 B*memf`, memf in {1,2,4} (its workspace model counts several tensors); the achieved width is recomputed and `cap_met` = width<=W.
- Phases run concurrently in one job: 8 workers x 1 thread on the 163 einsum networks (T up to 100 s) and 3 workers x 8 threads on the frontier networks (T up to 100, 1000 s for amp m18/m20, marg6 m20, marg21, syc70, syc67). Each runs `--exec 1` (executes small contractions on the GPU, `exec_s`) and worker e0 has a light cupy matmul keepalive thread (HPG kills idle-GPU jobs).
- Submit: `sbatch code/cutn.sbatch /blue/fairbanksj/itaykadosh/gpu/baselines/tn/full_cutn` (add `smoke` as 2nd arg for a 2-minute check). Results `full_cutn/res/cutn_t{1,8}_*.tsv`; logs `full_cutn/logs`.

## 2. TensorOrder / QTensor ordering step (`td/`, job name itay-tdb, hpg-milan, 60 pinned single-core workers)
- Built from the upstream repos (cloned to `src/`): FlowCutter PACE'17 (g++ with `-include string`), Tamaki TCS-Meiji heuristic (javac of /apps/java/jdk-14; module java/17 is broken for non-login shells), htd 1.2 (cmake). `JAVA_BIN=/apps/java/jdk-14/bin/java`.
- Methods `to-flow, to-tamaki, to-htd` (TensorOrder line-graph + TD solver + their tree extraction), `qt-tamaki, qt-greedy` (QTensor). Details and deviations in the `td_worker.py` docstring. NOT run: TensorOrder's own binary (Cython/py3.6 stack), htd/FlowCutter portfolio (`line-portfolio3`), FT (factor) methods.
- Budgets 1/10/100 s (and 1000 s on frontier nets), seed 1, all 163 einsum + 17 frontier networks. Output rows use the einsumbench column layout: `full_td/res/td_c1_w*.tsv`.
- Submit: `sbatch code_td/td.sbatch /blue/fairbanksj/itaykadosh/gpu/baselines/tn/full_td`.
- WMC: the einsum corpus contains the model-counting-competition networks (mc_* 46, wmc_* 13 of the 163; TensorOrder's target domain); they are the WMC subset. TensorOrder's repo ships only cubic-vertex-cover CNFs; no extra download was made. `harvest.py` writes `wmc_td.tsv` and, once einsumbench run1 (jobs 46743040/41/42) is harvested, `wmc_compare.tsv` (ours/cotengra vs TD baselines at equal budget).

## Harvest
    cd /home/itaykadosh/Projects/GATAS/tn-work/baselines
    python harvest.py --fetch --local harvest        # needs pandas (tncorpus venv or any); partial results are fine
Check job state: `ssh hpg 'squeue -u itaykadosh -n itay-cutn,itay-tdb'`. After a kill, leftover queue lines are in `full_*/queue_*.<jobid>`; copy back to `tasks_*.txt` and resubmit.
Caveats: cuTensorNet single run/seed 1; time of the hyper-sampler is not linear in samples (ladder granularity x2/x4); the GPU job's CPU part runs 11 concurrent processes on 32 cores.
