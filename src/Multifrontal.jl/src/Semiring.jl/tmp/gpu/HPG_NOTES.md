# SemiringGPU on UF HiPerGator (HPG)

Set up and validated on 2026-10-01. Everything is under `/blue/fairbanksj/itaykadosh/gpu/`. Nothing is written to `/home/itaykadosh`, which is 100% full; a 20 GB core dump there was left untouched.

## Layout on HPG
```
/blue/fairbanksj/itaykadosh/gpu/
  env.sh                 # source in every shell/job (see below)
  SemiringGPU/           # rsync of local tree (no .git, external/cuda-env, ROME build dirs, *.log)
  CliqueTrees.jl/        # rsync of working tree, branch gpu-semiring incl. uncommitted tropical.jl fix
  julia_depot/           # JULIA_DEPOT_PATH (packages, artifacts ~2.5 GB, compiled)
  jobs/validate.sbatch   # validation job (tests + bench_gemm + ROME)
  jobs/build_rome.sh     # ROME/METIS/Scotch build (login node, compile only)
  logs/                  # all SLURM --output/--error and build logs
  tmp/ cache/ conda_pkgs/
```
Sync from the laptop:
```bash
cd ~/Projects/GATAS
rsync -a --exclude=.git --exclude=external/cuda-env --exclude=external/ROME/build \
  --exclude='external/ROME/modules/*/build' --exclude='*.log' \
  SemiringGPU/ hpg:/blue/fairbanksj/itaykadosh/gpu/SemiringGPU/
rsync -a --exclude=.git --exclude='*.log' CliqueTrees.jl/ hpg:/blue/fairbanksj/itaykadosh/gpu/CliqueTrees.jl/
```
**Warning:** also exclude `Project.toml`: `CUDA.set_runtime_version!` stores `CUDA_Runtime_jll` in its `[extras]`, and overwriting it from the laptop made every job fail with "CUDA runtime not found" (2026-10-01; fixed by re-running `Pkg.develop` + `Pkg.instantiate` + `CUDA.set_runtime_version!(v"13.0")` + `Pkg.precompile()` on the login node). Likewise, a plain rsync of `Manifest.toml` puts back the laptop's absolute CliqueTrees path, and a plain rsync of `external/ROME/CMakeLists.txt` undoes the HPG arch change. Exclude `Manifest.toml`, `LocalPreferences.toml` and `external/ROME/CMakeLists.txt`, or re-run the `Pkg.develop` step below afterwards.

## Environment (`/blue/fairbanksj/itaykadosh/gpu/env.sh`)
```bash
G=/blue/fairbanksj/itaykadosh/gpu
export JULIA_DEPOT_PATH=$G/julia_depot
export JULIA_HISTORY=/dev/null
export TMPDIR=$G/tmp
export XDG_CACHE_HOME=$G/cache
export CONDA_PKGS_DIRS=$G/conda_pkgs
export JULIA_PKG_PRECOMPILE_AUTO=0
ulimit -c 0   # no core dumps (the default limit is unlimited, which is how home got a 20 GB core)
module load julia/1.12.6 >/dev/null 2>&1
export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)"
```
Environment fixes (no algorithm code was changed):
1. **CliqueTrees path.** The remote project ran `Pkg.develop(path="../CliqueTrees.jl"); Pkg.instantiate(); Pkg.precompile()`. The remote Manifest now holds the relative path `../CliqueTrees.jl`. The local Manifest is unchanged.
2. **CUDA runtime preference.** The login nodes have no NVIDIA driver, so CUDA.jl could not choose a runtime ("CUDA runtime not found"). I ran `CUDA.set_runtime_version!(v"13.0")` on the login node. It wrote `SemiringGPU/LocalPreferences.toml` (`[CUDA_Runtime_jll] version = "13.0"`) on HPG only. Loading CUDA on the login node afterwards downloaded the runtime artifacts into the depot, so jobs need no internet. The node driver is 580.178.04 (CUDA 13.0). CUDA.jl reports "driver 580.178.4 for 13.3", which means it uses the CUDA_Driver_jll forward-compat driver.
3. **Portable precompile (`JULIA_CPU_TARGET`).** Julia normally compiles for `-C native`. The login node (EPYC 75F3, Zen3), the RTX6000 node (EPYC 9555, Zen5) and the B200 node (Xeon 8570) all differ, so each job recompiled every package into the shared depot. When two jobs did this at the same time, they deadlocked on each other's `.pidfile` locks across nodes. With the multi-target setting above, one precompile on the login node serves all node types. Also, **do not run two jobs that may precompile at the same time**. After changing packages, precompile on the login node first.

## GPU resources (`sinfo`, 2026-10-01)
**There are no A100 (or H100) nodes on HPG anymore.** The old `gpu` partition is gone.

| partition | GPU (per node) | node CPU | cores / RAM per node | sbatch |
|---|---|---|---|---|
| `hpg-b200` | 8× NVIDIA B200, sm_100, 179 GiB HBM | 2× Intel Xeon Platinum 8570 (Emerald Rapids, 56 cores each, up to 4.0 GHz, AVX-512/AMX) | 112 / 2 TB | `--partition=hpg-b200 --gpus=b200:1` |
| `hpg-rtx6000` | 8× RTX PRO 6000 Blackwell Server Edition, sm_120, 95 GiB, 188 SMs | 2× AMD EPYC 9555 (Zen5 Turin, 64 cores each) | 128 / 1.5 TB | `--partition=hpg-rtx6000 --gpus=rtx_pro_6000:1` |
| `hpg-turin` | 3× NVIDIA L4, sm_89 | AMD Turin (hpg4) | 96 / 771 GB | `--partition=hpg-turin --gpus=l4:1` (not tested) |
| `hwgui` | 3× L4 | AMD Turin | 96 | interactive GUI partition, not for batch work |

QOS: `fairbanksj` has group limits of cpu=262, gpu=50, mem=2 TB, and a maximum wall time of 31 days. The burst QOS `fairbanksj-b` has **gres/gpu=0**, so GPU jobs must use `--qos=fairbanksj`.

### CPU-only nodes (for an all-cores CPU baseline)
| partition | CPU | cores | RAM |
|---|---|---|---|
| `hpg-default` | 2× AMD EPYC 7702 (Zen2 Rome, 64 cores each) | 128 | 1 TB |
| `bigmem` | 2× AMD EPYC 7702 (2 nodes); 1 node is Intel Skylake with 96 cores | 128 | 4 TB |
| `hpg-milan` / `hpg-dev` | AMD EPYC 75F3 (Zen3 Milan) | 64 | 512 GB |

An all-cores run would be `--partition=hpg-default --cpus-per-task=128 --exclusive --mem=0` with `julia -t 128`. It fits the group's 262-CPU limit, but it is larger than the "16 CPUs or fewer" limit used for this validation. The RTX6000 nodes (2× EPYC 9555, 128 Zen5 cores) are the most modern CPUs on HPG, but they belong to a GPU partition.

## sbatch template (worked on B200 and RTX PRO 6000)
`/blue/fairbanksj/itaykadosh/gpu/jobs/validate.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=sgpu-validate
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/logs/%x-%j.err
#SBATCH --chdir=/blue/fairbanksj/itaykadosh/gpu/SemiringGPU
source /blue/fairbanksj/itaykadosh/gpu/env.sh
export JULIA_NUM_THREADS=16 OMP_NUM_THREADS=16
julia --project=. -t 16 test/test_solve.jl   # ... etc.
```
Submit:
```bash
cd /blue/fairbanksj/itaykadosh/gpu/jobs
j=$(sbatch --parsable --partition=hpg-b200 --gpus=b200:1 validate.sbatch)
sbatch --dependency=afterany:$j --partition=hpg-rtx6000 --gpus=rtx_pro_6000:1 validate.sbatch
```
Both jobs started within about 1 minute of submission.

## Validation results
| job | node / GPU | test_solve | test_factor | test_closure | bench_gemm correctness | ROME grid3d-25 |
|---|---|---|---|---|---|---|
| 44355147 | c0909a-s25, B200 | all ok | all ok | all ok | all 6 ok | ok |
| 44355149 | c0806a-s21, RTX PRO 6000 | all ok | all ok | all ok | all 6 ok | ok |

There were no FAIL lines and no errors. `.err` was empty for both jobs. Logs: `logs/sgpu-validate-44355147.out` (B200) and `logs/sgpu-validate-44355149.out` (RTX PRO 6000). CUDA.jl 6.1 (CUDACore 6.1.1) used runtime 13.0 from artifacts: cuBLAS 13.1.0, cuSPARSE 12.6.3, LLVM 18.1.7.

bench_gemm with 16 CPU threads, square n=4096, in G multiply-adds/s:
| GPU | MinPlus32 | MaxMin32 | MinPlusI32 | MinPlus64 | PlusProd32 | cuBLAS32 |
|---|---|---|---|---|---|---|
| B200 | 6350 | 3846 | 1704 | 2790 | 9227 | 31417 |
| RTX PRO 6000 | 9649 | 7853 | 3595 | 321 | 12772 | 37730 |

The CPU column is NaN for n=4096 because the benchmark skips CPU timing when n > 2048.

## ROME
- Built on the login node (compilation only) with `jobs/build_rome.sh` (`ROME_CUDA=12.9.1`): `module load cuda/12.9.1 cmake/3.30.5`, system gcc/g++ 11.5, and the same METIS, Scotch and ROME cmake flags as `external/ROME_NOTES.md`. It produces a fat binary with **sm_89, sm_100 and sm_120** cubins (`cuobjdump --list-elf`). cudart is linked statically, so no module is needed at run time. Binary: `SemiringGPU/external/ROME/build/bin/rome`. Build log: `logs/build_rome_cuda12.9.txt`.
- The architecture flag is now configurable in the HPG copy of `external/ROME/CMakeLists.txt`. `set(CMAKE_CUDA_ARCHITECTURES 120)` became `if(NOT ROME_ARCHS) set(ROME_ARCHS 120) endif() set(CMAKE_CUDA_ARCHITECTURES ${ROME_ARCHS})`, and the build passes `-DROME_ARCHS="89;100;120"`. The original is saved as `CMakeLists.txt.orig`.
- **cuda/13.0.2 does not work.** CUDA 13 removed `cudaDeviceProp::clockRate`, which ROME's `src/APSP/rome.cu:799-800` uses (log: `logs/build_rome.txt`). I did not patch the source and used 12.9.1, which supports sm_100 and sm_120.
- Run: `ROME_SKIP_CHECK=1 OMP_NUM_THREADS=16 $BIN external/rome_mtx/grid3d-25.mtx ignore`

| GPU | tree build (s) | alloc+init (s) | **GPU compute (s)** | D2H (s) | ROME TFlops (min-plus upd/s) | E2E in-proc (s) |
|---|---|---|---|---|---|---|
| B200 | 0.051 | 0.036 | **0.0320** | 0.018 | 5.40 | 1.89 |
| RTX PRO 6000 | 0.024 | 0.005 | **0.0224** | 0.017 | 7.73 | 0.76 |
| (laptop RTX 5060, for reference) | 0.085 | 0.009 | 0.122 | 0.070 | 1.42 | 1.17 |

These are single runs, not medians.
