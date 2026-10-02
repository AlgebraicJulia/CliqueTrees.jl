# ROME (PPoPP'26) APSP baseline: build and benchmark notes

Upstream: https://github.com/LyleLuo/ROME, commit `25beef3`. Submodules: METIS `7f8f9b0` (scivision fork, v5.1.0.3), Scotch `7fd71f5` (v7.0.10).
Machine: RTX 5060 Laptop GPU (sm_120, 26 SMs, 8 GB), Ryzen AI 7 350 (16 threads), 14 GB RAM, driver 595.84.

## Toolchain (no sudo, everything under `external/`)
- The pip wheels `nvidia-cuda-nvcc-cu12` (12.8 and 12.9) contain only `ptxas`, with no `nvcc` driver binary, so they were not usable. The venv was removed.
- I used a conda env at `external/cuda-env`, built from conda-forge only:
  `conda create -p external/cuda-env -c conda-forge --override-channels cuda-nvcc=12.9 cuda-cudart-dev=12.9 cuda-cccl=12.9 cuda-version=12.9 cmake make flex bison`
  This gives nvcc 12.9.86 (supports `compute_120`), CMake 4.4, flex and bison.
- The host compiler is the system `/usr/bin/gcc` / `g++` 13.3 with system libgomp (OpenMP). The conda env is not activated; only its `bin/` goes on `PATH`.
- The build produces native **sm_120** SASS (`CMAKE_CUDA_ARCHITECTURES=120`). No PTX-JIT fallback was needed.

## Build commands
```bash
EXT=/home/itaykadosh/Projects/GATAS/SemiringGPU/external
export PATH=$EXT/cuda-env/bin:$PATH
cd $EXT && git clone --recurse-submodules https://github.com/LyleLuo/ROME.git
cd ROME && git apply ../rome_gatas.patch            # all source changes (see below)
# METIS
cd modules/metis && mkdir -p build && cd build
CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake -DCMAKE_BUILD_TYPE=Release -DUSE_THREAD=Y -Dintsize=64 .. && cmake --build . -j8
cd ../../..
# Scotch
cd modules/scotch && mkdir -p build && cd build
CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake .. -DBUILD_FORTRAN=OFF -DBUILD_PTSCOTCH=OFF -DINTSIZE=64 -DSCOTCH_DETERMINISTIC=FULL && make -j8
cd ../../..
# ROME
mkdir -p build && cd build
CC=/usr/bin/gcc CXX=/usr/bin/g++ cmake -Wno-dev -DCMAKE_BUILD_TYPE=Release -DUSE_THREAD=Y \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCUDA_TOOLKIT_ROOT_DIR=$EXT/cuda-env \
  -DCUDA_TOOLKIT_INCLUDE=$EXT/cuda-env/targets/x86_64-linux/include \
  -DCUDA_CUDART_LIBRARY=$EXT/cuda-env/targets/x86_64-linux/lib/libcudart.so ..
cmake --build . -j8          # -> ROME/build/bin/rome
```
The extra `CUDA_TOOLKIT_INCLUDE` and `CUDA_CUDART_LIBRARY` flags are needed because the legacy `find_package(CUDA)` does not find the conda layout on its own.

## Source changes (full diff: `external/rome_gatas.patch`). ROME's algorithms and kernels are unchanged.
1. `CMakeLists.txt`: `set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)` changed to `120`.
2. `src/Matrix/MtxReader.cpp` (input fix, **important**): upstream ignores the weights in the file and assigns `std::rand() % 10 + 1` to every non-pattern entry. It now reads the file's weight (`std::stod(line_parsed[2])`). `ROME_RANDOM_WEIGHTS=1` restores the upstream behaviour.
3. `src/App/APSP.cu` (driver only), all controlled by environment variables:
   - `[GATAS] End-to-end` timer around: read mtx, build CSR, pinned host alloc, `APSP()`.
   - `ROME_DUMP_ROWS=<file>` (and `ROME_DUMP_NROWS`, default 3) writes the first rows of the result (original labeling). This runs after all timers.
   - `ROME_SKIP_CHECK=1` skips the CPU SuperFW reference check, which takes about 22 s on grid3d-25 and needs a third n² host buffer.
   - `ROME_LOWMEM=1` allocates only one n² pinned host buffer instead of two, and implies no check and no dump.
   - The return codes of `cudaMallocHost` are now checked.
4. `src/APSP/rome.cu`: one guard, `if (distance != nullptr)`, around the final host-side un-permutation loop. This loop runs after every timed GPU region and only matters for `ROME_LOWMEM`.

## Input format
ROME reads Matrix Market coordinate files. For every entry `(i,j,w)` with i≠j it adds **both** arcs i→j and j→i, so it only handles undirected graphs. It aborts on duplicate edges ("No multigraph supports"). Our files list each arc in both directions, so they would produce duplicates. All 4 graphs were checked: every weight is symmetric, an integer from 1 to 100, with no self loops.

`rome_mtx/convert.py` writes `rome_mtx/<g>.mtx`. It keeps only the i>j entries, uses the header `real symmetric`, and copies the weights verbatim. Weights are stored as float32. That is exact here because all distances are below 2^24.

## How to run
```bash
EXT=/home/itaykadosh/Projects/GATAS/SemiringGPU/external
ROME_SKIP_CHECK=1 OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 $EXT/ROME/build/bin/rome $EXT/rome_mtx/grid3d-25.mtx ignore
# add ROME_LOWMEM=1 for grid2d-180; benchmark loop: $EXT/run_rome_bench.sh  (GRAPHS=..., TAG=..., EXTRA_ENV=ROME_LOWMEM=1)
```
All engines are enabled, as in the upstream default: `BATCH_TILE_ENGINE = MERGE_ENGINE = true`, `NO_ASYNC` not defined. Logs are in `external/logs/`.

## What ROME's timers cover
- **tree build**: Scotch nested-dissection ordering, the supernode elimination tree, and the permuted CSR (CPU).
- **alloc/init**: `cudaMallocPitch` of the n² matrix, H2D of the CSR (small), and GPU kernels that fill the matrix with ∞, 0 on the diagonal, and the edge weights. There is **no n² H2D copy**.
- **computing** (ROME's headline GPU time): the whole level-by-level supernodal FW. It includes host-side metadata construction and the small async metadata H2D copies, and ends with `cudaDeviceSynchronize`. It excludes the ordering, the allocation, and the n² **D2H**.
- **D2H**: `cudaMemcpy2D` of the n² matrix into pinned host memory. After that, a single-threaded host un-permutation into the original labeling, which is not separately timed.
- **"Performance: X TFlops"** = Σ(m·n·k over all issued min-plus GEMM blocks) / computing time. It counts one (min,+) update as one op, not two. "ROME work" is that sum. "only X %" is that work as a share of n³.

## Results (median of 3 runs, CPU check off)
| graph | n | tree build (s) | alloc+init (s) | **GPU compute (s)** | D2H (s) | ROME "TFlops" (min-plus upd/s) | work % of n³ | E2E in-proc, stock (s) | E2E in-proc, LOWMEM (s) | process wall, LOWMEM (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| grid3d-25 | 15625 | 0.085 | 0.009 | **0.122** | 0.070 | 1.42 | 4.54 | 1.17 | 0.58 | 0.73 |
| grid2d-150 | 22500 | 0.085 | 0.014 | **0.082** | 0.146 | 1.40 | 1.00 | 1.77 | 0.75 | 0.97 |
| grid3d-30 | 27000 | 0.131 | 0.023 | **0.677** | 0.209 | 1.46 | 5.02 | 3.76 | 1.57 | 1.84 |
| grid2d-180 | 32400 | 0.039* | 0.025* | **0.209*** | 0.303* | 1.32* | 0.81 | not run (host RAM) | 1.54 | 1.89 |

The first five columns come from the stock runs. Compute and TFlops were identical within 2% in LOWMEM runs. *grid2d-180 values are from LOWMEM runs only.

- Stock E2E includes a second n² pinned allocation and the single-threaded un-permutation. LOWMEM E2E excludes both. Both include CUDA context creation, which is triggered by the first `cudaMallocHost`.
- The per-run compute spread was under 2%.
- GPU memory: the n² float matrix (pitched) plus small metadata. For grid2d-180 that is about 4.2 GB, which fits in 8 GB.
- Host memory: the stock driver needs 2·n²·4 B of pinned RAM (8.4 GB for n=32400) while only about 6–7 GB was free. So grid2d-180 was run only with `ROME_LOWMEM=1`. The GPU computation is unaffected.

## Correctness
- grid3d-25, stock mode with the CPU check enabled: ROME printed "Result is correct!" against its CPU SuperFW reference, which uses METIS ordering. The delaunay_n13 sample also passed.
- `external/rome_rows_grid3d-25.txt` holds 3 lines × 15625 integers: the distance rows for 1-based sources 1, 2 and 3, in the original labeling. They were checked against an independent Python Dijkstra on the original `data/mtx/grid3d-25.mtx`: **0 mismatches**.
- Command: `ROME_DUMP_ROWS=$EXT/rome_rows_grid3d-25.txt OMP_NUM_THREADS=16 $EXT/ROME/build/bin/rome $EXT/rome_mtx/grid3d-25.mtx ignore`

## Caveats
- The GPU is shared. Each batch was started only after `nvidia-smi` showed no other compute process. An unrelated Julia test briefly used the GPU before the first runs; I waited for it to exit.
- ROME's "theoretical TFlops" line assumes 64 ops/clk/SM at the reported clock (1560 MHz). Treat its "% achieved" as ROME's own metric.
