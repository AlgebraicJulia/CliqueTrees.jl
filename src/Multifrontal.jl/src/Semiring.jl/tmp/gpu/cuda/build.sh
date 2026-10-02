#!/usr/bin/env bash
# Build the CUDA C++ backend: cuda/build/libsemiring_cuda.so
#
#   cuda/build.sh                      # local RTX 5060 Laptop (sm_120)
#   ARCH="100 120" cuda/build.sh       # HiPerGator: B200 (sm_100) + RTX PRO 6000 (sm_120)
#   NVCC=/path/to/nvcc cuda/build.sh
#   PTX=1 cuda/build.sh                # also embed PTX (driver JIT; used to compare ptxas versions)
#   PTXONLY=1 BUILD=build-ptx cuda/build.sh   # PTX only: the driver's own (newer) ptxas compiles it at load time
#   BUILD=dir                          # output directory (default cuda/build)
#
# No --use_fast_math: it would replace 1/x by an approximation (sstar of PlusProd)
# and flush denormals. --fmad=false: every fma is explicit (PlusProd muladd), as in
# Julia, where a*b + c is never contracted unless written as muladd.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
NVCC=${NVCC:-$HERE/../external/cuda-env/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || { echo "nvcc not found (set NVCC=...)" >&2; exit 1; }
ARCH=${ARCH:-120}
GENCODE=""
for a in $ARCH; do
    if [ "${PTXONLY:-0}" = 1 ]; then GENCODE="$GENCODE -gencode arch=compute_$a,code=compute_$a"; continue; fi
    GENCODE="$GENCODE -gencode arch=compute_$a,code=sm_$a"
    if [ "${PTX:-0}" = 1 ]; then GENCODE="$GENCODE -gencode arch=compute_$a,code=compute_$a"; fi
done
BUILD=${BUILD:-build}
mkdir -p "$BUILD"
FLAGS="-O3 -std=c++17 --fmad=false -Xcompiler -fPIC -cudart static $GENCODE ${EXTRA:-}"
"$NVCC" $FLAGS -c semiring_gemm.cu -o $BUILD/semiring_gemm.o
"$NVCC" $FLAGS -c semiring_solve.cu -o $BUILD/semiring_solve.o
"$NVCC" $FLAGS -shared $BUILD/semiring_gemm.o $BUILD/semiring_solve.o -o $BUILD/libsemiring_cuda.so
echo "built $(cd "$BUILD" && pwd)/libsemiring_cuda.so for sm_{$ARCH} with $("$NVCC" --version | tail -1)"
