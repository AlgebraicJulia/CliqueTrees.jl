#!/bin/bash
#SBATCH --job-name=itay-gpucost
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/gpucost/logs/gpucost-%j.out
# NOT SUBMITTED (prototype). Method (b) of tn-work/gpucost on a B200: GPU mass scoring by exact C +
# batched GPU subtree reconfiguration + population ILS, at 1, 10 and 60 s per network.
# Setup: rsync tn-work/gpucost (src, scripts, env, data/ord, data/*.txt) and tn-work/tncorpus to $G,
# fix the two absolute paths in src/Host.jl (tncorpus load.jl) and scripts/tnio.py (SCR), then
#   julia --project=$G/env -e 'using Pkg; Pkg.instantiate()'   (in a CPU job; CliqueTrees is not needed here)
# The GPU is busy throughout (mass scoring and DP launches back to back); host work is the chunk building.
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/gpupid/common.sh   # JULIA_DEPOT_PATH
G=/blue/fairbanksj/itaykadosh/gpu/gpucost
mkdir -p $G/logs
nvidia-smi --query-gpu=name,memory.total --format=csv
cd $G
export TNCORPUS_LOAD=/blue/fairbanksj/itaykadosh/gpu/tncorpus/load.jl TNCORPUS_DIR=/blue/fairbanksj/itaykadosh/gpu/tncorpus
julia --project=env scripts/run_gpu.jl data/nets30.txt data/ord results_hpg/gpu1 1 hbtT0.5s1
julia --project=env scripts/run_gpu.jl data/nets30.txt data/ord results_hpg/gpu10 10 hbtndT5.0s1
julia --project=env scripts/run_gpu.jl data/nets30.txt data/ord results_hpg/gpu60 60 hbtndT10.0s1
echo "done $(date)"
# afterwards, on a CPU node:  python -I scripts/verify.py results_hpg/verified.tsv results_hpg/gpu1 results_hpg/gpu10 results_hpg/gpu60
