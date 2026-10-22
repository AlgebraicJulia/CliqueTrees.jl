#!/bin/bash
#SBATCH --job-name=itay-csel
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/costsel/logs/gpu-%j.out
# GPU scoring of every pool TD, width-capped GPU mass search (10 s), GPU polish of 3 selected trees (10 s each).
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
source /blue/fairbanksj/itaykadosh/gpu/costsel/hpg/common.sh
export JULIA_DEPOT_PATH=$GPU_DEPOT
cd $C; mkdir -p gpu
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "start $(date)"
julia --project=genv src/score.jl queue.txt pool gpu 10 10
echo "done $(date): $(ls gpu/*.gpu | wc -l)"
