#!/bin/bash
#SBATCH --job-name=itay-gpu
#SBATCH --dependency=singleton
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/gpuanneal/logs/gpu-%j.out
# usage: sbatch --partition=hpg-b200 --gpus=b200:1 --time=HH:MM:SS gpu.sbatch <tasks file> <result tag>
# One GPU, one host thread (the annealer keeps the GPU busy between short host steps).
set -u
H=/blue/fairbanksj/itaykadosh/gpu/gpuanneal
source /blue/fairbanksj/itaykadosh/gpu/env.sh
cd $H
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
julia --project=env -t 1 scripts/batch.jl $1 results/$2.tsv results/$2_trees
echo "done $(date)"
