#!/bin/bash
#SBATCH --job-name=itay-gnd
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200gb
#SBATCH --time=03:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/rounds/nd/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/rounds/nd/logs/%x-%j.err
# "Closing the AMD gap" (ND_NOTES.md): CPU+GPU hybrid refinement, device leaves, top-k variant, policies.
# ndbench.jl over NAMES x CFGS with JT Julia threads (the hybrid's CPU FM uses them); code in nd/code3,
# GRounds/GSmall (with grounds_batch_dev) in nd/grd3, env nd/henv (= rounds/env + AMD.jl).
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
V=$R/nd
mkdir -p $V/logs $V/results3
cd $V/code3
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
export GRDIR=${GRDIR:-$V/grd3}
export OUT=${OUT:-$V/results3/hyb_b200.tsv}
JT=${JT:-32}
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"
echo "JT=$JT GRDIR=$GRDIR NAMES=$NAMES CFGS=$CFGS OUT=$OUT"
julia --project=$V/henv -t$JT ndbench.jl
echo DONE
