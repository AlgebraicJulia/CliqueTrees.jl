#!/bin/bash
#SBATCH --job-name=itay-gnd
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --time=02:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/rounds/nd/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/rounds/nd/logs/%x-%j.err
# GPU ND + batched GPU MF1 leaves (rounds/ND_NOTES.md) on one B200: ndbench.jl over NAMES x CFGS.
# GRDIR selects the GRounds/GSmall version used for the leaves.
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
V=$R/nd
mkdir -p $V/logs $V/results
cd $V/code
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
export GRDIR=${GRDIR:-$R/small/code2}
export OUT=${OUT:-$V/results/gnd_b200.tsv}
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
echo "GRDIR=$GRDIR NAMES=$NAMES CFGS=$CFGS"
julia --project=$R/env -t1 ndbench.jl
echo DONE
