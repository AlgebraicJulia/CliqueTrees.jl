#!/bin/bash
#SBATCH --job-name=itay-gnd
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --time=00:40:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/rounds/nd/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/rounds/nd/logs/%x-%j.err
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
cd $R/nd/code
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
julia --project=$R/env -t1 hpg/fmdiag2.jl
