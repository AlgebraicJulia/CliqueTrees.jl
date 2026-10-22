#!/bin/bash
#SBATCH --job-name=itay-csel
#SBATCH --account=fairbanksj --qos=fairbanksj-b
#SBATCH --partition=hpg-default,hpg-turin,hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/costsel/logs/prep-%j.out
# instantiate + precompile the GPU env (genv) on a CPU node
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
source /blue/fairbanksj/itaykadosh/gpu/costsel/hpg/common.sh
export JULIA_DEPOT_PATH=$GPU_DEPOT
cd $C
julia --project=genv -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA, Random, Printf; println("ok")'
