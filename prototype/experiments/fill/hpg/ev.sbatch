#!/bin/bash
#SBATCH --job-name=itay-ev
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=04:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/fill/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/fill/logs/%x-%j.err
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
F=/blue/fairbanksj/itaykadosh/gpu/fill
export CODE=$F/code DATA=$F/data RES=$F/results
mkdir -p $F/logs $RES $DATA/perms
source $F/instances.sh
JL="julia --project=/blue/fairbanksj/itaykadosh/gpu/fill/env -t1"
# evaluate GPU permutations (nnz(L), flops, width); run with --dependency=afterok:<gpu job>
cd $CODE
export JL CODE RES DATA
for i in $ALL; do echo $i; done | xargs -P 6 -I{} bash -c '[ -f $DATA/{}.bin ] || [[ {} == lat* ]] || exit 0; $JL $CODE/ev_gpu.jl {}'
echo DONE
