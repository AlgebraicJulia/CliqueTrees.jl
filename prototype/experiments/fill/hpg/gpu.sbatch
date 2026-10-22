#!/bin/bash
#SBATCH --job-name=itay-gpu
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --dependency=singleton
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --time=03:00:00
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
nvidia-smi --query-gpu=name,memory.total --format=csv
cd $CODE
# only instances whose data exists (dl job skips failures)
L=""; for i in $ALL; do if [ -f $DATA/$i.bin ] || [[ $i == lat* ]]; then L="$L $i"; fi; done
$JL gpu_fill.jl $L
echo DONE
