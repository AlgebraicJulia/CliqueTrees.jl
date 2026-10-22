#!/bin/bash
#SBATCH --job-name=itay-csel
#SBATCH --account=fairbanksj --qos=fairbanksj-b
#SBATCH --partition=hpg-default,hpg-turin,hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/costsel/logs/gen-%j.out
# CPU TD generation (src/gen.jl): 32 single-thread Julia workers share queue.txt -> pool/<net>.pool
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
source /blue/fairbanksj/itaykadosh/gpu/costsel/hpg/common.sh
export JULIA_DEPOT_PATH=$CPU_DEPOT
cd $C; mkdir -p pool logs/gen tmp
# (stale claims of a killed run: rm -rf pool/claims by hand before resubmitting)
echo "start $(date) on $(hostname)"
for k in $(seq 1 ${SLURM_CPUS_PER_TASK:-32}); do
  julia --project=$E/env -t1 --heap-size-hint=4G src/gen.jl queue.txt K.tsv pool 1000 1000 150 > logs/gen/w$k.log 2>&1 &
done
wait
echo "done $(date): $(ls pool/*.pool | wc -l) pools"
