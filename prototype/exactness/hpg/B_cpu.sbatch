#!/bin/bash
#SBATCH --job-name=itay-exact
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-dev
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/exactness/logs/B-%j.out
# Experiment B, CPU side: orders (Julia: ours10/ours60 hybridcost, MF1, HBT 1 s, exact 300 s) and trees
# (costB.py: cotengra re-costing, subtree reconfiguration, slicing to 2^30, cotengra hyper 10/60 s).
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
E=/blue/fairbanksj/itaykadosh/gpu/exactness
source $E/common.sh
cd $E
OUT=$E/resB; mkdir -p $OUT/td $OUT/trees $OUT/logs
rm -rf $OUT/td/claims
PY=$E/venv/bin/python
echo "start $(date) on $(hostname)"
NW=$((${SLURM_CPUS_PER_TASK:-32} / 2))
for k in $(seq 1 $NW); do
  julia --project=env -t2 --heap-size-hint=5G src/tdgen.jl B queueB.txt $OUT/td 300 > $OUT/logs/jl_$k.log 2>&1 &
done
wait
echo "julia stage done $(date): $(ls $OUT/td/*.td | wc -l) td files"
ls $OUT/td/*.td | xargs -P ${SLURM_CPUS_PER_TASK:-32} -I{} sh -c 'n=$(basename {} .td); [ -f '$OUT'/trees/$n.pkl ] || '$PY' -I src/costB.py {} '$OUT'/trees 30 > '$OUT'/logs/py_$n.log 2>&1'
echo "all done $(date)"
