#!/bin/bash
#SBATCH --job-name=itay-exact
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-dev
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/exactness/logs/A-%j.out
# Experiment A (exactness vs contraction cost), CPU only.
# Stage 1: cpus/2 Julia workers (-t2) over queueA.txt -> resA/td/<net>.td (exact 300 s + relabels, heuristics, HBT).
# Stage 2: costA.py on every .td (cpus processes) -> resA/cost/<net>.tsv.  Re-running resumes both stages.
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
E=/blue/fairbanksj/itaykadosh/gpu/exactness
source $E/common.sh
cd $E
OUT=$E/resA; mkdir -p $OUT/td $OUT/cost $OUT/logs
rm -rf $OUT/td/claims   # stale claims of a killed run
PY=$E/venv/bin/python
echo "start $(date) on $(hostname)"
NW=$((${SLURM_CPUS_PER_TASK:-32} / 2))
for k in $(seq 1 $NW); do
  julia --project=env -t2 --heap-size-hint=5G src/tdgen.jl A queueA.txt $OUT/td 300 > $OUT/logs/jl_$k.log 2>&1 &
done
wait
echo "julia stage done $(date): $(ls $OUT/td/*.td | wc -l) td files"
ls $OUT/td/*.td | xargs -P ${SLURM_CPUS_PER_TASK:-32} -I{} sh -c 'n=$(basename {} .td); grep -q "^#END" '$OUT'/cost/$n.tsv 2>/dev/null || '$PY' -I src/costA.py {} '$OUT'/cost/$n.tsv 10 > '$OUT'/logs/py_$n.log 2>&1'
echo "all done $(date)"
