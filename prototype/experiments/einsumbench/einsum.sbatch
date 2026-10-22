#!/bin/bash
#SBATCH --job-name=itay-ein
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/einsumbench/logs/ein-%j.out
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
# Definitive einsum-benchmark run. usage: sbatch einsum.sbatch <outdir> <cfg: c1|c8> <kinds: both|ours|ctg>
#   163 networks x methods x budgets {0.1,1,10,100} s x 3 seeds (queues: python gen_tasks.py <outdir>).
#   cfg c1 = equal single core: every worker 1 core (ours Julia -t1; cotengra parallel=False); kinds=both, 36 ours + 28 ctg workers.
#   cfg c8 = equal 8 cores: every worker 8 cores (ours -t8, TreeSA ntrials=8; cotengra parallel=8); one kind per job, 8 workers.
# Submit as 3 jobs (all named itay-ein, no dependency): c1 both; c8 ours; c8 ctg.  Estimated wall: c1 ~2-3 h, c8 ours ~6.5 h,
# c8 ctg ~5 h (limit 12 h). Results are appended per task (res/<kind>_<cfg>_<tag>.tsv); queues are copied per job
# (queue_<kind>_<cfg>.<jobid>), so a killed job loses only its in-flight tasks; the leftover lines of its queue file can be
# re-queued (cp queue_.. tasks_..; sbatch again).
# PREP (once; see HPG.md): julia env ctgplugin/jlenv-hpg, frontier/venv (cotengra 0.8.2, kahypar, optuna), code + tncorpus synced.
set -u
B=/blue/fairbanksj/itaykadosh/gpu
source $B/ct-mt/common.sh
module load python/3.12 >/dev/null 2>&1
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
OUT=${1:?outdir}; CFG=${2:?cfg c1|c8}; KINDS=${3:?both|ours|ctg}
export TNCORPUS_DIR=$B/tncorpus CTGPLUGIN=$B/ctgplugin CTG_JLENV=$B/ctgplugin/jlenv-hpg CTG_JULIA=/apps/julia/1.12.6/bin/julia
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=$B/frontier/cache MPLCONFIGDIR=$B/frontier/cache/mpl
PY=$B/frontier/venv/bin/python
W=$B/einsumbench
if [ $CFG = c1 ]; then TO=1; CP=0; NO=36; NC=28; else TO=8; CP=8; NO=8; NC=8; fi
[ $KINDS = ours ] && NC=0
[ $KINDS = ctg ] && NO=0
mkdir -p $OUT/res $OUT/logs
cores=($(taskset -cp $$ | awk -F': ' '{print $2}' | tr ',' '\n' | awk -F- '{if(NF==2){for(i=$1;i<=$2;i++)print i}else print $1}'))
echo "node $(hostname), ${#cores[@]} cores, cfg $CFG kinds $KINDS: $NO ours workers x $TO threads, $NC ctg workers x $TO cores (parallel=$CP), start $(date)"
for kind in ours ctg; do [ -f $OUT/tasks_$kind.txt ] && cp $OUT/tasks_$kind.txt $OUT/queue_${kind}_$CFG.$SLURM_JOB_ID; done
pids=(); pos=0
for ((k = 0; k < NO; k++)); do
  cl=$(IFS=,; echo "${cores[*]:$pos:$TO}"); pos=$((pos + TO))
  taskset -c $cl $PY -I $W/worker.py $OUT/queue_ours_$CFG.$SLURM_JOB_ID $OUT ours --cfg $CFG --threads $TO --tag o$k > $OUT/logs/worker_${CFG}_o$k.log 2>&1 &
  pids+=($!)
done
for ((k = 0; k < NC; k++)); do
  cl=$(IFS=,; echo "${cores[*]:$pos:$TO}"); pos=$((pos + TO))
  taskset -c $cl $PY -I $W/worker.py $OUT/queue_ctg_$CFG.$SLURM_JOB_ID $OUT ctg --cfg $CFG --ctg-parallel $CP --tag c$k > $OUT/logs/worker_${CFG}_c$k.log 2>&1 &
  pids+=($!)
done
wait
echo "all done $(date)"
