#!/bin/bash
#SBATCH --job-name=itay-tdb
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/baselines/tn/logs/td-%j.out
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
# usage: sbatch td.sbatch <outdir> [smoke]   -- TensorOrder (line-Flow/Tamaki/htd) and QTensor (tamaki, greedy) ordering step, 60 pinned single-core workers
set -u
B=/blue/fairbanksj/itaykadosh/gpu/baselines/tn
G=/blue/fairbanksj/itaykadosh/gpu
PY=$G/frontier/venv/bin/python
export TNCORPUS_DIR=$G/tncorpus FRONTIER_DIR=$G/frontier CTGPLUGIN=$G/ctgplugin FTN_DIR=$G/frontier/scripts
export TO_SOLVERS=$B/src/TensorOrder/solvers JAVA_BIN=/apps/java/jdk-14/bin/java TMPDIR=$B/tmp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=$G/frontier/cache
OUT=${1:?outdir}; MODE=${2:-full}
mkdir -p $OUT/res $OUT/logs $TMPDIR
$PY $B/code_td/gen_queue.py $OUT
if [ $MODE = smoke ]; then
  grep -E "einsum (gm_1fas|mc_2021_074|wmc_2021_183|qc_qft_27)" $OUT/tasks_td.txt | awk '$4<=10' > $OUT/q_s
  for k in 0 1; do taskset -c $k $PY -I $B/code_td/td_worker.py $OUT/q_s $OUT --tag smoke$k; done 2>&1 | tail -60
  exit 0
fi
cp $OUT/tasks_td.txt $OUT/queue_td.$SLURM_JOB_ID
cores=($(taskset -cp $$ | awk -F': ' '{print $2}' | tr ',' '\n' | awk -F- '{if(NF==2){for(i=$1;i<=$2;i++)print i}else print $1}'))
for ((k = 0; k < 60 && k < ${#cores[@]}; k++)); do
  taskset -c ${cores[$k]} $PY -I $B/code_td/td_worker.py $OUT/queue_td.$SLURM_JOB_ID $OUT --tag w$k > $OUT/logs/td_w$k.log 2>&1 &
done
wait
echo done $(date)
