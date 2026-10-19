#!/bin/bash
#SBATCH --job-name=itay-ct
#SBATCH --dependency=singleton
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=400G
#SBATCH --time=04:30:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/frontier/logs/frontier-%j.out
# usage: sbatch frontier.sbatch <tasks file> <outdir>
# One single-threaded process per core, pulling tasks (longest first) from a shared queue.
set -u
F=/blue/fairbanksj/itaykadosh/gpu/frontier
source /blue/fairbanksj/itaykadosh/gpu/ct-mt/common.sh
module load python/3.12 >/dev/null 2>&1
ENV=/blue/fairbanksj/itaykadosh/gpu/ct-mt/envs/mt-65e72a3
export TNCORPUS_DIR=/blue/fairbanksj/itaykadosh/gpu/tncorpus FRONTIER_DIR=$F
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=$F/cache MPLCONFIGDIR=$F/cache/mpl
PY=$F/venv/bin/python
LIST=$1; OUT=$2
mkdir -p $OUT/logs $OUT/ord $OUT/res
Q=$OUT/queue.$SLURM_JOB_ID; cp $LIST $Q
NW=${NW:-$SLURM_CPUS_PER_TASK}
cores=($(taskset -cp $$ | awk -F': ' '{print $2}' | tr ',' '\n' | awk -F- '{if(NF==2){for(i=$1;i<=$2;i++)print i}else print $1}'))
echo "node $(hostname), ${#cores[@]} cores, $NW workers, $(wc -l < $Q) tasks, start $(date)"
worker() {
  local core=$1
  while true; do
    local line
    line=$(flock $Q.lock sh -c "head -n 1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && break
    set -- $line
    local kind=$2 name=$3 meth=$4 T=$5 seed=$6 tg=$7
    local tag=${name}__${meth}__T${T}__s${seed}
    echo "$(date +%T) core $core start $kind $tag"
    if [ $kind = J ]; then
      taskset -c $core julia --project=$ENV -t 1 --gcthreads=1 $F/scripts/orders.jl $name $meth $T $OUT/ord $F $seed > $OUT/logs/$tag.jl.log 2>&1
      local f=$OUT/ord/$tag.ord
      [ -f $f ] && taskset -c $core $PY $F/scripts/eval_orders.py $f $OUT/res $tg > $OUT/logs/$tag.ev.log 2>&1
    else
      taskset -c $core $PY $F/scripts/baselines.py $name $meth $T $seed $OUT/res $tg > $OUT/logs/$tag.py.log 2>&1
    fi
    echo "$(date +%T) core $core done $tag (exit $?)"
  done
}
for ((k = 0; k < NW; k++)); do worker ${cores[$k]} & done
wait
echo "all done $(date)"
