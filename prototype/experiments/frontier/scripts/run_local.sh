#!/bin/bash
# Laptop runner: same tasks/format as frontier.sbatch, one pinned single-threaded process per core.
# usage: run_local.sh <tasks file> <outdir> <core list, e.g. "11 12 13 14 15">
set -u
SP=/path/to/tn-work
F=$SP/frontier; PY=$SP/tncorpus/venv/bin/python; JL=$HOME/.juliaup/bin/julia; ENV=$SP/mtenv
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
LIST=$1; OUT=$2; CORES=($3)
mkdir -p $OUT/logs $OUT/ord $OUT/res
Q=$OUT/queue.$$; cp $LIST $Q
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
      taskset -c $core $JL --project=$ENV -t 1 --gcthreads=1 $F/scripts/orders.jl $name $meth $T $OUT/ord $F $seed > $OUT/logs/$tag.jl.log 2>&1
      local f=$OUT/ord/$tag.ord
      [ -f $f ] && taskset -c $core $PY $F/scripts/eval_orders.py $f $OUT/res $tg > $OUT/logs/$tag.ev.log 2>&1
    else
      taskset -c $core $PY $F/scripts/baselines.py $name $meth $T $seed $OUT/res $tg > $OUT/logs/$tag.py.log 2>&1
    fi
    echo "$(date +%T) core $core done $tag (exit $?)"
  done
}
for c in "${CORES[@]}"; do worker $c & done
wait
echo "all done $(date)"
