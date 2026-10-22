#!/bin/bash
# Local smoke test: 5 networks x 2 budgets x 1 seed (4 Julia threads; the machine is shared).
# usage: run_local.sh <outdir>
set -u
HERE=$(cd $(dirname $0) && pwd)
PY=${PY:-python3}
OUT=$1; mkdir -p $OUT/res $OUT/logs
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
$PY $HERE/gen_tasks.py $OUT --nets gm_pedigree50,lm_batch_likelihood_brackets_3_16d,str_nw_mera_open_26,qc_qft_27,mc_2021_074 --budgets 0.1,1 --seeds 1
$PY -I $HERE/worker.py $OUT/tasks_ctg.txt $OUT ctg --tag a > $OUT/logs/ctg.log 2>&1 &
$PY -I $HERE/worker.py $OUT/tasks_ours.txt $OUT ours --threads 4 --tag a > $OUT/logs/ours.log 2>&1 &
wait
$PY -I $HERE/analyze.py $OUT $OUT/plots
