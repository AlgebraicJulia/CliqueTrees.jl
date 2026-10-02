#!/bin/bash
# Runs ROME 3x per graph (sequentially), CPU reference check disabled. Logs to external/logs/.
EXT=/home/itaykadosh/Projects/GATAS/SemiringGPU/external
BIN=$EXT/ROME/build/bin/rome
mkdir -p $EXT/logs
for g in ${GRAPHS:-grid3d-25 grid2d-150 grid3d-30 grid2d-180}; do
  for r in 1 2 3; do
    log=$EXT/logs/$g${TAG}.run$r.txt
    s=$(date +%s.%N)
    env ROME_SKIP_CHECK=1 ${EXTRA_ENV} OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 /usr/bin/time -f "MAXRSS_KB %M" \
      $BIN $EXT/rome_mtx/$g.mtx ignore > $log 2>&1
    rc=$?
    e=$(date +%s.%N)
    echo "PROCESS_WALL $(echo "$e - $s" | bc) rc=$rc" >> $log
    echo "$g run$r rc=$rc $(grep -E 'computing|PROCESS_WALL' $log | tr '\n' ' ')"
  done
done
