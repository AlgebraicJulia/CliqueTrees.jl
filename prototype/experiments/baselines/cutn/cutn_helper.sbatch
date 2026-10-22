#!/bin/bash
#SBATCH --job-name=itay-cutn
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/baselines/tn/logs/cutnh-%j.out
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
# usage: sbatch cutn.sbatch <outdir> [smoke | resume <old_jobid>]
#  einsum networks: 4 workers x 1 thread (cfg c1); frontier: 3 workers x 8 threads (c8); concurrently on one B200
#  (--exec 1 runs small contractions on the GPU; keepalive on einsum worker 0).
#  Each worker runs in a relaunch loop: a native crash (SIGFPE) costs one item (recorded as status=crash), not the worker.
set -u
ulimit -c 0
B=/blue/fairbanksj/itaykadosh/gpu/baselines/tn
module load python/3.12 >/dev/null 2>&1
. $B/venv_cutn/bin/activate
export TNCORPUS_DIR=/blue/fairbanksj/itaykadosh/gpu/tncorpus FRONTIER_DIR=/blue/fairbanksj/itaykadosh/gpu/frontier
export CTGPLUGIN=/blue/fairbanksj/itaykadosh/gpu/ctgplugin FTN_DIR=/blue/fairbanksj/itaykadosh/gpu/frontier/scripts
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUPY_CACHE_DIR=$B/tmp/cupy
# helper: extra workers on the queues of a running main job (flock-shared). usage: sbatch cutn_helper.sbatch <outdir> <main_jobid> <tagprefix>
OUT=${1:?outdir}; MAIN=${2:?main jobid}; P=${3:?tag prefix}
QE=$OUT/queue_einsum.$MAIN; QF=$OUT/queue_frontier.$MAIN
runw() {  # queue threads tag keepalive : relaunch until the queue is empty (the worker exits 0 on empty queue)
  local n=0
  while true; do
    python $B/code/cutn_worker.py $1 $OUT --threads $2 --tag $3 --exec 1 --keepalive $4 >> $OUT/logs/w_$3.log 2>&1 && break
    n=$((n+1)); echo "$(date) worker $3 exited nonzero (#$n), relaunching" >> $OUT/logs/w_$3.log
    [ $n -ge 200 ] && break
  done
}
for k in 0 1 2 3; do runw $QE 1 ${P}e$k $([ $k = 0 ] && echo 1 || echo 0) & done
for k in 0 1 2; do runw $QF 8 ${P}f$k 0 & done
wait
echo done $(date)
