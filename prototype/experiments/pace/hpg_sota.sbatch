#!/bin/bash
#SBATCH --job-name=itay-sota
#SBATCH --dependency=singleton
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/pace/logs/pace-%j.out
# usage: sbatch hpg_sota.sbatch <solver> <set>
#   solver: rtw | arboretum | jdrasil | exacttw | tamaki17      set: exact | bonus
# 1800 s per instance, one single-threaded process per worker (RTW: ActiveProcessorCount=1, so it is compared 1-thread vs 1-thread).
# Priority: rtw bonus, rtw exact (strongest current baseline), then tamaki17 bonus if wanted.
# Prereq: this directory synced to /blue/fairbanksj/itaykadosh/gpu/pace/ (see README, "HPG").
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
P=/blue/fairbanksj/itaykadosh/gpu/pace
SOLVER=${1:?solver}; SET=${2:?set}; T=${T:-1800}
export JULIA=${JULIA:-julia}
export MTENV=/blue/fairbanksj/itaykadosh/gpu/ct-mt/envs/mt-65e72a3
source /blue/fairbanksj/itaykadosh/gpu/ct-mt/common.sh
export JAVA=$P/tools/jdk/bin/java JXMX=${JXMX:-30g} TMPDIR=${TMPDIR:-/tmp}
NW=$SLURM_CPUS_PER_TASK; [ "$SOLVER" = uplow22 ] && NW=$((NW / 2))
# Tamaki runs with -Xmx30g as in PACE: 7 workers x 30 GB fits in 240 GB
case "$SOLVER" in tamaki17|rtw|jdrasil|exacttw) NW=${NW_JAVA:-7};; esac  # JVM solvers: -Xmx30g each, 7 x 30 GB fits in 240 GB
# build artifacts are synced from the laptop: tools/build/{rtw,jdrasil,tw2022}, tools/arboretum/src/target/release/arboretum-cli
mkdir -p $P/results_hpg $P/logs
echo "node $(hostname), $NW workers, $SOLVER $SET T=$T, start $(date)"
python3 -I $P/tools/run.py $SOLVER $SET $T $NW $P/results_hpg
echo "all done $(date)"
