#!/bin/bash
#SBATCH --job-name=itay-ct
#SBATCH --dependency=singleton
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/pace/logs/pace-%j.out
# usage: sbatch hpg_cpu.sbatch <solver> <set>
#   solver: tamaki17 | tamaki16 | uplow22 | septw      set: exact | bonus
# Runs every instance of the set, 1800 s each, one process per core (uplow22: two cores each).
# Prereq: this directory synced to /blue/fairbanksj/itaykadosh/gpu/pace/ (see README, "HPG").
set -u
P=/blue/fairbanksj/itaykadosh/gpu/pace
SOLVER=${1:?solver}; SET=${2:?set}; T=${T:-1800}
export JULIA=${JULIA:-julia}
export MTENV=/blue/fairbanksj/itaykadosh/gpu/ct-mt/envs/mt-65e72a3
source /blue/fairbanksj/itaykadosh/gpu/ct-mt/common.sh
export JAVA=$P/tools/jdk/bin/java JXMX=${JXMX:-30g} TMPDIR=${TMPDIR:-/tmp}
NW=$SLURM_CPUS_PER_TASK; [ "$SOLVER" = uplow22 ] && NW=$((NW / 2))
# Tamaki runs with -Xmx30g as in PACE: 7 workers x 30 GB fits in 240 GB
case "$SOLVER" in tamaki17|tamaki16) NW=${NW_JAVA:-7};; esac
mkdir -p $P/results_hpg $P/logs
echo "node $(hostname), $NW workers, $SOLVER $SET T=$T, start $(date)"
python3 -I $P/tools/run.py $SOLVER $SET $T $NW $P/results_hpg
echo "all done $(date)"
