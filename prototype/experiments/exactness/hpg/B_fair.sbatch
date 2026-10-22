#!/bin/bash
#SBATCH --job-name=itay-exact
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/exactness/logs/BFair-%j.out
# Experiment B, fair executor: cuTensorNet on ALL trees (crash-proof driver). ONLY=net1,net2 and OUT=name override (smoke test).
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
E=/blue/fairbanksj/itaykadosh/gpu/exactness
cd $E
ulimit -c 0
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
CUTN=/blue/fairbanksj/itaykadosh/gpu/baselines/tn/venv_cutn/bin/python
ARGS=""; [ -n "${ONLY:-}" ] && ARGS="--only $ONLY"
$CUTN -I src/contract_cutn.py resB/trees resB/${OUT:-gpu_cutn_fair.tsv} --maxc 13.5 --reps 3 --samples 8 --mem-gb 120 $ARGS
echo "done $(date)"
