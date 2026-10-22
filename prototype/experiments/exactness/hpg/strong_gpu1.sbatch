#!/bin/bash
#SBATCH --job-name=itay-exact
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/exactness/logs/BStrong1-%j.out
# cuTensorNet own path finder (2^30 cap, ~60 s, 32 threads) on the B networks, then execute cutn60 + ours60 + ctg_hyper60 (fixed contract_cutn.py).
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
E=/blue/fairbanksj/itaykadosh/gpu/exactness; cd $E; ulimit -c 0
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
CUTN=/blue/fairbanksj/itaykadosh/gpu/baselines/tn/venv_cutn/bin/python
mkdir -p resBS
$CUTN -I src/cutn_trees.py resB/trees resBS/cutntrees 32 60
$CUTN -I src/contract_cutn.py resBS/cutntrees resBS/gpu_cutn_strong1.tsv --maxc 13.5 --reps 3 --mem-gb 120 --own 0
echo "done $(date)"
