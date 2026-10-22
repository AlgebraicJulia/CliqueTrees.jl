#!/bin/bash
#SBATCH --job-name=itay-exact
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/exactness/logs/BStrong2-%j.out
# Execute all stronger-cotengra trees (resBS/trees, from costStrong.py) + ours60 + ctg_hyper60 with the fixed contract_cutn.py.
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
E=/blue/fairbanksj/itaykadosh/gpu/exactness; cd $E; ulimit -c 0
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
CUTN=/blue/fairbanksj/itaykadosh/gpu/baselines/tn/venv_cutn/bin/python
$CUTN -I src/contract_cutn.py resBS/trees resBS/gpu_cutn_strong2.tsv --maxc 13.5 --reps 3 --mem-gb 120 --own 0
echo "done $(date)"
