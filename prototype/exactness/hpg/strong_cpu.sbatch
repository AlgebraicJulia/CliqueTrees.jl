#!/bin/bash
#SBATCH --job-name=itay-strongctg
#SBATCH --account=fairbanksj --qos=fairbanksj-b
#SBATCH --partition=hpg-default,hpg-turin,hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/exactness/logs/strong-%A_%a.out
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
E=/blue/fairbanksj/itaykadosh/gpu/exactness; cd $E
export TNCORPUS_DIR=/blue/fairbanksj/itaykadosh/gpu/tncorpus FRONTIER_DIR=/blue/fairbanksj/itaykadosh/gpu/frontier
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
venv/bin/python -I src/costStrong.py ${SLURM_ARRAY_TASK_ID} 6 $E/resBS/trees ${SLURM_CPUS_PER_TASK}
