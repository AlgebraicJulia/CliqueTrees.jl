#!/bin/bash
#SBATCH --job-name=itay-exact
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/exactness/logs/Brerun-%j.out
# Experiment B rerun of the affected rows only: networks whose complex64 result underflowed (relerr==1.0 artefact; now
# rescaled so |result|~1) + the trees that hit torch's 25-dim limit (cupy fallback). Output resB/gpu_torch_rerun.tsv.
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
E=/blue/fairbanksj/itaykadosh/gpu/exactness
source $E/common.sh
cd $E
nvidia-smi --query-gpu=name,memory.total --format=csv
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
ONLY=rnd_oe_04,qc_rand_supremacy2d_1_12_400,mc_2020_arjun_102,qc_rand_supremacy2d_4_4_100,gm_Promedas_33,wmc_2021_187:td_exact,qc_maxcut_n29_r4_p10:ours10+ours60
$E/venv/bin/python -I src/contractB.py resB/trees resB/gpu_torch_rerun.tsv --maxc 13.5 --reps 3 --budget 6000 --only $ONLY
echo "done $(date)"
