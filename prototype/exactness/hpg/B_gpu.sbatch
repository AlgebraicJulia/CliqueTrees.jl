#!/bin/bash
#SBATCH --job-name=itay-exact
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/exactness/logs/Bgpu-%j.out
# Experiment B, GPU side: execute every tree of resB/trees/*.pkl on one B200 (torch, complex64, warm median of 3),
# then (if the cuquantum venv of the baselines agent imports) cuTensorNet on the same paths + its own path.
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
set -u
E=/blue/fairbanksj/itaykadosh/gpu/exactness
source $E/common.sh
cd $E
OUT=$E/resB
nvidia-smi --query-gpu=name,memory.total --format=csv
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
$E/venv/bin/python -I src/contractB.py $OUT/trees $OUT/gpu_torch.tsv --maxc 13.5 --reps 3 --budget 14000 || {
  echo "torch run failed: falling back to cupy (resumes, skips the rows already written)"
  $E/venv/bin/python -I src/contractB.py $OUT/trees $OUT/gpu_torch.tsv --maxc 13.5 --reps 3 --budget 12000 --backend cupy; }
CUTN=/blue/fairbanksj/itaykadosh/gpu/baselines/tn/venv_cutn/bin/python
if $CUTN -c "import cupy, cuquantum; cupy.zeros(1)" 2>/dev/null; then
  $CUTN -I src/contract_cutn.py $OUT/trees $OUT/gpu_cutn.tsv --maxc 13.5 --reps 3 --samples 8
else
  echo "cuquantum venv not usable: skipping cuTensorNet"
fi
echo "done $(date)"
