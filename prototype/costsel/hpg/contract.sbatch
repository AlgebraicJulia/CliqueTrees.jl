#!/bin/bash
#SBATCH --job-name=itay-csel
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-b200 --gpus=b200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/costsel/logs/contract-%j.out
# Real B200 contraction (torch complex64, exactness contractB.py with its rescaling fix) of pklB/*.pkl:
# csel / csel_gpu trees next to ours60, ctg_hyper60, td_exact re-timed in the same job.
set -u
source /blue/fairbanksj/itaykadosh/gpu/costsel/hpg/common.sh
cd $C
nvidia-smi --query-gpu=name,memory.total --format=csv
ls pklB/*.pkl >/dev/null 2>&1 || { echo "no pickles"; exit 0; }
$E/venv/bin/python -I $E/src/contractB.py pklB res_contract.tsv --maxc 13.5 --reps 3
echo "done $(date)"
