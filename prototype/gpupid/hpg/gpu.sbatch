#!/bin/bash
#SBATCH --job-name=itay-gpu
#SBATCH --dependency=singleton
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-rtx6000 --gpus=rtx_pro_6000:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/gpupid/logs/gpu-%j.out
# usage: sbatch [--partition=hpg-b200 --gpus=b200:1] gpu.sbatch <names file> <tlimit> <out.tsv>
set -u
source /blue/fairbanksj/itaykadosh/gpu/gpupid/common.sh
cd $GP
nvidia-smi --query-gpu=name,memory.total --format=csv
for attempt in 1 2 3 4 5 6; do
  julia --project=$GP/env -t 2 $GP/scripts/run_gpu.jl $3 $2 @$1
  rc=$?; echo "attempt $attempt exit $rc"; [ $rc -eq 0 ] && break
done
echo "done $(date)"
