#!/bin/bash
#SBATCH --job-name=itay-gpu
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/gpuord/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/gpuord/logs/%x-%j.out
# one GPU (request with --partition=hpg-b200 --gpus=b200:1 or --partition=hpg-rtx6000 --gpus=rtx_pro_6000:1)
# usage: sbatch --dependency=singleton [partition/gpus] gpu.sbatch <script> [args...]
source /blue/fairbanksj/itaykadosh/gpu/gpuord/common.sh
cd $GO
nvidia-smi --query-gpu=name,driver_version --format=csv
julia --project=env "$@"
