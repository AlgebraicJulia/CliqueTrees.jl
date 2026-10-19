#!/bin/bash
#SBATCH --job-name=itay-ct
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/gpuord/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/gpuord/logs/%x-%j.out
# CPU-only references for the exact GPU benchmark (no GPU in this job)
source /blue/fairbanksj/itaykadosh/gpu/gpuord/common.sh
cd $GO
julia --project=env -t 32 exact/cpu_ref.jl exact/data results/cpu_ref_milan_t32.jls
