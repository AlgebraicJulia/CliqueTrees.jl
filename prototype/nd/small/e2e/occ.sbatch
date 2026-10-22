#!/bin/bash
#SBATCH --job-name=itay-gpus
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=01:30:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/rounds/small/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/rounds/small/logs/%x-%j.err
# Batch kernel throughput vs block width / registers (resident blocks per SM) on large batches (small/e2e/occ.jl);
# twin detection sort vs hash. Code: e2e/code4 (GSmall.jl with the hash twins and the occupancy counters).
set -u
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
E=$R/small/e2e
cd $E/code4
export LEAVES=$R/small/leaves SETS=$E/sets
export OTAGS=ecology2_L2000,G3_circuit_L2000,audikw_1_L2000,sweep_L1000_x3000,sweep_L2000_x3000,thermal2_L10000,ecology2_L10000,sweep_L10000_x800,sweep_L50000_x100,ecology2_L50000
OUT=$E/results/occ.tsv julia --project=$R/env -t8 occ.jl
echo DONE
