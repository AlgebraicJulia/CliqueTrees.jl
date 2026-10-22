#!/bin/bash
#SBATCH --job-name=itay-cudss
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240gb
#SBATCH --time=10:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/cudss/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/cudss/logs/%x-%j.err
# End-to-end cuDSS Cholesky with our orderings (tn-work/cudss/NOTES.md): cudss_bench.jl over NAMES x CFGS,
# 32 Julia threads (GND hybrid CPU FM) and cuDSS MT layer with host_nthreads 32. OUT, CFGS, NAMES from the caller.
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
C=/blue/fairbanksj/itaykadosh/gpu/cudss
export OMP_NUM_THREADS=32          # common.sh sets 1; cuDSS's gomp threading layer gets host_nthreads=32 explicitly too
export SRC=$C/src PERMDIR=$C/perms
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
export OUT=${OUT:-$C/results/cudss_b200.tsv}
export NAMES=${NAMES:-"lat2_224,thermomech_TC,Dubcova3,G2_circuit,lat3_40,thermomech_dM,lat2_362,lat2_512,parabolic_fem,lat3_64,audikw_1,ecology2,lat2_724,thermal2,Serena,Geo_1438,StocF-1465,Hook_1498,af_shell10,Flan_1565,G3_circuit,lat2_1024,lat3_100,lat2_1448,lat3_140,lat2_2048,Bump_2911,Queen_4147"}
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"
echo "CFGS=$CFGS OUT=$OUT"
cd $C/code
julia --project=$C/env -e 'using Pkg; Pkg.precompile()' 2>&1 | tail -3
julia --project=$C/env -t32 cudss_bench.jl
echo DONE
