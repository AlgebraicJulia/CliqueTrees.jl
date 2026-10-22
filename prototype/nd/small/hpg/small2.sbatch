#!/bin/bash
#SBATCH --job-name=itay-gpus
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96gb
#SBATCH --time=04:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/rounds/small/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/rounds/small/logs/%x-%j.err
# Small-graph modes of GPU-MF1 (rounds/SMALL_NOTES.md, GSmall.jl) on one B200, fill config (mf1, relax 0.1, unit weights):
#   single graphs 0.5k-100k (lattices, SuiteSparse small set, TN corpus): replay (CUDA-graph driver), one block
#   (nw 8/16/32 warps), cooperative grid (G = 2 ... 2*nSM blocks), auto; phase profiles; CPU/GPU race.
#   ND leaves (recursive METIS separators of 1M-vertex graphs, leaf bounds 50k/10k/2k): grounds_batch (one block
#   per leaf, nw 8/16) and the leaves one after another (auto).
# Then, on the same node, Richard's new serial MF1 (rr/renv) on the same single graphs and leaves: 1 thread and
# 8 threads over the leaves (short).
set -u
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
V=$R/small
mkdir -p $V/logs $V/results $V/leaves
cd $V/code2
export CORPUS=/blue/fairbanksj/itaykadosh/gpu/tncorpus
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
export LEAVES=$V/leaves
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"
SS=bcsstk17,cbuckle,olafu,crystm03,jnlbrng1,ct20stif,nasasrb,apache1,thermal1,s3dkt3m2,thermomech_TC
LAT=lat2_16,lat2_23,lat2_32,lat2_45,lat2_64,lat2_90,lat2_100,lat2_128,lat2_181,lat2_224,lat3_6,lat3_8,lat3_10,lat3_13,lat3_16,lat3_20,lat3_25,lat3_32
TN=corpus:qaoaamp_N240_p3_s0,corpus:qaoaamp_N1000_p3_s0,corpus:qaoaamp_N2000_p3_s0,corpus:qaoaamp_N4000_p3_s0,corpus:qaoaamp_N4000_p5_s0,corpus:rndhyp_T1000,corpus:rndhyp_T3000,corpus:rndhyp_T10000,corpus:mc3sat_N1000,corpus:mc3sat_N3000,corpus:reg3_N1000_s0,corpus:reg3_N10000_s0,corpus:reg3_N30000_s0,corpus:rqc10x10_m24_s0,corpus:syc54_m40_s0,corpus:lm_mera_d7_b64
NDG=ecology2,thermal2,G3_circuit,lat2_724,audikw_1
LMAX=50000,10000,2000
LTAGS=$(for g in ${NDG//,/ }; do for l in ${LMAX//,/ }; do echo -n "${g}_L$l,"; done; done)
export OUT=$V/results/small_b200_v2.tsv SINGLE=$SS,$LAT,$TN NDG LMAX
export RACE=bcsstk17,lat2_100,jnlbrng1,corpus:qaoaamp_N4000_p5_s0,apache1,thermal1
export PROFG=lat2_100,bcsstk17,jnlbrng1,corpus:qaoaamp_N4000_p5_s0,lat2_32
julia --project=$R/env -t2 small/hpg/sgpu.jl
echo GPU DONE
export OUT=$V/results/small_cpu_v2.tsv LTAGS
julia --project=$R/rr/renv -t8 small/hpg/sser.jl
echo DONE
