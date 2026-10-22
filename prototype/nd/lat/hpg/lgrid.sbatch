#!/bin/bash
#SBATCH --job-name=itay-gpus
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200gb
#SBATCH --time=03:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/rounds/lat/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/rounds/lat/logs/%x-%j.err
# LATENCY_NOTES.md step 1b: the per-round phase profile with narrow grids (bpsm blocks per SM).
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
V=$R/lat
TAGD=${TAGD:-grid1}
mkdir -p $V/logs $V/results/$TAGD
cd $V/${CODE:-code2}
export CORPUS=/blue/fairbanksj/itaykadosh/gpu/tncorpus
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
export OUT=$V/results/$TAGD/prof_b200.txt RDIR=$V/results/$TAGD/rounds
JL="julia --project=$R/env -t1"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
# grid size test: blocks per SM (bpsm) 1, 2, 4 against the occupancy default (prof1 job)
S=bcsstk17,crystm03,jnlbrng1,lat2_100,lat3_20,ct20stif,apache1,thermal1,lat3_32,G2_circuit,parabolic_fem,ecology2,thermal2,audikw_1,Geo_1438
for b in 1 2 4; do
  EXTRA="(; bpsm = $b)" TAG="b$b-" $JL lprof.jl $S
done
echo DONE
