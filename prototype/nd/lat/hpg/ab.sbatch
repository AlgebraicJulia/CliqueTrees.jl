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
#SBATCH --time=04:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/rounds/lat/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/rounds/lat/logs/%x-%j.err
# LATENCY_NOTES.md: before/after of the latency changes (lat/base vs lat/ GRounds; VARS),
# then the candidate-rule sweep (rsel.jl), then the exactness check incl. persistent modes.
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
V=$R/lat
TAGD=${TAGD:-ab1}
CODE=${CODE:-code3}
mkdir -p $V/logs $V/results/$TAGD
cd $V/$CODE
export CORPUS=/blue/fairbanksj/itaykadosh/gpu/tncorpus
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
JL="julia --project=$R/env -t1"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
SMALL=bcsstk17,cbuckle,olafu,crystm03,jnlbrng1,lat2_100,lat3_20,ct20stif,nasasrb,apache1,thermal1,s3dkt3m2,lat3_32,lat2_224,thermomech_TC,Dubcova3,G2_circuit,lat3_40,thermomech_dM,lat2_512,parabolic_fem
LARGE=lat3_64,audikw_1,ecology2,lat2_724,thermal2,Serena,Geo_1438,StocF-1465,Hook_1498,af_shell10,Flan_1565,G3_circuit,lat2_1024,lat3_100,Queen_4147,Bump_2911
# 1. before/after (base vs esub=1 vs all changes), phase profile of the last variant
OUT=$V/results/$TAGD/ab_b200.tsv PROFOUT=$V/results/$TAGD/prof_b200.txt VARS="esub1=(; esub = 1)|no34=(; e34 = false)|new=(;)" $JL ab.jl $SMALL,$LARGE
echo AB DONE
[ "${ABONLY:-0}" = 1 ] && exit 0
# 2. candidate rule: rounds / fill / time
OUT=$V/results/$TAGD/rsel_b200.tsv CFGS=v2,r0.1u0,r0.1u0.02,r0.1u0.05,r0.1a16c0.05,r0.1a64c0.05,r0.1a16c0.02,r0.2u0 $JL rsel.jl $SMALL,$LARGE
echo RSEL DONE
# 3. exactness vs lat/base incl. the persistent modes and the batch
$JL hcheck.jl lat2_30 lat3_8 bcsstk17 crystm03 cbuckle 2>&1 | tail -20
echo DONE
