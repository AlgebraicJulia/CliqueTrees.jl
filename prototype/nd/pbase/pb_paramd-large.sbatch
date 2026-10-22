#!/bin/bash
#SBATCH --job-name=pb-paramd-large
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400gb
#SBATCH --time=04:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/pbase/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/pbase/logs/%x-%j.err
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH; export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
module load gcc/14.2.0 openmpi/5.0.7 >/dev/null 2>&1
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
B=/blue/fairbanksj/itaykadosh/gpu/pbase
export PARAMD=/blue/fairbanksj/itaykadosh/gpu/rounds/paramd/paramd PND=$B/pnd WORKDIR=$B/tmp/$SLURM_JOB_ID
export DATA=/blue/fairbanksj/itaykadosh/gpu/rounds/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data CORPUS=/blue/fairbanksj/itaykadosh/gpu/tncorpus
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p $WORKDIR; cd $B
hostname; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"
# graphs ascending by n; rows are appended per (graph, method, threads) so partial results are usable
julia --project=env -t1 pbench.jl res/paramd-large.tsv PARAMD 1,8,32,64 lat2_724 thermal2 Serena Geo_1438 StocF-1465 Hook_1498 af_shell10 Flan_1565 G3_circuit lat2_1024 Bump_2911 lat3_100 Queen_4147 lat2_1448 lat3_140 lat2_2048
rm -rf $WORKDIR
echo DONE
