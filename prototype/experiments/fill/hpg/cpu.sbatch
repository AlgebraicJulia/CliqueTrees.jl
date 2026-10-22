#!/bin/bash
#SBATCH --job-name=itay-ct
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --dependency=singleton
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=400gb
#SBATCH --time=08:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/fill/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/fill/logs/%x-%j.err
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
F=/blue/fairbanksj/itaykadosh/gpu/fill
export CODE=$F/code DATA=$F/data RES=$F/results
mkdir -p $F/logs $RES $DATA/perms
source $F/instances.sh
JL="julia --project=/blue/fairbanksj/itaykadosh/gpu/fill/env -t1"
# one julia process per (instance, method), killed after CAP seconds; 10 in parallel
CAP=${CAP:-1800}; P=${P:-10}
cd $CODE
rm -f $F/tasks.txt
for i in $(echo $ALL | tr ' ' '\n' | tac); do
  if [ -f $DATA/$i.bin ] || [[ $i == lat* ]]; then for m in MF MF1 AMF AMD MMD METIS-ND; do echo "$i $m" >> $F/tasks.txt; done; fi
done
run1() {
  i=$1; m=$2
  timeout --signal=KILL $CAP $JL $CODE/cpu_one.jl $i $m; rc=$?
  [ $rc -eq 0 ] || echo -e "$i\t$m\trc=$rc(124/137=timeout>${CAP}s)" >> $RES/status.tsv
}
export -f run1; export CAP JL CODE RES DATA
xargs -P $P -L 1 bash -c 'run1 $0 $1' < $F/tasks.txt
echo DONE
