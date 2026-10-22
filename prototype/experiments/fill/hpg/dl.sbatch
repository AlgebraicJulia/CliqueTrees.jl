#!/bin/bash
#SBATCH --job-name=itay-dl
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
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
cd $DATA
for p in $SS; do
  n=${p#*/}
  [ -f $n.bin ] && continue
  d=$DATA/dl_$n; mkdir -p $d
  env -u LD_LIBRARY_PATH /usr/bin/curl -sfL --retry 3 -o $d/$n.tar.gz https://www.cise.ufl.edu/research/sparse/MM/$p.tar.gz || { echo "SKIP $n download failed"; echo -e "$n\tDL\tdownload-failed" >> $RES/status.tsv; rm -rf $d; continue; }
  tar -xzf $d/$n.tar.gz -C $d && $JL $CODE/convert.jl $d/$n/$n.mtx $DATA/$n.bin || echo "SKIP $n convert failed"
  rm -rf $d
done
for l in $LAT; do [ -f $l.bin ] || $JL $CODE/convert.jl $l $DATA/$l.bin || echo "SKIP $l"; done
ls -la $DATA/*.bin
echo DONE
