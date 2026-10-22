#!/bin/bash
#SBATCH --job-name=itay-gndcpu
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=320gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/rounds/nd/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/rounds/nd/logs/%x-%j.err
# CPU baselines for GPU ND + MF1 (rounds/ND_NOTES.md): CliqueTrees rmain (598b013) + Metis, 1 core per process:
#   METIS     = METIS_NodeND (CliqueTrees METIS())
#   NDMF1_L   = ND{3}(MF(strategy=1), METISND(); width=L, level=60): METIS separators, serial MF1 on leaves of <= L
# nnz(L), flops, tw by the sbench.jl recount (unit weights). Serial MF1 itself: main_table (job 46754897, same env).
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
V=$R/nd
P=$V/results/cpu_parts
mkdir -p $V/logs $P
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
export CAP=${CAP:-5400} TMAX=600 V P
G="lat2_2048 lat3_140 Queen_4147 lat2_1448 lat3_100 Bump_2911 lat2_1024 G3_circuit Flan_1565 Hook_1498 StocF-1465 af_shell10 Geo_1438 Serena thermal2 lat2_724 ecology2 audikw_1 lat3_64 parabolic_fem lat2_512 lat2_362 thermomech_dM lat3_40 G2_circuit Dubcova3 thermomech_TC lat2_224"
rm -f $V/cpu_tasks.txt
for g in $G; do
  for a in METIS NDMF1_2048 NDMF1_8192; do echo "rmain $V/cenv $a $g" >> $V/cpu_tasks.txt; done
done
run1() {
  tag=$1; env=$2; alg=$3; g=$4
  f=$P/${tag}_${alg}_${g}.tsv
  [ -s $f ] && return 0
  timeout $CAP julia --project=$env -t1 $V/code/hpg/ndcpu.jl $tag $f.tmp $alg $g > $P/${tag}_${alg}_${g}.log 2>&1
  st=$?
  if [ -s $f.tmp ]; then mv $f.tmp $f; else echo -e "$(hostname)\t$tag\t$g\t\t\t$alg\t\t\t\t$([ $st = 124 ] && echo timeout || echo error$st)" > $f; rm -f $f.tmp; fi
}
export -f run1
cd $V/code
xargs -P 10 -L 1 bash -c 'run1 "$@"' _ < $V/cpu_tasks.txt
cat $P/*.tsv > $V/results/cpu_milan.tsv
echo DONE
