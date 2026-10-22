#!/bin/bash
#SBATCH --job-name=itay-cudss
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200gb
#SBATCH --time=08:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/cudss/logs/%x-perms-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/cudss/logs/%x-perms-%j.err
# CPU reference orders for cudss_bench.jl (NOTES.md): AMD and METIS-ND (env cudss/env = rounds/nd/henv + CUDSS),
# serial MF1 = Richard's origin/main (env rounds/rr/renv -> rounds/ct-rmain). One julia process per (alg, graph),
# 6 at a time, each killed after CAP s. Writes cudss/perms/NAME.ALG.perm.
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
C=/blue/fairbanksj/itaykadosh/gpu/cudss
export SRC=$C/src C CAP=${CAP:-5400}
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
GRAPHS=${GRAPHS:-"lat2_2048 lat3_140 Queen_4147 lat2_1448 lat3_100 Bump_2911 lat2_1024 G3_circuit Flan_1565 af_shell10 Hook_1498 StocF-1465 Geo_1438 Serena thermal2 lat2_724 ecology2 audikw_1 lat3_64 parabolic_fem lat2_512 lat2_362 thermomech_dM lat3_40 G2_circuit Dubcova3 thermomech_TC lat2_224"}
T=$C/logs/perm_tasks_$SLURM_JOB_ID.txt; rm -f $T
for g in $GRAPHS; do
  echo "rmain $R/rr/renv MF1 $g" >> $T
  echo "henv $C/env METIS $g" >> $T
  echo "henv $C/env AMD $g" >> $T
done
run1() {
  tag=$1; env=$2; alg=$3; g=$4
  [ -s $C/perms/$g.$alg.perm ] && return 0
  [ -n "$g" ] || return 0
  timeout $CAP julia --project=$env -t1 --heap-size-hint=30G $C/code/perms.jl $tag $alg $C/perms $g > $C/logs/perm_${alg}_$g.log 2>&1 || echo "FAILED/TIMEOUT $alg $g (status $?)"
}
export -f run1
xargs -P 6 -L 1 bash -c 'run1 "$@"' _ < $T
ls $C/perms | wc -l
echo DONE
