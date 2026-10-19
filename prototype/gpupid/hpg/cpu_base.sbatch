#!/bin/bash
#SBATCH --job-name=itay-gpupid-cpu
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/gpupid/logs/cpu-%j.out
# usage: sbatch cpu_base.sbatch <queue file: "mode seconds name" per line> <out.tsv>
set -u
GP=/blue/fairbanksj/itaykadosh/gpu/gpupid
source /blue/fairbanksj/itaykadosh/gpu/ct-mt/common.sh
ENV=/blue/fairbanksj/itaykadosh/gpu/ct-mt/envs/mt-65e72a3
LIST=$1; OUT=$2
Q=$GP/queue.$SLURM_JOB_ID; cp $LIST $Q
NW=${NW:-$SLURM_CPUS_PER_TASK}
cores=($(taskset -cp $$ | awk -F': ' '{print $2}' | tr ',' '\n' | awk -F- '{if(NF==2){for(i=$1;i<=$2;i++)print i}else print $1}'))
echo "node $(hostname), ${#cores[@]} cores, $NW workers, $(wc -l < $Q) items"
worker() {
  local core=$1
  while true; do
    local line
    line=$(flock $Q.lock sh -c "head -n 1 $Q; sed -i 1d $Q")
    [ -z "$line" ] && break
    set -- $line
    timeout $(( ${2%.*} + 300 )) taskset -c $core julia --project=$ENV -t 1 --gcthreads=1 $GP/scripts/cpu_base.jl $GP/atoms $3 $1 $2 $OUT > /dev/null 2>> $GP/logs/cpu-$SLURM_JOB_ID.err || echo -e "$3\t-\t-\t$1\tKILLED" >> $OUT
  done
}
for ((k = 0; k < NW; k++)); do worker ${cores[$k]} & done
wait
echo "all done $(date)"
