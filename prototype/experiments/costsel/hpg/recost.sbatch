#!/bin/bash
#SBATCH --job-name=itay-csel
#SBATCH --account=fairbanksj --qos=fairbanksj-b
#SBATCH --partition=hpg-default,hpg-turin,hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/costsel/logs/recost-%j.out
# cotengra re-costing of the selected / mass-search / polished trees -> rc/<net>.tsv
set -u
source /blue/fairbanksj/itaykadosh/gpu/costsel/hpg/common.sh
cd $C; mkdir -p rc logs/rc
PY=$E/venv/bin/python
cat queue.txt | xargs -P ${SLURM_CPUS_PER_TASK:-32} -I{} sh -c '[ -f pool/{}.pool ] && ! grep -q "^#END" rc/{}.tsv 2>/dev/null && '$PY' -I src/recost.py {} pool gpu rc/{}.tsv > logs/rc/{}.log 2>&1; true'
echo "done $(date): $(grep -l '^#END' rc/*.tsv | wc -l)"
# pickles for the real-contraction rerun (networks of experiment B where the selection changed the tree)
$PY -I src/mkpklB.py queueB24.txt pool gpu rc $E/resB/trees pklB 0.01 > logs/mkpklB.log 2>&1
echo "pklB: $(ls pklB/*.pkl 2>/dev/null | wc -l) networks"
