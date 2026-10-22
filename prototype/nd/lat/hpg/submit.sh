#!/bin/bash
# sync a code bundle and the hpg/ scripts to HPG and submit an sbatch file (needs the ssh master)
# usage: hpg/submit.sh CODEDIR SBATCH [VAR=value ...]   e.g. hpg/submit.sh code3 hpg/ab.sbatch TAGD=ab1
set -e
cd "$(dirname "$0")/.."
C=$1; B=$2; shift 2
R=/blue/fairbanksj/itaykadosh/gpu/rounds/lat
rsync -a --delete $C/ hpg:$R/$C/
rsync -a hpg/ hpg:$R/hpg/
EXP=$(IFS=,; echo "ALL,CODE=$C${*:+,$*}")
ssh hpg "cd $R && sbatch --export=$EXP $B"
