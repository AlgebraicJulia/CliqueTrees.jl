#!/bin/bash
# Fixed-slicing experiment on one sliced cotengra tree:
#   cutexp.sh <name> <hslice path.json> <W target> <outdir> <core> <hbt budgets comma>
# 1. cut network = network minus the sliced indices S (make_cut.py), registered in $FRONTIER_DIR
# 2. cotengra's own per-slice tree on it (cutcheck.py) and that tree's induced ordering
# 3. our orderings of the cut network (bw, nd, hbt, hbtnd) -> trees -> raw / refined / sliced to W
# Total cost = per-slice cost x 2^|S| (the analysis adds |S| log10 2).
set -u
name=$1; pj=$2; W=$3; out=$4; core=$5; budgets=$6
S=$(dirname "$0")
FD=${FRONTIER_DIR:-/path/to/tn-work/frontier}
PY=${PY:-/path/to/tn-work/tncorpus/venv/bin/python}
JL=${JL:-$HOME/.juliaup/bin/julia}; JENV=${JENV:-/path/to/tn-work/mtenv}
mkdir -p $out/ord $out/res
exec 9>$FD/manifest.lock; flock 9
cut=$($PY $S/make_cut.py $name $pj $FD | awk '{print $1}')
# a distinct name per source tree
tag=$(basename $pj .path.json | sed 's/.*__\(hslice[0-9]*\)__T\([0-9]*\)__s\([0-9]*\)/\1T\2s\3/')
new=${cut}_${tag}
for ext in json tnc; do mv $FD/data/$cut.$ext $FD/data/$new.$ext; done
sed "s/^$cut\t/$new\t/; s|data/$cut.tnc|data/$new.tnc|" $FD/manifest_parts/$cut.tsv > $FD/manifest_parts/$new.tsv; rm $FD/manifest_parts/$cut.tsv
{ printf 'name\tfamily\tsource\tn\tm\tntensors\tnopen\tmaxdeg\twsum\twmax\tparams\tfile\n'; cat $FD/manifest_parts/*.tsv; } > $FD/manifest.tsv.$$ && mv $FD/manifest.tsv.$$ $FD/manifest.tsv
flock -u 9
nS=$(echo $cut | sed 's/.*_cut//')
echo "cut network $new, |S|=$nS"
taskset -c $core $PY $S/cutcheck.py $new $pj $out/ord/${new}__ctgslice__T0__s1.ord
taskset -c $core $PY $S/eval_orders.py $out/ord/${new}__ctgslice__T0__s1.ord $out/res $W
for m in bw nd; do taskset -c $core $JL -t 1 --project=$JENV $S/orders.jl $new $m 0 $out/ord $FD 1 2>&1 | grep "bag width"; done
for m in hbt hbtnd; do taskset -c $core $JL -t 1 --project=$JENV $S/orders.jl $new $m $budgets $out/ord $FD 1 2>&1 | grep "bag width"; done
for f in $out/ord/${new}__*.ord; do
  case $f in *ctgslice*) continue;; esac
  taskset -c $core $PY $S/eval_orders.py $f $out/res $W
done
echo "done $new"
