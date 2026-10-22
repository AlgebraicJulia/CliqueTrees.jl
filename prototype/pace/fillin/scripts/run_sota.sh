#!/bin/bash
# run_sota.sh <solver: tamaki|laakeri> <tlimit s> <out.tsv> <instance files...>
# Wall time per instance, output saved to results/sol_<solver>/<name>.fill; timeout -> TIMEOUT.
S=$1; TL=$2; OUT=$3; shift 3
D=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p $D/results/sol_$S
JAVA=$D/../tools/jdk/bin/java
for f in "$@"; do
  name=$(basename $f); sol=$D/results/sol_$S/$name.fill
  t0=$(date +%s.%N)
  case $S in
    tamaki) timeout -s KILL $TL $JAVA -Xss256m -Xmx2g -XX:+UseSerialGC -XX:ActiveProcessorCount=1 -cp $D/tools/build/tamakiB fillin.main.Solver < $f > $sol 2>/dev/null; rc=$? ;;
    laakeri) timeout -s KILL $TL $D/tools/src/PACE2017-min-fill/main < $f > $sol 2>/dev/null; rc=$? ;;
  esac
  t1=$(date +%s.%N); t=$(echo "$t1 - $t0" | bc)
  if [ $rc -eq 0 ]; then st=OK; elif [ $rc -eq 137 ]; then st=TIMEOUT; else st=ERR$rc; fi
  echo -e "$name\t$st\t$(wc -l < $sol)\t$t" >> $OUT
done
