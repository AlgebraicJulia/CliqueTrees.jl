#!/bin/bash
cd /home/itaykadosh/Projects/GATAS/tn-work/lightcone
export CT_CAP=60
for f in data/*.pkl; do n=$(basename $f .pkl); [ $n = er_N1000_p4 ] && continue; [ -f results/$n.json ] || { /usr/bin/time -f "$n %es" nice -n 10 ./venv/bin/python scripts/bench.py $f; }; done
S_CT=4 S_EX=3 CT_CAP=60 /usr/bin/time -f "er_N1000_p4 %es" nice -n 10 ./venv/bin/python scripts/bench.py data/er_N1000_p4.pkl
