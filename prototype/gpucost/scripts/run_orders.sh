#!/bin/bash
# all seed orderings for the network list: two single-thread julia processes (laptop memory is tight)
cd /home/itaykadosh/Projects/GATAS/tn-work/gpucost
for p in data/nets30_part00 data/nets30_part01; do
  julia -t 1 --project=env scripts/mkorders.jl $p data/ord greedy,nd,hbt:0.5:1,hbt:1:1,hbtnd:5:1,hbtnd:10:1 > logs/ord_$(basename $p).log 2>&1 &
done
wait; echo ALLDONE
