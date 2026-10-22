#!/bin/bash
# method (b) at B = 1 s and 10 s, then the reconfiguration-only ablation (laptop GPU, one at a time)
cd /home/itaykadosh/Projects/GATAS/tn-work/gpucost
julia --project=env scripts/run_gpu.jl data/nets30.txt data/ord results/gpu1 1 hbtT0.5s1 > logs/gpu1.log 2>&1
julia --project=env scripts/run_gpu.jl data/nets30.txt data/ord results/gpu10 10 hbtndT5.0s1 > logs/gpu10.log 2>&1
julia --project=env scripts/ablate_reconf.jl data/nets30.txt data/ord results/ablate > logs/ablate.log 2>&1
echo ALLDONE
