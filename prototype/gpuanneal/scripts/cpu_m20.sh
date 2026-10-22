#!/bin/bash
# CPU baselines on syc53_m20_amp, laptop (16 threads), run alone.
cd /path/to/tn-work/gpuanneal
PY="/path/to/tn-work/tncorpus/venv/bin/python -I"
SD=data/seeds/syc53_m20_amp
for ni in 5 20 50 200; do $PY scripts/ctg_pt.py syc53_m20_amp $SD.rg1.tree 600 16 16 results/ctgpt/m20_unc_ni$ni Inf $ni 1; done
for w in 30 27; do for r in 1 2 3; do for ni in 5 20; do $PY scripts/ctg_pt.py syc53_m20_amp $SD.rg1.tree 600 16 16 results/ctgpt/m20_W${w}_ni${ni}_r$r $w $ni $r; done; done; done
for r in 1 2 3; do ~/.juliaup/bin/julia -t 16 --project=treesa_env scripts/treesa_run.jl /path/to/tn-work/frontier/data/syc53_m20_amp.tnc results/treesa/rep$r.W27 2,5,20 16 27; done
for ni in 1 2 5 10 20 50; do ~/.juliaup/bin/julia -t 16 --project=treesa_env scripts/treesa_run.jl /path/to/tn-work/frontier/data/syc53_m20_amp.tnc results/treesa/any.unc $ni 16; done
echo ALLDONE
