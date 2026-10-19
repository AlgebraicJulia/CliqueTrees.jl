#!/bin/bash
cd "$(dirname "$0")"
J=~/.juliaup/bin/julia
export OUT=results/scale_laptop.tsv MODES=serial,gpu
FAMS=lat2,corpus:lattice2d,corpus:rqc,corpus:syc,corpus:qaoazz,corpus:qaoaamp,corpus:lm taskset -c 1-3 $J --project=genv -t1 scale.jl
FAMS=lat3,corpus:lattice3d NMAX=100000 taskset -c 1-3 $J --project=genv -t1 scale.jl
FAMS=corpus:regk,corpus:rnd NMAX=40000 taskset -c 1-3 $J --project=genv -t1 scale.jl
FAMS=corpus:mc3sat NMAX=3000 taskset -c 1-3 $J --project=genv -t1 scale.jl
echo DONE
