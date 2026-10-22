#!/bin/bash
# usage: mkorders.sh <name> <dir: corpus or frontier root>  -> data/ord/<name>__{hbt__T10,nd__T0}__s1.ord
n=$1; d=$2
export TNCORPUS_DIR=/path/to/tn-work/tncorpus
julia=~/.juliaup/bin/julia
$julia -t 1 --project=/path/to/tn-work/mtenv /path/to/tn-work/frontier/scripts/orders.jl $n hbt 10 /path/to/tn-work/gpuanneal/data/ord $d 1 > /path/to/tn-work/gpuanneal/logs/ord/$n.hbt.log 2>&1
$julia -t 1 --project=/path/to/tn-work/mtenv /path/to/tn-work/frontier/scripts/orders.jl $n nd 0 /path/to/tn-work/gpuanneal/data/ord $d 1 > /path/to/tn-work/gpuanneal/logs/ord/$n.nd.log 2>&1
