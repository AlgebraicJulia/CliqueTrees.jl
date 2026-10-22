#!/bin/bash
n=$1
export TNCORPUS_DIR=/path/to/tn-work/tncorpus
~/.juliaup/bin/julia -t 1 --project=/path/to/tn-work/mtenv /path/to/tn-work/frontier/scripts/orders.jl $n hbt 10,100 /path/to/tn-work/gpuanneal/data/ord /path/to/tn-work/frontier 1 > /path/to/tn-work/gpuanneal/logs/ord/$n.hbt.log 2>&1
~/.juliaup/bin/julia -t 1 --project=/path/to/tn-work/mtenv /path/to/tn-work/frontier/scripts/orders.jl $n nd 0 /path/to/tn-work/gpuanneal/data/ord /path/to/tn-work/frontier 1 > /path/to/tn-work/gpuanneal/logs/ord/$n.nd.log 2>&1
