#!/bin/bash
# usage: mkorders.sh <name> <dir: corpus or frontier root>  -> data/ord/<name>__{hbt__T10,nd__T0}__s1.ord
n=$1; d=$2
export TNCORPUS_DIR=/tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/tncorpus
julia=~/.juliaup/bin/julia
$julia -t 1 --project=/tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/mtenv /tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/frontier/scripts/orders.jl $n hbt 10 /tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/gpuanneal/data/ord $d 1 > /tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/gpuanneal/logs/ord/$n.hbt.log 2>&1
$julia -t 1 --project=/tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/mtenv /tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/frontier/scripts/orders.jl $n nd 0 /tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/gpuanneal/data/ord $d 1 > /tmp/claude-1000/-home-itaykadosh-Projects-GATAS/2e762ebb-ae79-4e3b-a677-b7caf4118124/scratchpad/gpuanneal/logs/ord/$n.nd.log 2>&1
