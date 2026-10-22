#!/bin/bash
#SBATCH --job-name=itay-gpucost-inst
#SBATCH --account=fairbanksj --qos=fairbanksj
#SBATCH --partition=hpg-milan
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/gpucost/logs/inst-%j.out
set -u
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/gpupid/common.sh
cd /blue/fairbanksj/itaykadosh/gpu/gpucost
julia --project=env -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
