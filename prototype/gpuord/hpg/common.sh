# sourced by gpuord jobs / login-node prep
source /blue/fairbanksj/itaykadosh/gpu/env.sh
GO=/blue/fairbanksj/itaykadosh/gpu/gpuord
export JULIA_DEPOT_PATH=$GO/depot:/blue/fairbanksj/itaykadosh/gpu/julia_depot
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
