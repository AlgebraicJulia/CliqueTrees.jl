source /blue/fairbanksj/itaykadosh/gpu/env.sh
C=/blue/fairbanksj/itaykadosh/gpu/costsel
E=/blue/fairbanksj/itaykadosh/gpu/exactness
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TNCORPUS_DIR=/blue/fairbanksj/itaykadosh/gpu/tncorpus
export TNCORPUS_LOAD=$TNCORPUS_DIR/load.jl
export GPUCOST_SRC=$C/gsrc EXACT_SRC=$E/src
export TMPDIR=$C/tmp
# CPU Julia: the exactness env (CliqueTrees itay/pipeline 90964ec copy, precompiled); GPU Julia: genv (copy of rounds/env)
export CPU_DEPOT=$E/depot:/blue/fairbanksj/itaykadosh/gpu/pipeline/depot:/blue/fairbanksj/itaykadosh/gpu/gpusep/depot:/blue/fairbanksj/itaykadosh/gpu/julia_depot
export GPU_DEPOT=$C/depot:/blue/fairbanksj/itaykadosh/gpu/rounds/depot:/blue/fairbanksj/itaykadosh/gpu/julia_depot
export REF_TD=$E/resA/td:$E/resB/td
