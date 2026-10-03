#!/bin/bash
# SemiringGPU full test suite (test/runtests.jl full) on one HiPerGator GPU node. TEMPLATE: not submitted
# by anything; conventions from HPG_NOTES.md.
#
# Before submitting:
#   - sync the tree to $G/SemiringGPU as HPG_NOTES.md says (exclude Project.toml, Manifest.toml,
#     LocalPreferences.toml and external/ROME/CMakeLists.txt);
#   - after any package change, precompile once on the login node (source env.sh; julia --project=.
#     -e 'using Pkg; Pkg.precompile()'), and do not run two jobs that may precompile at the same time.
#
# Submit (the partition and GPU below are the RTX PRO 6000 defaults; override them on the command line):
#   sbatch /blue/fairbanksj/itaykadosh/gpu/SemiringGPU/test/hpg_tests.sbatch
#   sbatch --partition=hpg-b200 --gpus=b200:1 /blue/fairbanksj/itaykadosh/gpu/SemiringGPU/test/hpg_tests.sbatch
#   sbatch --export=ALL,SUITE=quick ...             (the quick suite instead of full)
#
# Output: the summary in logs/%x-%j.out; one log per test file in logs/runtests-<jobid>/ (and the GEMM
# tuning file the suite wrote there; the depot's tuning cache is not touched).
#
#SBATCH --job-name=sgpu-tests
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=08:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/logs/%x-%j.err
#SBATCH --chdir=/blue/fairbanksj/itaykadosh/gpu/SemiringGPU

G=/blue/fairbanksj/itaykadosh/gpu
source $G/env.sh                       # depot, TMPDIR, caches, no core dumps, julia module, JULIA_CPU_TARGET
export JULIA_NUM_THREADS=16 OMP_NUM_THREADS=16
export SEMIRINGGPU_TEST_THREADS=16     # the -t of every test process (runtests.jl's default is auto)
export SEMIRINGGPU_TEST_LOGDIR=$G/logs/runtests-${SLURM_JOB_ID}

echo "job ${SLURM_JOB_ID} on $(hostname), $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
julia --version

julia --project=. test/runtests.jl "${SUITE:-full}"
status=$?
echo "runtests.jl exit status $status ($(date)); logs in $SEMIRINGGPU_TEST_LOGDIR"
exit $status
