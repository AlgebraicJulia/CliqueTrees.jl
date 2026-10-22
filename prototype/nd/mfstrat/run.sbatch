#!/bin/bash
#SBATCH --job-name=itay-mfs
#SBATCH --account=fairbanksj
#SBATCH --qos=fairbanksj
#SBATCH --partition=hpg-default,hpg-turin,hpg-milan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400gb
#SBATCH --time=06:00:00
#SBATCH --output=/blue/fairbanksj/itaykadosh/gpu/mfstrat/logs/%x-%j.out
#SBATCH --error=/blue/fairbanksj/itaykadosh/gpu/mfstrat/logs/%x-%j.err
export PATH=/apps/julia/1.12.6/bin:/apps/compilers/gcc/14.2.0/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/gcc/14.2.0/lib64:${LD_LIBRARY_PATH:-}
source /blue/fairbanksj/itaykadosh/gpu/rounds/common.sh
M=/blue/fairbanksj/itaykadosh/gpu/mfstrat
export CORPUS=/blue/fairbanksj/itaykadosh/gpu/tncorpus
export DATA=$R/v2/data:/blue/fairbanksj/itaykadosh/gpu/fill/data
export TMAX=600 M
hostname; lscpu | grep -E "Model name|^CPU\(s\)"
cd $M/code
export JULIA_NUM_THREADS=1
julia --project=$M/env -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile(); using CliqueTrees, AMD, Metis; println("precompiled ", pkgversion(CliqueTrees))' || exit 1
ALGS=MF0,MF1,MF2,MF3,MF4,AMD,AMF,METIS
rm -f $M/res/mfs_smoke.tsv
julia --project=$M/env -t1 mfbench.jl smoke $M/res/mfs_smoke.tsv $ALGS lat2_32 || exit 2
G="lat2_2048 lat3_140 Queen_4147 lat2_1448 lat3_100 Bump_2911 lat2_1024 G3_circuit Flan_1565 Hook_1498 StocF-1465 af_shell10 Geo_1438 Serena thermal2 ecology2 audikw_1 lat3_64 lat2_724 lat2_512 parabolic_fem lat2_362 lat2_224 lat3_40 thermomech_dM G2_circuit Dubcova3 thermomech_TC lat2_100 s3dkt3m2 apache1 thermal1 nasasrb ct20stif lat3_25 lat3_32 jnlbrng1 crystm03 lat3_20 olafu lat3_16 cbuckle bcsstk17 corpus:qaoaamp_N4000_p5_s0 corpus:rndhyp_T10000 corpus:rndhyp_T30000 corpus:mc3sat_N3000"
rm -f $M/tasks.txt; for g in $G; do echo $g >> $M/tasks.txt; done
export ALGS CAP=14400
run1() {
  g=$1; f=$M/parts/${g//:/_}.tsv
  [ -s $f ] && return 0
  timeout $CAP julia --project=$M/env -t1 $M/code/mfbench.jl mf9514320 $f.tmp $ALGS $g > $M/parts/${g//:/_}.log 2>&1
  [ -s $f.tmp ] && mv $f.tmp $f || echo -e "$(hostname)\tmf\t$g\tfail" > $f
}
export -f run1
cd $M/code
xargs -P 28 -L 1 bash -c 'run1 "$@"' _ < $M/tasks.txt
cat $M/parts/*.tsv > $M/res/mfs_all.tsv
echo DONE
