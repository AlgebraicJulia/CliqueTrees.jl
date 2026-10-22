#!/bin/bash
cd "$(dirname "$0")"
for g in jnlbrng1 bcsstk17 crystm03 cbuckle olafu ct20stif nasasrb thermal1 apache1 s3dkt3m2 thermomech_TC Dubcova3 G2_circuit thermomech_dM parabolic_fem ecology2 \
  lat2_64 lat2_128 lat2_256 lat2d_256x256 lat3d_32 qaoaamp_N4000_p5_s0 qaoazz_N4000_p8_s0 rqc16x16_m48_s0 syc54_m56_s0 lat2_512 lat2_724; do
  echo "== $g $(date +%T)"
  timeout 1500 taskset -c 1-8 ~/.juliaup/bin/julia --project=../rounds/genv -t1 run.jl $g 2>&1 | grep -v "^$"
  echo "exit $?"
done
echo ALLDONE
