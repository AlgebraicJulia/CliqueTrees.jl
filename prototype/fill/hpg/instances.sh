# SuiteSparse group/name (downloaded + converted by dl.sbatch) and generated lattices
SS="McRae/ecology2 Schmid/thermal2 AMD/G3_circuit GHS_psdef/audikw_1 Janna/StocF-1465 Janna/Flan_1565 Janna/Serena Schenk_AFE/af_shell10 Janna/Geo_1438 Janna/Hook_1498 Janna/Bump_2911 Janna/Queen_4147"
LAT="lat3_25 lat3_40 lat2_512 lat3_64 lat2_1024 lat3_100 lat2_2048 lat3_140"
SSN=""; for p in $SS; do SSN="$SSN ${p#*/}"; done
# ascending-ish size order for the GPU job; ALL used by the CPU job
ALL="$LAT $SSN"
