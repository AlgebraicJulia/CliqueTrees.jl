## Status counts

error: 5, ok: 235, skip_cost: 85, timeout: 2

Non-ok rows:

| net | tag | status |
|---|---|---|
| mc_2020_arjun_102 | cutn_own | error:cuTensorNetError:INTERNAL_ERROR (14): CUTENSORNET_STATUS_INTERNAL_ERROR |
| rnd_oe_04 | ctg_greedy | error:MemoryLimitExceeded:GPU memory limit exceeded. Device id: 0. The memory limit is 128849018880, while the minimum w |
| gm_pedigree50 | ctg_greedy | error:MemoryLimitExceeded:GPU memory limit exceeded. Device id: 0. The memory limit is 128849018880, while the minimum w |
| str_ctg_regular_04 | ctg_greedy | error:MemoryLimitExceeded:GPU memory limit exceeded. Device id: 0. The memory limit is 128849018880, while the minimum w |
| mc_2022_085 | cutn_own | timeout |
| rnd_oe_03 | ctg_greedy | error:MemoryLimitExceeded:GPU memory limit exceeded. Device id: 0. The memory limit is 128849018880, while the minimum w |
| qc_maxcut_n29_r4_p10 | cutn_own | timeout |

relerr vs complex128 reference (ok rows): median 2.36e-06, max 2.08e-04, >1e-3: 0

## 60 s budget: ours (ours60) vs cotengra (ctg_hyper60), cuTensorNet execution

| net | ours logC | ctg logC | ours W | ctg W | ours #sl | ctg #sl | ours t (s) | ctg t (s) | speedup | ours search | ctg search | TTS speedup | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gm_1kp6 | 9.90 | 9.97 | 27.4 | 22.6 | 1 | 27 | 0.00521 | 0.02788 | 5.35 | 1.1 | 60.1 | 52.03 | b |
| gm_pedigree50 | 10.36 | 12.41 | 29.9 | 29.0 | 1 | 180 | 0.01935 | 2.228 | 115.15 | 60.6 | 66.9 | 1.14 | b |
| lm_batch_likelihood_brackets_4_16d | 11.38 | 11.65 | 27.0 | 17.6 | 1 | 1996 | 0.0337 | 1.15 | 34.12 | 0.1 | 60.1 | 333.66 | b |
| lm_batch_likelihood_sentence_4_12d | 10.76 | 10.89 | 25.2 | 14.3 | 1 | 1900 | 0.01044 | 0.9636 | 92.30 | 0.1 | 60.0 | 392.34 | b |
| lm_first_last_sentence_4_16c | 12.76 | 12.76 | 26.9 | 26.9 | 16 | 16 | 0.6853 | 0.6878 | 1.00 | 1.9 | 60.0 | 23.71 | c |
| mc_2022_085 | 11.21 | 13.15 | 30.0 | 29.0 | 1 | 64 | 0.09878 | 3.603 | 36.48 | 66.5 | 70.4 | 1.11 | b |
| qaoazz_N120_p5_s1 | 9.65 | 10.42 | 27.0 | 27.0 | 1 | 1 | 0.00973 | 0.00655 | 0.67 | 61.3 | 63.0 | 1.03 | a |
| qaoazz_N240_p5_s0 | 8.01 | 9.85 | 22.0 | 25.0 | 1 | 1 | 0.00251 | 0.00388 | 1.55 | 60.8 | 98.0 | 1.61 | a |
| qc_circuit_n53_m12_s2_e8_pABCDCDAB_simplified | 9.58 | 10.07 | 25.0 | 25.0 | 1 | 2 | 0.00309 | 0.00743 | 2.40 | 60.4 | 60.7 | 1.01 | b |
| qc_maxcut_n28_r3_p3 | 8.96 | 9.10 | 23.0 | 24.0 | 1 | 1 | 0.00167 | 0.00195 | 1.17 | 60.3 | 60.4 | 1.00 | a |
| qc_qft_28 | 9.21 | 9.81 | 28.0 | 27.0 | 1 | 2 | 0.02497 | 0.04321 | 1.73 | 1.4 | 62.0 | 45.12 | b |
| qc_rand_google53_26_15 | 10.31 | 10.38 | 26.0 | 26.0 | 1 | 2 | 0.01794 | 0.01967 | 1.10 | 13.3 | 76.6 | 5.74 | b |
| qc_rand_supremacy2d_1_12_400 | 9.82 | 10.07 | 23.0 | 26.0 | 1 | 1 | 0.01966 | 0.01129 | 0.57 | 3.0 | 71.4 | 23.90 | a |
| qc_rand_supremacy2d_4_4_100 | 10.96 | 13.32 | 28.0 | 30.0 | 1 | 64 | 0.2125 | 5.83 | 27.44 | 1.7 | 83.2 | 46.42 | b |
| qc_variational_29 | 9.93 | 9.81 | 29.0 | 29.0 | 1 | 1 | 0.00508 | 0.00729 | 1.44 | 0.6 | 60.4 | 93.01 | a |
| rnd_mixed_06 | 10.17 | 10.86 | 28.8 | 28.8 | 1 | 5 | 0.02117 | 0.1063 | 5.02 | 63.8 | 70.0 | 1.10 | b |
| rnd_mixed_10 | 10.20 | 11.78 | 26.4 | 27.6 | 1 | 16 | 0.0226 | 0.74 | 32.74 | 63.8 | 120.5 | 1.90 | b |
| rnd_oe_03 | 11.30 | 13.23 | 27.4 | 29.1 | 9 | 30 | 0.1116 | 2.482 | 22.25 | 61.9 | 61.9 | 1.04 | b |
| rnd_oe_04 | 10.99 | 13.48 | 29.0 | 29.8 | 3 | 1152 | 0.06463 | 25.32 | 391.73 | 64.1 | 337.0 | 5.65 | b |
| str_ctg_lattice_02 | 10.80 | 12.44 | 26.9 | 29.9 | 1 | 4 | 0.03913 | 1.023 | 26.13 | 61.4 | 118.7 | 1.95 | b |
| str_ctg_regular_04 | 11.05 | 11.23 | 28.8 | 29.7 | 3 | 3 | 0.06452 | 0.116 | 1.80 | 60.3 | 60.2 | 1.00 | c |
| str_nw_mera_closed_120 | 10.36 | 10.36 | 25.0 | 25.0 | 1 | 1 | 0.00451 | 0.0044 | 0.98 | 18.4 | 60.1 | 3.26 | a |
| str_nw_peps_closed_333 | 9.63 | 10.51 | 23.4 | 25.6 | 1 | 4 | 0.00363 | 0.01843 | 5.08 | 61.1 | 63.8 | 1.04 | b |
| wmc_2021_187 | 10.44 | 12.75 | 30.0 | 30.0 | 2 | 256 | 0.07736 | 6.908 | 89.29 | 8.4 | 60.0 | 7.90 | b |

Summary (ours60 vs ctg_hyper60; speedup = ctg time / ours time; win >1.05, loss <0.95, else tie):

| class | n | wins | ties | losses | median speedup | geomean speedup | median TTS speedup | geomean TTS |
|---|---|---|---|---|---|---|---|---|
| a:both unsliced | 6 | 3 | 1 | 2 | 1.07 | 1.00 | 2.44 | 4.79 |
| b:ctg sliced, ours fewer/none | 16 | 16 | 0 | 0 | 26.79 | 18.24 | 3.80 | 6.79 |
| c:other | 2 | 1 | 1 | 0 | 1.40 | 1.34 | 12.35 | 4.86 |
| all | 24 | 20 | 2 | 2 | 5.05 | 7.10 | 2.61 | 6.05 |

## 10 s budget: ours (ours10) vs cotengra (ctg_hyper10), cuTensorNet execution

| net | ours logC | ctg logC | ours W | ctg W | ours #sl | ctg #sl | ours t (s) | ctg t (s) | speedup | ours search | ctg search | TTS speedup | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gm_1kp6 | 9.90 | 10.18 | 27.4 | 24.2 | 1 | 9 | 0.0052 | 0.01333 | 2.56 | 1.2 | 11.1 | 9.65 | b |
| gm_pedigree50 | 11.06 | 10.79 | 28.1 | 29.5 | 30 | 4 | 1.008 | 0.08338 | 0.08 | 11.8 | 10.7 | 0.84 | c |
| lm_batch_likelihood_brackets_4_16d | 11.38 | 11.68 | 27.0 | 17.6 | 1 | 1996 | 0.0337 | 1.193 | 35.40 | 0.2 | 10.1 | 51.19 | b |
| lm_batch_likelihood_sentence_4_12d | 10.76 | 10.89 | 25.2 | 14.3 | 1 | 1900 | 0.01046 | 0.9586 | 91.65 | 0.2 | 10.2 | 55.70 | b |
| lm_first_last_sentence_4_16c | 12.76 | 12.76 | 26.9 | 26.9 | 16 | 16 | 0.6944 | 0.6932 | 1.00 | 6.2 | 10.3 | 1.60 | c |
| mc_2020_arjun_102 | 9.89 | 10.37 | 28.0 | 28.0 | 1 | 2 | 0.09468 | 0.05908 | 0.62 | 2.7 | 159.2 | 56.16 | b |
| qaoazz_N120_p5_s1 | 9.65 | 12.50 | 27.0 | 30.0 | 1 | 64 | 0.01019 | 4.95 | 485.78 | 10.9 | 19.4 | 2.23 | b |
| qc_circuit_n53_m12_s2_e8_pABCDCDAB_simplified | 9.74 | 10.45 | 26.0 | 28.0 | 1 | 1 | 0.00457 | 0.03141 | 6.87 | 13.9 | 10.8 | 0.78 | a |
| qc_maxcut_n28_r3_p3 | 8.95 | 9.43 | 23.0 | 24.0 | 1 | 1 | 0.00163 | 0.00211 | 1.29 | 13.5 | 11.0 | 0.82 | a |
| qc_qft_28 | 9.21 | 10.82 | 28.0 | 30.0 | 1 | 4 | 0.02496 | 0.1248 | 5.00 | 1.3 | 10.9 | 8.05 | b |
| qc_rand_google53_27_20 | 10.87 | 11.11 | 27.0 | 29.0 | 1 | 2 | 0.06123 | 0.09394 | 1.53 | 8.5 | 20.8 | 2.46 | b |
| qc_rand_supremacy2d_1_12_400 | 9.82 | 10.51 | 23.0 | 26.0 | 1 | 1 | 0.01967 | 0.2661 | 13.53 | 2.9 | 19.1 | 6.54 | a |
| qc_variational_29 | 9.93 | 9.93 | 29.0 | 29.0 | 1 | 1 | 0.00508 | 0.00508 | 1.00 | 0.6 | 10.5 | 16.36 | a |
| str_ctg_lattice_02 | 10.92 | 12.68 | 28.0 | 28.3 | 1 | 48 | 0.05441 | 2.477 | 45.53 | 12.9 | 38.7 | 3.18 | b |
| str_ctg_regular_04 | 11.08 | 13.38 | 28.6 | 29.3 | 3 | 432 | 0.05938 | 12.16 | 204.74 | 13.6 | 11.1 | 1.70 | b |
| str_nw_mera_closed_120 | 10.36 | 10.36 | 25.0 | 25.0 | 1 | 1 | 0.0045 | 0.00442 | 0.98 | 10.1 | 10.3 | 1.02 | a |
| str_nw_peps_closed_333 | 9.71 | 10.67 | 23.4 | 23.8 | 1 | 6 | 0.00382 | 0.02814 | 7.37 | 14.7 | 10.1 | 0.69 | b |

Summary (ours10 vs ctg_hyper10; speedup = ctg time / ours time; win >1.05, loss <0.95, else tie):

| class | n | wins | ties | losses | median speedup | geomean speedup | median TTS speedup | geomean TTS |
|---|---|---|---|---|---|---|---|---|
| a:both unsliced | 5 | 3 | 2 | 0 | 1.29 | 2.60 | 1.02 | 2.34 |
| b:ctg sliced, ours fewer/none | 10 | 9 | 0 | 1 | 21.39 | 16.30 | 5.62 | 6.93 |
| c:other | 2 | 0 | 1 | 1 | 0.54 | 0.29 | 1.22 | 1.16 |
| all | 17 | 12 | 3 | 2 | 5.00 | 5.91 | 2.46 | 4.08 |

## Slices at the memory cap (2^30 elements per intermediate), all networks with trees

| net | ours10 #sl | ours60 #sl | ctg_hyper10 #sl | ctg_hyper60 #sl | ctg_greedy #sl | ours60 logC | ctg60 logC |
|---|---|---|---|---|---|---|---|
| gm_1kp6 | 1 | 1 | 9 | 27 | 1 | 9.9032 | 9.9657 |
| gm_Promedas_33 | 32 | 16 | 127314748520905380391777855525586135065716774604121015664758778084648831235208544136462336 | 2417851639229258349412352 | 2048 | 12.0976 | 35.7231 |
| gm_pedigree50 | 30 | 1 | 4 | 180 | 4 | 10.3609 | 12.4060 |
| lm_batch_likelihood_brackets_4_16d | 1 | 1 | 1996 | 1996 | 1 | 11.3812 | 11.6497 |
| lm_batch_likelihood_sentence_4_12d | 1 | 1 | 1900 | 1900 | 11 | 10.7582 | 10.8856 |
| lm_first_last_sentence_4_16c | 16 | 16 | 16 | 16 | 16 | 12.7569 | 12.7587 |
| mc_2020_arjun_102 | 1 | 1 | 2 | 19807040628566084398385987584 | 16 | 9.8924 | 38.9034 |
| mc_2022_085 | 1 | 1 | 2048 | 64 | 288230376151711744 | 11.2101 | 13.1523 |
| qaoaamp_N240_p3_s0 | 1267650600228229401496703205376 | 162259276829213363391578010288128 | 5575186299632655785383929568162090376495104 | 196159429230833773869868419475239575503198607639501078528 | 42535295865117307932921825928971026432 | 42.8598 | 67.9665 |
| qaoazz_N120_p5_s1 | 1 | 1 | 64 | 1 | 1024 | 9.6483 | 10.4194 |
| qaoazz_N240_p5_s0 | 1 | 1 | 35184372088832 | 1 | 4 | 8.0131 | 9.8511 |
| qc_circuit_n53_m12_s2_e8_pABCDCDAB_simplified | 1 | 1 | 1 | 2 | 32 | 9.5780 | 10.0671 |
| qc_maxcut_n27_r6_p5 | 1 | 1 | 256 | 6129982163463555433433388108601236734474956488734408704 | 2147483648 | 11.1250 | 65.9979 |
| qc_maxcut_n28_r3_p3 | 1 | 1 | 1 | 1 | 1 | 8.9559 | 9.1043 |
| qc_maxcut_n29_r4_p10 | 4 | 1 | 4951760157141521099596496896 | 5575186299632655785383929568162090376495104 | 1152921504606846976 | 11.9939 | 54.2196 |
| qc_qft_28 | 1 | 1 | 4 | 2 | 1 | 9.2071 | 9.8116 |
| qc_rand_google53_26_15 | 1 | 1 | 2147483648 | 2 | 1 | 10.3056 | 10.3846 |
| qc_rand_google53_27_20 | 1 | 1 | 2 | 9903520314283042199192993792 | 4 | 10.8879 | 39.4280 |
| qc_rand_supremacy2d_1_12_400 | 1 | 1 | 1 | 1 | 1 | 9.8228 | 10.0663 |
| qc_rand_supremacy2d_4_4_100 | 1 | 1 | 4398046511104 | 64 | 512 | 10.9575 | 13.3204 |
| qc_variational_29 | 1 | 1 | 1 | 1 | 1 | 9.9342 | 9.8092 |
| reg4_N200_s0 | 2199023255552 | 2199023255552 | 17592186044416 | 34359738368 | 1125899906842624 | 23.5065 | 21.6763 |
| reg5_N100_s0 | 524288 | 524288 | 4194304 | 1048576 | 33554432 | 16.6992 | 16.8191 |
| rnd_mixed_06 | 1 | 1 | 3840 | 5 | 6 | 10.1729 | 10.8561 |
| rnd_mixed_10 | 1 | 1 | 46438023168000000000000000 | 16 | 6 | 10.2039 | 11.7754 |
| rnd_oe_03 | 8 | 9 | 33136142011531618712492755518624197206764100255744 | 30 | 192 | 11.2998 | 13.2332 |
| rnd_oe_04 | 1 | 3 | 980465899840971486480596626642527136764465541329573080203264 | 1152 | 36 | 10.9865 | 13.4808 |
| rqc7x7_m14_s0 | 4096 | 8192 | 67108864 | 16384 | 4194304 | 14.5865 | 15.9943 |
| rqc7x7_m16_s0 | 4294967296 | 134217728 | 332306998946228968225951765070086144 | 67108864 | 137438953472 | 19.0002 | 19.2915 |
| str_ctg_lattice_01 | 6 | 210 | 360 | 48 | 2449440 | 13.8013 | 13.6698 |
| str_ctg_lattice_02 | 1 | 1 | 48 | 4 | 1 | 10.7986 | 12.4417 |
| str_ctg_regular_04 | 3 | 3 | 432 | 3 | 1296 | 11.0546 | 11.2349 |
| str_nw_mera_closed_120 | 1 | 1 | 1 | 1 | 1 | 10.3615 | 10.3615 |
| str_nw_peps_closed_333 | 1 | 1 | 6 | 4 | 1 | 9.6277 | 10.5058 |
| syc54_m12_s0 | 256 | 256 | 65536 | 8388608 | 32768 | 13.4559 | 17.9742 |
| syc54_m14_s0 | 1024 | 8192 | 562949953421312 | 16384 | 4194304 | 14.8715 | 15.1936 |
| wmc_2021_187 | 2 | 2 | 9223372036854775808 | 256 | 65536 | 10.4398 | 12.7460 |

60 s: ours fewer slices in 26, equal in 8, more in 3 of 37; unsliced (1): ours 23, cotengra 6; median log10(ctg/ours slices) = 0.60

10 s: ours fewer slices in 30, equal in 6, more in 1 of 37; unsliced (1): ours 22, cotengra 5; median log10(ctg/ours slices) = 2.41

## Executed trees: real time of all tags (s), cuTensorNet

| net | ours10 | ours60 | td_MF1 | td_hbt1 | td_exact | td_hyb300 | ctg_greedy | ctg_hyper10 | ctg_hyper60 | cutn_own |
|---|---|---|---|---|---|---|---|---|---|---|
| gm_1kp6 | 0.0052 | 0.00521 | 0.0052 | 0.00441 | 0.00459 |  | 0.00995 | 0.01333 | 0.02788 | 0.03333 |
| gm_Promedas_33 | 2.836 | 1.387 | skip_cost | 1.276 |  | 2.864 | skip_cost | skip_cost | skip_cost | 0.7966 |
| gm_pedigree50 | 1.008 | 0.01935 | 0.6362 | 1.255 |  | 0.02221 | error | 0.08338 | 2.228 | 0.02491 |
| lm_batch_likelihood_brackets_4_16d | 0.0337 | 0.0337 | 0.08384 | 0.0338 | 0.03372 |  | 0.03182 | 1.193 | 1.15 | 0.08904 |
| lm_batch_likelihood_sentence_4_12d | 0.01046 | 0.01044 | 0.01085 | 0.01046 | 0.01045 |  | 2.173 | 0.9586 | 0.9636 | 0.01207 |
| lm_first_last_sentence_4_16c | 0.6944 | 0.6853 | 0.6857 | 0.8029 | 0.6947 |  | 9.328 | 0.6932 | 0.6878 | 0.4132 |
| mc_2020_arjun_102 | 0.09468 | 0.09474 | 0.01645 | 0.09486 | 0.09477 |  | 1.186 | 0.05908 | skip_cost | error |
| mc_2022_085 | 0.1204 | 0.09878 | 0.1321 | 0.1496 |  | 3.506 | skip_cost | skip_cost | 3.603 | timeout |
| qaoaamp_N240_p3_s0 | skip_cost | skip_cost | skip_cost | skip_cost |  | skip_cost | skip_cost | skip_cost | skip_cost |  |
| qaoazz_N120_p5_s1 | 0.01019 | 0.00973 | 0.03903 | 0.01195 |  | 0.01469 | skip_cost | 4.95 | 0.00655 | 0.1214 |
| qaoazz_N240_p5_s0 | 0.00294 | 0.00251 | 0.00264 | 0.00242 |  | 0.00294 | 0.1589 | skip_cost | 0.00388 | 0.00266 |
| qc_circuit_n53_m12_s2_e8_pABCDCDAB_simplified | 0.00457 | 0.00309 | 0.022 | 0.00387 |  | 0.00389 | 6.483 | 0.03141 | 0.00743 | 0.01209 |
| qc_maxcut_n27_r6_p5 | 0.1202 | 0.1206 | skip_cost | 0.3863 |  | 0.1193 | skip_cost | skip_cost | skip_cost | 1.33 |
| qc_maxcut_n28_r3_p3 | 0.00163 | 0.00167 | 0.00452 | 0.002 |  | 0.00172 | 0.00633 | 0.00211 | 0.00195 | 0.00296 |
| qc_maxcut_n29_r4_p10 | 2.904 | 0.9427 | skip_cost | skip_cost |  | 0.9606 | skip_cost | skip_cost | skip_cost | timeout |
| qc_qft_28 | 0.02496 | 0.02497 | 0.02495 | 0.02497 | 0.02496 |  | 0.06002 | 0.1248 | 0.04321 | 0.02823 |
| qc_rand_google53_26_15 | 0.01839 | 0.01794 | 0.01199 | 0.01391 | 0.01859 |  | 0.05014 | skip_cost | 0.01967 | 0.1488 |
| qc_rand_google53_27_20 | 0.06123 | 0.06371 | 0.03709 | 0.03619 | 0.06057 |  | 0.6302 | 0.09394 | skip_cost | 0.1294 |
| qc_rand_supremacy2d_1_12_400 | 0.01967 | 0.01966 | 0.01099 | 0.01968 | 0.01964 |  | 0.02085 | 0.2661 | 0.01129 | 0.03386 |
| qc_rand_supremacy2d_4_4_100 | 0.2124 | 0.2125 | 0.8333 | 0.2125 | 0.2125 |  | skip_cost | skip_cost | 5.83 | 0.3877 |
| qc_variational_29 | 0.00508 | 0.00508 | 0.0157 | 0.00902 | 0.00903 |  | 0.00512 | 0.00508 | 0.00729 | 0.00579 |
| reg4_N200_s0 | skip_cost | skip_cost | skip_cost | skip_cost |  | skip_cost | skip_cost | skip_cost | skip_cost |  |
| reg5_N100_s0 | skip_cost | skip_cost | skip_cost | skip_cost |  | skip_cost | skip_cost | skip_cost | skip_cost |  |
| rnd_mixed_06 | 0.02057 | 0.02117 | 0.0207 | 0.02053 |  | 0.02063 | 0.1285 | skip_cost | 0.1063 | 0.6662 |
| rnd_mixed_10 | 0.02368 | 0.0226 | 0.05419 | 0.03834 |  | 0.02436 | 1.382 | skip_cost | 0.74 | 8.46 |
| rnd_oe_03 | 0.2997 | 0.1116 | 0.1382 | 0.08859 |  | 0.119 | error | skip_cost | 2.482 | 2.238 |
| rnd_oe_04 | 0.0284 | 0.06463 | 0.107 | 0.07918 |  | 0.02146 | error | skip_cost | 25.32 | 0.07469 |
| rqc7x7_m14_s0 | skip_cost | skip_cost | skip_cost | skip_cost |  | skip_cost | skip_cost | skip_cost | skip_cost |  |
| rqc7x7_m16_s0 | skip_cost | skip_cost | skip_cost | skip_cost |  | skip_cost | skip_cost | skip_cost | skip_cost |  |
| str_ctg_lattice_01 | 1.071 | skip_cost | 7.208 | 0.9794 |  | 0.9833 | skip_cost | skip_cost | skip_cost | 2.093 |
| str_ctg_lattice_02 | 0.05441 | 0.03913 | 0.07391 | 0.04972 |  | 0.04478 | 0.06318 | 2.477 | 1.023 | 0.2658 |
| str_ctg_regular_04 | 0.05938 | 0.06452 | 2.02 | 0.139 |  | 0.04811 | error | 12.16 | 0.116 | 0.04634 |
| str_nw_mera_closed_120 | 0.0045 | 0.00451 | 0.0045 | 0.00564 | 0.00561 |  | 0.00448 | 0.00442 | 0.0044 | 0.00554 |
| str_nw_peps_closed_333 | 0.00382 | 0.00363 | 0.0039 | 0.00373 |  | 0.00369 | 0.01544 | 0.02814 | 0.01843 | 0.0102 |
| syc54_m12_s0 | 51.79 | 68.61 | skip_cost | skip_cost |  | 54.36 | skip_cost | skip_cost | skip_cost | 56.57 |
| syc54_m14_s0 | skip_cost | skip_cost | skip_cost | skip_cost |  | skip_cost | skip_cost | skip_cost | skip_cost |  |
| wmc_2021_187 | 0.1366 | 0.07736 | 0.1366 | 0.1365 | 0.176 |  | skip_cost | skip_cost | 6.908 | 0.5976 |
