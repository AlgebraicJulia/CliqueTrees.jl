# Exact treewidth benchmark: the numbers GPU PID-BT must beat

Everything lives in `tn-work/pace/`. No CliqueTrees repo and nothing in `tn-work/gpupid` was modified.

## Sources (all downloads are untrusted data, never executed; they are in `data/`)

| what | URL | local |
|---|---|---|
| PACE 2017 Track A exact instances, 200 `.gr` (ex001-ex200; odd/even = public/hidden halves, the repo ships both) plus optimal `.td` | https://github.com/PACE-challenge/Treewidth-PACE-2017-instances | `data/inst2017/` |
| PACE 2017 bonus instances (100, hard; tw known for 91 by `instances.csv`, `?` otherwise) | https://github.com/PACE-challenge/Treewidth-PACE-2017-bonus-instances | `data/bonus2017/` |
| Tamaki PACE 2017 Track A winner (Java, `tw.exact.MainDecomposer`) | https://github.com/TCS-Meiji/PACE2017-TrackA | `tools/src/PACE2017-TrackA/` |
| Tamaki PACE 2016 exact solver (C) | https://github.com/TCS-Meiji/treewidth-exact | `tools/src/treewidth-exact/` |
| Tamaki 2022 `twalgor/tw` (Java; `UpLow` = anytime upper+lower bound, needs 2 threads) | https://github.com/twalgor/tw | `tools/src/tw/` |
| td-validate (Holger Dell's PACE validator; the copy bundled in PACE2017-TrackA fails to compile on modern g++) | https://github.com/holgerdell/td-validate | `tools/src/td-validate/` |

PACE 2016 exact instances: not found as a public repo (pacechallenge.org/2016 links only to the PACE-challenge org, which has no 2016 treewidth instance repo), so skipped. Only the 2016 solver was tried, on the 2017 set.

Decompressed copies (`xz -d`) are in `data/plain/{exact,bonus,exact_td,bonus_td}/`; the runners read only these.

## What was tried to build

- Tamaki17: `javac` (needs JDK; none was installed, so a Temurin 21 JDK was unpacked into `tools/jdk/`). OK. Official `tw-exact` script uses `-Xmx30g`; we run `-Xmx8g -Xss10m`, serial GC, `-XX:ActiveProcessorCount=1`.
- td-validate: bundled copy lacks `#include <limits>`; holger's repo copy built with `CXXFLAGS="-std=c++11 -O3 -include limits"`. OK.
- Tamaki16 (C): fails to link on modern gcc (duplicate globals) -> built with `-fcommon`. OK, but weak (see below).
- twalgor/tw (2022): built with `javac`. `ExactTW` did not finish ex001 in 100 s (with or without `-acsd`), so only `UpLow` (the anytime bounds tool; solved = bounds meet) is used as the "newer Tamaki" baseline.
- CliqueTrees septw: `tools/ctrun.jl` (details below). One thread (`-t1`, `nt=1`), unit weights, per connected component (septw assumes connected input), shared deadline; the td is built from the elimination order and validated with td-validate. Julia warm-up is excluded from the time, load time is not.

## How to run

```
cd tn-work/pace
./run_local.sh                      # 30 s, 4 at a time: tamaki17, septw (exact+bonus), tamaki16, uplow22 -> results/*.tsv
T=30 python3 -I tools/run.py septw exact 30 4 results      # one solver/set; ONLY=ex001,ex002 restricts
python3 -I tools/report.py results  # markdown tables + summary (this README's tables)
julia --project=../mtenv tools/ctrun.jl 30 in.gr out.td     # single instance, prints "FOUND w secs"
```
Status `OK` = finished with a td that td-validate accepts (and, for the exact set, width equals the published optimum: column "septw width ok?"). `TO` = not finished in the limit. For UpLow, `TO (ub w)` gives the best upper bound written in time.

HPG (written; submitted Oct 19, see "GPU PID-BT and HPG jobs" below): `hpg_cpu.sbatch`, hpg-milan, 1800 s per instance, `--job-name=itay-ct --dependency=singleton`, logs/results under `/blue/fairbanksj/itaykadosh/gpu/pace/`. Sync `tn-work/pace/{tools,data/plain}` (and the JDK) there first, then `sbatch hpg_cpu.sbatch septw bonus` (also `tamaki17 bonus`, `... exact`). Uses the ct-mt env path from `gpupid/hpg/cpu_base.sbatch` (`envs/mt-65e72a3`); check it still points at the intended CliqueTrees commit, and that the 32 workers sharing a node do not skew timing (set `--cpus-per-task` lower if so). Note `--dependency=singleton` serializes jobs of the same name: submit one solver/set at a time.

## Headline (30 s, 1 thread)

| set | Tamaki17 | septw | UpLow22 | Tamaki16 |
|---|---|---|---|---|
| exact (200) | 173 | 134 | 128 | 2 (ex003 crashes: "Trie area exhausted") |
| bonus (100) | 7 | 21 | not run | not run |

Tamaki17 beats septw on the PACE set (54 instances only Tamaki17, 15 only septw, 12 neither); septw beats Tamaki17 on the bonus set (14 only septw, 0 only Tamaki17, 79 neither). Every OK result passed td-validate and every septw width equals the published optimum.

## Results at 30 s, one thread, shared 16-core laptop (4 concurrent jobs)

Caveats: wall-clock on a shared laptop, so times are noisy (septw time is measured in-process, so it excludes Julia startup and warm-up; Tamaki times are wall-clock including JVM start); Tamaki17 uses `-Xmx8g` not 30g; JIT warm-up counts for Java.

## Set `exact` (200 instances)

| instance | n | m | tw (known) | Tamaki17 time | septw time | Tamaki16 time | UpLow22 time | septw width ok? |
|---|---|---|---|---|---|---|---|---|
| ex001 | 262 | 648 | 10 | 1.1s | 11.0s | TO | 3.6s | y |
| ex002 | 145 | 2368 | 49 | 1.0s | 0.3s | TO | TO (ub 50) | y |
| ex003 | 92 | 2113 | 44 | 10.9s | TO | FAIL | 0.8s |  |
| ex004 | 490 | 118403 | 486 | 1.8s | 2.0s | 25.6s | 8.5s | y |
| ex005 | 377 | 597 | 7 | 13.3s | 26.0s | TO | 19.0s | y |
| ex006 | 370 | 561 | 7 | 6.9s | 22.3s | TO | 10.1s | y |
| ex007 | 137 | 451 | 12 | 0.3s | 0.1s | TO | 2.7s | y |
| ex008 | 170 | 396 | 10 | 1.0s | 1.9s | TO | 3.2s | y |
| ex009 | 466 | 662 | 7 | 11.8s | TO | TO | 6.3s |  |
| ex010 | 251 | 430 | 9 | 0.8s | 8.7s | TO | 3.4s | y |
| ex011 | 465 | 1004 | 9 | 1.0s | 3.3s | TO | 4.3s | y |
| ex012 | 365 | 878 | 12 | 1.0s | 1.2s | TO | 6.5s | y |
| ex013 | 56 | 280 | 29 | 16.8s | 0.7s | TO | TO (ub 29) | y |
| ex014 | 370 | 1767 | 18 | 10.7s | TO | TO | 12.2s |  |
| ex015 | 177 | 669 | 15 | 0.3s | 0.1s | TO | 4.3s | y |
| ex016 | 275 | 438 | 8 | TO | TO | TO | TO (ub 8) |  |
| ex017 | 330 | 571 | 9 | 1.7s | 17.2s | TO | 4.9s | y |
| ex018 | 266 | 460 | 9 | 2.3s | 8.2s | TO | 9.7s | y |
| ex019 | 291 | 752 | 11 | TO | 5.6s | TO | 25.4s | y |
| ex020 | 106 | 558 | 20 | 4.0s | 0.2s | TO | TO (ub 20) | y |
| ex021 | 318 | 572 | 9 | 2.8s | 14.8s | TO | 0.7s | y |
| ex022 | 116 | 522 | 16 | 1.6s | 1.0s | TO | 2.6s | y |
| ex023 | 690 | 1355 | 8 | 1.8s | TO | TO | 14.4s |  |
| ex024 | 116 | 522 | 16 | 2.5s | 1.0s | TO | TO (ub 16) | y |
| ex025 | 92 | 472 | 20 | 2.1s | 0.1s | TO | TO (ub 20) | y |
| ex026 | 176 | 308 | 9 | 1.7s | 8.0s | TO | 12.0s | y |
| ex027 | 274 | 715 | 11 | TO | 3.4s | TO | TO (ub 12) | y |
| ex028 | 198 | 352 | 9 | 1.9s | 1.9s | TO | 1.7s | y |
| ex029 | 238 | 411 | 9 | 1.5s | 22.9s | TO | 2.4s | y |
| ex030 | 404 | 607 | 7 | 22.0s | TO | TO | 10.6s |  |
| ex031 | 219 | 382 | 8 | 9.4s | 9.8s | TO | TO (ub 8) | y |
| ex032 | 140 | 336 | 11 | 0.6s | 0.4s | TO | 1.8s | y |
| ex033 | 363 | 541 | 7 | TO | TO | TO | 8.5s |  |
| ex034 | 285 | 735 | 11 | TO | 3.7s | TO | TO (ub 11) | y |
| ex035 | 247 | 804 | 14 | 3.8s | 1.5s | TO | 3.4s | y |
| ex036 | 226 | 8013 | 119 | 0.3s | 0.4s | TO | 0.3s | y |
| ex037 | 272 | 615 | 10 | 3.8s | 7.2s | TO | 1.0s | y |
| ex038 | 71 | 925 | 26 | 0.3s | 0.3s | TO | 0.2s | y |
| ex039 | 56 | 280 | 32 | TO | 1.9s | TO | TO (ub 32) | y |
| ex040 | 285 | 526 | 9 | 6.5s | 8.8s | TO | 3.8s | y |
| ex041 | 205 | 341 | 9 | 1.1s | 12.9s | TO | 8.6s | y |
| ex042 | 278 | 522 | 9 | 3.2s | 6.3s | TO | 12.1s | y |
| ex043 | 279 | 513 | 9 | 3.7s | 16.2s | TO | 8.4s | y |
| ex044 | 1969 | 4228 | 6 | 3.9s | 0.3s | TO | 0.5s | y |
| ex045 | 600 | 865 | 7 | 8.5s | TO | TO | TO (ub 7) |  |
| ex046 | 207 | 368 | 9 | 2.1s | 10.9s | TO | 8.4s | y |
| ex047 | 1854 | 21118 | 21 | 16.4s | 4.6s | TO | TO (ub 25) | y |
| ex048 | 238 | 1077 | 15 | 0.3s | 0.0s | TO | 0.1s | y |
| ex049 | 117 | 332 | 13 | 0.4s | 0.2s | TO | 2.4s | y |
| ex050 | 66 | 838 | 28 | 0.3s | 0.4s | TO | 0.7s | y |
| ex051 | 136 | 254 | 10 | 1.1s | 1.5s | TO | 9.9s | y |
| ex052 | 244 | 717 | 9 | 0.5s | 0.1s | TO | TO (ub 9) | y |
| ex053 | 218 | 383 | 9 | 1.8s | 14.4s | TO | 4.2s | y |
| ex054 | 243 | 428 | 9 | 2.0s | 8.4s | TO | 1.2s | y |
| ex055 | 197 | 813 | 18 | 0.4s | 2.1s | TO | 14.7s | y |
| ex056 | 114 | 584 | 34 | 7.1s | 3.3s | TO | TO (ub 34) | y |
| ex057 | 281 | 9075 | 117 | 0.3s | 3.3s | TO | 0.2s | y |
| ex058 | 152 | 492 | 10 | 0.6s | 0.1s | TO | 4.1s | y |
| ex059 | 298 | 780 | 10 | 0.9s | 3.7s | TO | 3.7s | y |
| ex060 | 249 | 434 | 10 | 1.3s | TO | TO | 11.7s |  |
| ex061 | 158 | 1058 | 22 | 10.5s | 0.2s | TO | TO (ub 22) | y |
| ex062 | 99 | 549 | 18 | 0.5s | 0.3s | TO | 1.6s | y |
| ex063 | 103 | 582 | 34 | 4.2s | 1.9s | TO | TO (ub 34) | y |
| ex064 | 589 | 825 | 7 | TO | TO | TO | 14.8s |  |
| ex065 | 50 | 175 | 25 | TO | 2.0s | TO | TO (ub 25) | y |
| ex066 | 136 | 471 | 15 | 2.0s | 0.4s | TO | 6.0s | y |
| ex067 | 235 | 424 | 10 | 2.9s | 12.4s | TO | 1.7s | y |
| ex068 | 115 | 354 | 8 | 0.3s | 0.0s | TO | 2.2s | y |
| ex069 | 235 | 441 | 9 | 1.7s | 15.0s | TO | 2.8s | y |
| ex070 | 48 | 96 | 8 | 0.8s | 0.0s | 1.7s | 0.9s | y |
| ex071 | 253 | 434 | 9 | 2.8s | TO | TO | 8.7s |  |
| ex072 | 221 | 383 | 9 | 2.3s | TO | TO | 1.7s |  |
| ex073 | 712 | 1085 | 7 | 12.4s | TO | TO | 7.1s |  |
| ex074 | 359 | 6166 | 47 | 4.6s | 2.6s | TO | TO (ub 48) | y |
| ex075 | 111 | 360 | 8 | 0.5s | 0.0s | TO | 0.9s | y |
| ex076 | 102 | 446 | 17 | 1.4s | 1.0s | TO | 5.7s | y |
| ex077 | 237 | 423 | 10 | 2.9s | 9.3s | TO | 13.1s | y |
| ex078 | 256 | 465 | 9 | 7.6s | TO | TO | 12.2s |  |
| ex079 | 314 | 4943 | 42 | 2.2s | 2.0s | TO | TO (ub 44) | y |
| ex080 | 329 | 573 | 9 | 3.0s | TO | TO | 6.1s |  |
| ex081 | 188 | 638 | 6 | 0.7s | 0.0s | TO | 0.1s | y |
| ex082 | 193 | 860 | 16 | 6.5s | 3.0s | TO | 3.9s | y |
| ex083 | 213 | 380 | 10 | 3.5s | 7.1s | TO | 7.8s | y |
| ex084 | 161 | 4431 | 70 | 4.1s | 0.9s | TO | TO (ub 71) | y |
| ex085 | 229 | 370 | 8 | 10.0s | 16.0s | TO | TO (ub 8) | y |
| ex086 | 102 | 511 | 31 | 3.8s | 2.5s | TO | TO (ub 31) | y |
| ex087 | 380 | 5790 | 47 | TO | 2.6s | TO | TO (ub 47) | y |
| ex088 | 387 | 5827 | 47 | 23.4s | 3.5s | TO | TO (ub 48) | y |
| ex089 | 318 | 576 | 9 | 7.7s | TO | TO | 2.4s |  |
| ex090 | 201 | 365 | 11 | 4.6s | 1.5s | TO | 5.1s | y |
| ex091 | 193 | 336 | 9 | 25.5s | 12.1s | TO | 10.7s | y |
| ex092 | 377 | 6788 | 53 | 15.8s | 4.1s | TO | TO (ub 53) | y |
| ex093 | 454 | 664 | 7 | 29.9s | 23.7s | TO | 20.9s | y |
| ex094 | 257 | 453 | 11 | 12.0s | 12.6s | TO | 2.5s | y |
| ex095 | 220 | 555 | 11 | 1.0s | 14.5s | TO | 24.1s | y |
| ex096 | 119 | 415 | 9 | 0.5s | 0.0s | TO | TO (ub 9) | y |
| ex097 | 286 | 4079 | 48 | 2.6s | 1.1s | TO | TO (ub 48) | y |
| ex098 | 342 | 588 | 9 | 1.9s | TO | TO | TO (ub 9) |  |
| ex099 | 616 | 923 | 7 | TO | TO | TO | TO (ub 7) |  |
| ex100 | 170 | 415 | 12 | 5.3s | 2.6s | TO | TO (ub 14) | y |
| ex101 | 1038 | 291034 | 540 | 12.5s | 8.1s | TO | 14.2s | y |
| ex102 | 291 | 5212 | 54 | 10.4s | 3.0s | TO | TO (ub 58) | y |
| ex103 | 237 | 419 | 10 | 3.0s | 21.0s | TO | 8.9s | y |
| ex104 | 1038 | 291037 | 540 | 13.3s | 7.3s | TO | 15.8s | y |
| ex105 | 1038 | 291037 | 540 | 13.4s | 10.3s | TO | 14.5s | y |
| ex106 | 166 | 4455 | 70 | 4.7s | 1.2s | TO | TO (ub 71) | y |
| ex107 | 166 | 396 | 12 | 2.3s | 2.8s | TO | 7.6s | y |
| ex108 | 194 | 603 | 11 | 2.1s | 1.1s | TO | 22.9s | y |
| ex109 | 1212 | 1794 | 7 | TO | TO | TO | 12.1s |  |
| ex110 | 254 | 355 | 8 | TO | TO | TO | 24.1s |  |
| ex111 | 395 | 668 | 9 | 4.9s | TO | TO | 9.9s |  |
| ex112 | 318 | 559 | 10 | 5.6s | 21.8s | TO | 14.3s | y |
| ex113 | 93 | 488 | 14 | 0.2s | 0.0s | TO | 1.0s | y |
| ex114 | 268 | 456 | 10 | 2.9s | TO | TO | 22.1s |  |
| ex115 | 963 | 419877 | 908 | 16.6s | 14.2s | TO | TO | y |
| ex116 | 453 | 742 | 9 | 3.8s | TO | TO | 11.0s |  |
| ex117 | 77 | 181 | 13 | 20.2s | 1.2s | TO | TO (ub 13) | y |
| ex118 | 81 | 810 | 54 | TO | 7.3s | TO | TO (ub 54) | y |
| ex119 | 84 | 479 | 23 | 15.4s | 2.1s | TO | TO (ub 26) | y |
| ex120 | 188 | 318 | 9 | TO | 15.5s | TO | TO (ub 10) | y |
| ex121 | 204 | 1164 | 34 | TO | 14.5s | TO | TO (ub 34) | y |
| ex122 | 131 | 4093 | 76 | 0.3s | 4.4s | TO | 0.2s | y |
| ex123 | 122 | 635 | 35 | 16.2s | 5.0s | TO | TO (ub 35) | y |
| ex124 | 496 | 868 | 10 | 12.3s | TO | TO | 9.3s |  |
| ex125 | 320 | 8862 | 70 | 7.1s | 14.9s | TO | TO (ub 71) | y |
| ex126 | 440 | 766 | 9 | 4.9s | TO | TO | 5.2s |  |
| ex127 | 228 | 527 | 10 | 0.5s | 0.1s | TO | 1.8s | y |
| ex128 | 194 | 2513 | 28 | 0.7s | 8.4s | TO | 2.8s | y |
| ex129 | 737 | 2826 | 14 | 1.4s | TO | TO | TO (ub 14) |  |
| ex130 | 293 | 1647 | 19 | 0.9s | 5.9s | TO | TO (ub 19) | y |
| ex131 | 292 | 1386 | 18 | 0.7s | 8.6s | TO | 22.3s | y |
| ex132 | 392 | 1626 | 18 | 1.2s | TO | TO | 27.2s |  |
| ex133 | 522 | 1296 | 11 | 3.6s | TO | TO | TO (ub 11) |  |
| ex134 | 345 | 540 | 8 | TO | 16.7s | TO | TO (ub 8) | y |
| ex135 | 2822 | 129474 | 87 | TO | TO | TO | 1.4s |  |
| ex136 | 338 | 1752 | 34 | TO | TO | TO | TO (ub 34) |  |
| ex137 | 196 | 1098 | 19 | 1.1s | 2.2s | TO | 1.9s | y |
| ex138 | 209 | 2775 | 27 | 0.8s | 6.3s | TO | TO (ub 27) | y |
| ex139 | 334 | 568 | 9 | 7.6s | TO | TO | TO (ub 9) |  |
| ex140 | 258 | 467 | 10 | 13.0s | TO | TO | 19.2s |  |
| ex141 | 226 | 1168 | 34 | TO | 17.6s | TO | TO (ub 34) | y |
| ex142 | 392 | 726 | 10 | 4.4s | TO | TO | 12.8s |  |
| ex143 | 130 | 660 | 35 | TO | 8.9s | TO | TO (ub 35) | y |
| ex144 | 257 | 455 | 10 | 10.0s | TO | TO | TO (ub 11) |  |
| ex145 | 48 | 96 | 12 | 20.7s | 15.4s | TO | TO (ub 12) | y |
| ex146 | 152 | 377 | 12 | 22.0s | 2.4s | TO | 5.7s | y |
| ex147 | 101 | 606 | 16 | 0.3s | 0.0s | TO | 0.4s | y |
| ex148 | 746 | 2784 | 12 | 1.5s | TO | TO | 3.1s |  |
| ex149 | 698 | 2604 | 12 | 1.3s | TO | TO | 4.6s |  |
| ex150 | 839 | 27224 | 117 | 1.5s | TO | TO | 0.4s |  |
| ex151 | 279 | 733 | 12 | TO | 9.5s | TO | TO (ub 13) | y |
| ex152 | 770 | 2874 | 12 | 1.5s | TO | TO | 2.1s |  |
| ex153 | 772 | 11654 | 47 | TO | TO | TO | TO (ub 48) |  |
| ex154 | 570 | 8158 | 48 | 28.9s | 26.2s | TO | TO (ub 49) | y |
| ex155 | 758 | 11580 | 47 | TO | TO | TO | TO (ub 47) |  |
| ex156 | 264 | 630 | 11 | 5.9s | TO | TO | 2.2s |  |
| ex157 | 260 | 467 | 9 | 4.2s | TO | TO | 2.1s |  |
| ex158 | 596 | 1413 | 11 | 3.2s | TO | TO | 1.2s |  |
| ex159 | 582 | 2772 | 18 | 3.2s | TO | TO | 23.9s |  |
| ex160 | 320 | 846 | 12 | 0.7s | TO | TO | 5.2s |  |
| ex161 | 1046 | 3906 | 12 | 2.9s | TO | TO | 2.5s |  |
| ex162 | 755 | 1302 | 9 | 7.3s | TO | TO | TO (ub 9) |  |
| ex163 | 244 | 445 | 10 | 5.0s | 25.6s | TO | 12.7s | y |
| ex164 | 402 | 1350 | 14 | 0.9s | TO | TO | 11.0s |  |
| ex165 | 222 | 742 | 14 | 0.6s | 2.9s | TO | 7.0s | y |
| ex166 | 217 | 402 | 10 | 2.0s | 14.4s | TO | TO (ub 10) | y |
| ex167 | 509 | 969 | 10 | 7.2s | TO | TO | TO (ub 10) |  |
| ex168 | 572 | 1923 | 14 | 1.1s | TO | TO | TO (ub 14) |  |
| ex169 | 3706 | 42236 | 22 | TO | TO | TO | TO (ub 27) |  |
| ex170 | 361 | 628 | 9 | 3.3s | TO | TO | 7.2s |  |
| ex171 | 647 | 2175 | 14 | 1.3s | TO | TO | 9.0s |  |
| ex172 | 114 | 1363 | 32 | 6.0s | 0.6s | TO | TO (ub 32) | y |
| ex173 | 536 | 1011 | 10 | 5.7s | TO | TO | 14.2s |  |
| ex174 | 350 | 1665 | 24 | 0.3s | TO | TO | TO (ub 24) |  |
| ex175 | 227 | 1000 | 17 | TO | 11.2s | TO | TO (ub 18) | y |
| ex176 | 743 | 1302 | 10 | 25.8s | TO | TO | 5.6s |  |
| ex177 | 227 | 759 | 14 | 0.8s | 2.6s | TO | 10.6s | y |
| ex178 | 695 | 1233 | 10 | 6.9s | TO | TO | TO (ub 10) |  |
| ex179 | 187 | 346 | 10 | 10.4s | TO | TO | 15.8s |  |
| ex180 | 479 | 13293 | 70 | 10.0s | 27.8s | TO | TO (ub 71) | y |
| ex181 | 109 | 732 | 18 | 0.6s | 0.0s | TO | 0.7s | y |
| ex182 | 394 | 746 | 10 | 14.5s | TO | TO | 12.7s |  |
| ex183 | 265 | 471 | 11 | 9.0s | 22.7s | TO | 24.4s | y |
| ex184 | 749 | 1290 | 10 | 6.7s | TO | TO | 23.6s |  |
| ex185 | 237 | 793 | 14 | 0.9s | 2.4s | TO | 11.3s | y |
| ex186 | 316 | 563 | 10 | 15.6s | TO | TO | TO (ub 10) |  |
| ex187 | 240 | 453 | 10 | 2.9s | 26.5s | TO | 11.1s | y |
| ex188 | 140 | 1850 | 27 | 0.5s | 5.2s | TO | TO (ub 27) | y |
| ex189 | 178 | 4517 | 70 | 3.5s | 1.1s | TO | TO (ub 71) | y |
| ex190 | 296 | 1131 | 15 | 0.8s | 5.9s | TO | 16.8s | y |
| ex191 | 492 | 1608 | 15 | 17.1s | TO | TO | TO (ub 15) |  |
| ex192 | 53 | 258 | 29 | TO | 2.9s | TO | TO (ub 29) | y |
| ex193 | 1391 | 3012 | 10 | 3.2s | TO | TO | TO (ub 10) |  |
| ex194 | 782 | 1944 | 11 | 4.8s | TO | TO | 7.4s |  |
| ex195 | 216 | 382 | 10 | 5.9s | 26.4s | TO | 13.0s | y |
| ex196 | 242 | 443 | 11 | 9.4s | 26.7s | TO | 4.6s | y |
| ex197 | 303 | 1158 | 15 | 0.9s | 6.3s | TO | 12.6s | y |
| ex198 | 3104 | 142422 | 87 | TO | TO | TO | 1.9s |  |
| ex199 | 310 | 537 | 9 | 16.6s | TO | TO | TO (ub 9) |  |
| ex200 | 270 | 942 | 16 | 5.1s | 15.7s | TO | 15.3s | y |

### Summary, `exact` (solved = optimal TD found and validated)

- Tamaki17: 173 / 200 solved
- septw: 134 / 200 solved
- Tamaki16: 2 / 200 solved
- UpLow22: 128 / 200 solved
- Neither Tamaki17 nor septw solves (12): ex016, ex033, ex064, ex099, ex109, ex110, ex135, ex136, ex153, ex155, ex169, ex198
- Only Tamaki17 (54): ex003, ex009, ex014, ex023, ex030, ex045, ex060, ex071, ex072, ex073, ex078, ex080, ex089, ex098, ex111, ex114, ex116, ex124, ex126, ex129, ex132, ex133, ex139, ex140, ex142, ex144, ex148, ex149, ex150, ex152, ex156, ex157, ex158, ex159, ex160, ex161, ex162, ex164, ex167, ex168, ex170, ex171, ex173, ex174, ex176, ex178, ex179, ex182, ex184, ex186, ex191, ex193, ex194, ex199
- Only septw (15): ex019, ex027, ex034, ex039, ex065, ex087, ex118, ex120, ex121, ex134, ex141, ex143, ex151, ex175, ex192

## Set `bonus` (100 instances)

| instance | n | m | tw (known) | Tamaki17 time | septw time | septw width ok? |
|---|---|---|---|---|---|---|
| 6s151.gaifman_3 | 253 | 634 | 14 | TO | 27.4s | y |
| FLA_13 | 280 | 456 | 9 | TO | TO |  |
| FLA_14 | 266 | 423 | 8 | 29.3s | 22.6s | y |
| FLA_15 | 325 | 522 | 9 | TO | TO |  |
| GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_10 | 187 | 385 | 14 | TO | TO |  |
| GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_11 | 192 | 395 | 14 | TO | TO |  |
| GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_12 | 197 | 404 | 14 | TO | TO |  |
| GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_13 | 190 | 390 | 14 | TO | TO |  |
| GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_15 | 198 | 406 | 14 | TO | TO |  |
| GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_9 | 143 | 303 | 13 | TO | 16.4s | y |
| GTFS_VBB_EndeApr_Dez2016.zip_train+metro_12 | 103 | 212 | 11 | 7.6s | 0.8s | y |
| GTFS_VBB_EndeApr_Dez2016.zip_train+metro_14 | 123 | 248 | 13 | TO | 19.0s | y |
| GTFS_VBB_EndeApr_Dez2016.zip_train+metro_15 | 124 | 250 | 13 | TO | 18.9s | y |
| LKS_13 | 293 | 484 | 9 | TO | TO |  |
| LKS_15 | 220 | 385 | 10 | TO | TO |  |
| MD5-32-1.gaifman_4 | 225 | 705 | 12 | 2.2s | 2.1s | y |
| NY_11 | 226 | 369 | 10 | TO | TO |  |
| NY_13 | 283 | 448 | 9 | TO | TO |  |
| Pedigree_11_6 | 205 | 503 | 14 | TO | TO |  |
| Pedigree_11_7 | 202 | 501 | 14 | TO | TO |  |
| Pedigree_12_10 | 286 | 712 | 16 | TO | TO |  |
| Pedigree_12_12 | 277 | 683 | 16 | TO | TO |  |
| Pedigree_12_14 | 284 | 703 | 15 | TO | TO |  |
| Pedigree_12_8 | 217 | 531 | 14 | TO | TO |  |
| Pedigree_13_12 | 264 | 646 | 15 | TO | TO |  |
| Pedigree_13_9 | 268 | 665 | 16 | TO | TO |  |
| Promedas_11_7 | 191 | 385 | 13 | TO | TO |  |
| Promedas_21_9 | 253 | 486 | 11 | TO | TO |  |
| Promedas_22_6 | 200 | 415 | 12 | TO | 29.2s | y |
| Promedas_22_8 | 224 | 441 | 13 | TO | TO |  |
| Promedas_23_6 | 253 | 500 | 12 | TO | TO |  |
| Promedas_24_11 | 273 | 494 | 12 | TO | TO |  |
| Promedas_25_8 | 204 | 378 | 11 | TO | TO |  |
| Promedas_27_8 | 165 | 323 | 12 | TO | TO |  |
| Promedas_28_10 | 333 | 605 | 11 | TO | TO |  |
| Promedas_30_7 | 164 | 320 | 13 | TO | TO |  |
| Promedas_31_8 | 376 | 740 | ? | TO | TO |  |
| Promedas_32_8 | 238 | 487 | 13 | TO | TO |  |
| Promedas_34_8 | 174 | 348 | 14 | TO | TO |  |
| Promedas_43_13 | 197 | 354 | 10 | TO | TO |  |
| Promedas_44_11 | 362 | 691 | ? | TO | TO |  |
| Promedas_44_9 | 276 | 534 | 12 | TO | TO |  |
| Promedas_45_7 | 159 | 313 | 12 | TO | TO |  |
| Promedas_46_15 | 227 | 416 | 13 | TO | TO |  |
| Promedas_46_8 | 175 | 318 | 11 | TO | TO |  |
| Promedas_48_5 | 134 | 278 | 11 | 19.5s | 6.8s | y |
| Promedas_49_8 | 184 | 367 | 10 | TO | 11.3s | y |
| Promedas_50_7 | 175 | 362 | 12 | TO | TO |  |
| Promedas_51_12 | 230 | 431 | 13 | TO | TO |  |
| Promedas_51_14 | 406 | 760 | ? | TO | TO |  |
| Promedas_55_9 | 221 | 425 | 13 | TO | TO |  |
| Promedas_56_8 | 155 | 299 | 10 | 9.5s | 8.6s | y |
| Promedas_58_11 | 359 | 654 | ? | TO | TO |  |
| Promedas_59_10 | 209 | 396 | 11 | TO | TO |  |
| Promedas_60_11 | 216 | 387 | 11 | TO | TO |  |
| Promedas_61_8 | 156 | 305 | 13 | TO | TO |  |
| Promedas_62_11 | 306 | 625 | ? | TO | TO |  |
| Promedas_62_9 | 217 | 427 | 13 | TO | TO |  |
| Promedas_63_8 | 181 | 374 | 14 | TO | TO |  |
| Promedas_68_13 | 420 | 749 | ? | TO | TO |  |
| Promedas_69_10 | 194 | 379 | 12 | TO | TO |  |
| Promedas_69_9 | 133 | 251 | 9 | 3.2s | 2.8s | y |
| Promedus_11_15 | 247 | 497 | 13 | TO | TO |  |
| Promedus_12_14 | 272 | 494 | 11 | TO | TO |  |
| Promedus_12_15 | 293 | 533 | 11 | TO | TO |  |
| Promedus_14_8 | 199 | 417 | 14 | TO | TO |  |
| Promedus_14_9 | 173 | 357 | 12 | TO | TO |  |
| Promedus_17_13 | 180 | 349 | 13 | TO | TO |  |
| Promedus_18_10 | 187 | 397 | 14 | TO | TO |  |
| Promedus_18_8 | 195 | 411 | 13 | TO | TO |  |
| Promedus_20_13 | 193 | 353 | 12 | TO | TO |  |
| Promedus_27_15 | 189 | 353 | 13 | TO | TO |  |
| Promedus_28_14 | 193 | 351 | 11 | TO | TO |  |
| Promedus_34_11 | 157 | 289 | 11 | TO | TO |  |
| Promedus_34_12 | 210 | 389 | 11 | TO | TO |  |
| Promedus_34_14 | 188 | 352 | 12 | TO | TO |  |
| Promedus_38_14 | 242 | 462 | 10 | TO | TO |  |
| Promedus_38_15 | 208 | 398 | 10 | TO | TO |  |
| SAT_dat.k80-24_1_rule_1.gaifman_3 | 130 | 698 | 22 | TO | 8.5s | y |
| Sz512_15127_1.smt2-stp212.gaifman_3 | 175 | 593 | 14 | 1.8s | 1.1s | y |
| aes_24_4_keyfind_5.gaifman_3 | 104 | 380 | 23 | TO | TO |  |
| aes_24_4_keyfind_5.gaifman_4 | 154 | 874 | ? | TO | TO |  |
| am_7_7.shuffled-as.sat03-363.gaifman_6 | 189 | 424 | 14 | TO | TO |  |
| am_9_9.gaifman_6 | 212 | 480 | 15 | TO | TO |  |
| atco_enc2_opt2_10_21.gaifman_3 | 104 | 797 | ? | TO | TO |  |
| countbitsarray04_32.gaifman_10 | 331 | 843 | 13 | TO | TO |  |
| jgiraldezlevy.2200.9086.08.40.167.gaifman_2 | 92 | 552 | 34 | TO | 21.4s | y |
| jgiraldezlevy.2200.9086.08.40.22.gaifman_2 | 111 | 675 | 33 | TO | 24.3s | y |
| jgiraldezlevy.2200.9086.08.40.41.gaifman_2 | 95 | 568 | 34 | TO | 6.0s | y |
| jgiraldezlevy.2200.9086.08.40.46.gaifman_2 | 105 | 658 | 33 | TO | 27.2s | y |
| jgiraldezlevy.2200.9086.08.40.93.gaifman_2 | 100 | 593 | 36 | TO | TO |  |
| minxor128.gaifman_2 | 231 | 606 | 4 | TO | 0.0s | y |
| modgen-n200-m90860q08c40-14808.gaifman_2 | 112 | 686 | 35 | TO | TO |  |
| modgen-n200-m90860q08c40-1585.gaifman_2 | 138 | 686 | ? | TO | TO |  |
| modgen-n200-m90860q08c40-22556.gaifman_2 | 135 | 855 | 33 | TO | TO |  |
| mrpp_4x4#8_8.gaifman_3 | 106 | 589 | 24 | TO | 2.5s | y |
| mrpp_8x8#24_14.gaifman_3 | 140 | 856 | 28 | TO | TO |  |
| newton.3.3.i.smt2-stp212.gaifman_2 | 119 | 459 | 19 | TO | 4.2s | y |
| post-cbmc-aes-d-r2.gaifman_10 | 263 | 505 | 11 | TO | TO |  |
| smtlib-qfbv-aigs-lfsr_004_127_112-tseitin.gaifman_6 | 316 | 669 | 13 | TO | TO |  |

### Summary, `bonus` (solved = optimal TD found and validated)

- Tamaki17: 7 / 100 solved
- septw: 21 / 100 solved
- Neither Tamaki17 nor septw solves (79): FLA_13, FLA_15, GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_10, GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_11, GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_12, GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_13, GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_15, LKS_13, LKS_15, NY_11, NY_13, Pedigree_11_6, Pedigree_11_7, Pedigree_12_10, Pedigree_12_12, Pedigree_12_14, Pedigree_12_8, Pedigree_13_12, Pedigree_13_9, Promedas_11_7, Promedas_21_9, Promedas_22_8, Promedas_23_6, Promedas_24_11, Promedas_25_8, Promedas_27_8, Promedas_28_10, Promedas_30_7, Promedas_31_8, Promedas_32_8, Promedas_34_8, Promedas_43_13, Promedas_44_11, Promedas_44_9, Promedas_45_7, Promedas_46_15, Promedas_46_8, Promedas_50_7, Promedas_51_12, Promedas_51_14, Promedas_55_9, Promedas_58_11, Promedas_59_10, Promedas_60_11, Promedas_61_8, Promedas_62_11, Promedas_62_9, Promedas_63_8, Promedas_68_13, Promedas_69_10, Promedus_11_15, Promedus_12_14, Promedus_12_15, Promedus_14_8, Promedus_14_9, Promedus_17_13, Promedus_18_10, Promedus_18_8, Promedus_20_13, Promedus_27_15, Promedus_28_14, Promedus_34_11, Promedus_34_12, Promedus_34_14, Promedus_38_14, Promedus_38_15, aes_24_4_keyfind_5.gaifman_3, aes_24_4_keyfind_5.gaifman_4, am_7_7.shuffled-as.sat03-363.gaifman_6, am_9_9.gaifman_6, atco_enc2_opt2_10_21.gaifman_3, countbitsarray04_32.gaifman_10, jgiraldezlevy.2200.9086.08.40.93.gaifman_2, modgen-n200-m90860q08c40-14808.gaifman_2, modgen-n200-m90860q08c40-1585.gaifman_2, modgen-n200-m90860q08c40-22556.gaifman_2, mrpp_8x8#24_14.gaifman_3, post-cbmc-aes-d-r2.gaifman_10, smtlib-qfbv-aigs-lfsr_004_127_112-tseitin.gaifman_6
- Only Tamaki17 (0): 
- Only septw (14): 6s151.gaifman_3, GTFS_VBB_EndeApr_Dez2016.zip_train+metro+tram_9, GTFS_VBB_EndeApr_Dez2016.zip_train+metro_14, GTFS_VBB_EndeApr_Dez2016.zip_train+metro_15, Promedas_22_6, Promedas_49_8, SAT_dat.k80-24_1_rule_1.gaifman_3, jgiraldezlevy.2200.9086.08.40.167.gaifman_2, jgiraldezlevy.2200.9086.08.40.22.gaifman_2, jgiraldezlevy.2200.9086.08.40.41.gaifman_2, jgiraldezlevy.2200.9086.08.40.46.gaifman_2, minxor128.gaifman_2, mrpp_4x4#8_8.gaifman_3, newton.3.3.i.smt2-stp212.gaifman_2


## GPU PID-BT on the PACE exact set, and the HPG jobs (Oct 19)

Harness: `gpu/run_pace_gpu.jl <out.tsv> <tlimit> <all | @file | ex001 ...>`; local env `gpu/env` (a copy of `gpusep/env` with CliqueTrees pointed at `CliqueTrees.jl-mt`; HPG uses the gpupid env, tree mt-65e72a3, also tested locally via `git archive`). Per instance: unit weights, `SafeRules(SafeSeparators(GPUExact, 4 minimal triangulations))` splits the graph into atoms; each atom gets the MF upper bound and MMW lower bound (equal: solved without the GPU), else GPU PID-BT (`gpu_tw`) with the remaining budget. CliqueTrees stitches the atom orders into one elimination order, so the width is the max over all atoms plus rule-eliminated vertices, and a `.td` is written and checked with td-validate (`PACE_TD_DIR`, `PACE_VALIDATE`). Time charged = preprocessing + solves; GPU kernel compilation (per bitset width) is not charged. Status `optimal` only if every atom is solved; every width is compared with the published optimum and a mismatch (or an invalid td) is status `MISMATCH`, printed in red, never a solve. Columns: instance, n, m, natoms, max_atom_n, width, optimum, status, time, valid, error. Status values: optimal, timeout, oom, error, crash, MISMATCH. Atoms over the bitset limit or GPU OOM show up as `error`/`oom`.

Local smoke (RTX 5060, 30 s, 20 instances, `gpu/smoke_list.txt`, `gpu/results/smoke_local.tsv`): 20/20 optimal, 0 mismatches, all 20 tds valid.

HPG jobs, submitted Oct 19 (not polled):
- 46696764 `itay-gpu`, hpg-rtx6000 (rtx_pro_6000:1), 48 h: `gpu/hpg_gpu.sbatch`, all 200 at 30 s then all 200 at 1800 s; log `/blue/fairbanksj/itaykadosh/gpu/pace/logs/gpu-46696764.out`; results `/blue/fairbanksj/itaykadosh/gpu/pace/results_hpg/gpu_30.tsv`, `gpu_1800.tsv`, tds in `results_hpg/td_30/`, `td_1800/`. A hard crash (segfault) is recorded as status `crash` for that instance and the job restarts (max 8 per pass).
- 46696783 `itay-ct`: `hpg_cpu.sbatch tamaki17 exact` (1800 s, `-Xmx30g`, 7 workers because 7 x 30 GB must fit in 240 GB). Results `results_hpg/tamaki17_exact.tsv`, log `logs/pace-46696783.out`.
- 46696785 `itay-ct`: `hpg_cpu.sbatch septw exact` (1800 s, 32 workers). Results `results_hpg/septw_exact.tsv`. Both `itay-ct` jobs run one after the other (singleton), also behind the older ParAMD job 46688560 of the same name.
- td-validate was rebuilt on the HPG login node (the laptop binary needs a newer glibc).

## PACE 2017 BONUS on HPG (Oct 19, submitted, not polled)

Runners (`gpu/run_pace_{cpu,gpu}.jl`) with `PACE_SET=bonus`: instance names = filenames without `.gr`; `@list` files (`results_hpg/list_bonus.txt`). Columns: `known_ub` (width of `data/plain/bonus_td`, an UPPER bound, -1 if none) and a final `flag`. `flag`: `NEW_UB` = certified width < known_ub and td-validate accepts it (a better decomposition); `NEW_UB_UNVALIDATED`; `NEW_PROOF` = status optimal and `data/bonus2017/instances.csv` has "?" for that instance (new treewidth proof). Highlighted in green on stderr. `MISMATCH` (red): valid=NOT_VALID, optimal width > known_ub, or a width that differs from a treewidth the organisers proved (numeric in instances.csv). Fixed: the JIT warm-up used ex070/ex001/ex013 (absent in bonus); it now uses the 3 smallest bonus instances.
Local sanity (septw pipeline, 30 s, mrpp_4x4#8_8, newton.3.3.i.smt2-stp212, Promedas_22_6): all optimal, widths 24/19/12 = known, td valid.

Jobs:
- 46726956 `itay-gpub`, hpg-b200 (b200:1), 96 h: `gpu/hpg_gpu_bonus.sbatch`; `/blue/fairbanksj/itaykadosh/gpu/pace/results_hpg/gpu_bonus_1800.tsv`, tds `td_bonus_1800/`, log `logs/gpub-46726956.out`.
- 46726957 `itay-ctb`, hpg-milan, 32 cpus: `gpu/hpg_cpu_bonus.sbatch` (CPU septw per atom, 1 thread, 32 instances in parallel); per-instance `results_hpg/cpu_bonus_parts/`, merged `results_hpg/cpu_bonus_1800.tsv`, tds `td_cpu_bonus_1800/`, log `logs/ctb-46726957.out`.
- 46726958 `itay-ctt` (name overridden): `hpg_cpu.sbatch tamaki17 bonus`; `results_hpg/tamaki17_bonus.tsv`, log `logs/pace-46726958.out`.
