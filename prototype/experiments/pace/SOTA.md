# Strongest current exact treewidth baselines (survey + 30 s local results)

All numbers: 1 thread per instance, 30 s, shared 16-core laptop, 3-4 concurrent jobs, JVM -Xmx8g, tds checked with td-validate. Harness: `tools/run.py` (new solver entries `rtw`, `exacttw`, `jdrasil`, `arboretum`); raw results in `results/<solver>_{exact,bonus}.tsv`.

## Survey (public code, 2018-2026)

| solver | year / venue | code | claimed on PACE 2017 | status here |
|---|---|---|---|---|
| RTW, Tamaki "contraction-recursive" (HPID) | 2023, arXiv 2307.01318 (submitted IPEC 2023) | https://github.com/twalgor/RTW (GPL-3, Java) | bonus set (100): 72 / 92 / 98 solved at 100 / 1000 / 10000 s; Tamaki PID 11/38/68; Tamaki SEA22 65/82/85 | built, run |
| twalgor/tw: UpLow + ExactTW, Tamaki "Heuristic computation of exact treewidth" | 2022, SEA 2022 (LIPIcs 17) | https://github.com/twalgor/tw | UpLow solves many bonus instances (see RTW row, 65/82/85) | UpLow run earlier (128/200); ExactTW run here |
| Tamaki PID (PACE 2017 Track A 2nd place) | 2017 | https://github.com/TCS-Meiji/PACE2017-TrackA | all 100 public instances | run earlier |
| Jdrasil (Bannach, Berndt, Ehlers), SAT/PID/DP library | 2017 SEA; last commit 2024 | https://github.com/maxbannach/Jdrasil | PACE 2017 entrant | built with javac (gradle fails offline), run |
| arboretum (Meintrup), Tamaki-style exact + safe separators, Rust | ~2021/22 | https://github.com/jmeintrup/arboretum | none stated | built (cargo, Cargo.lock refreshed, unicode-segmentation pinned), run |
| freetdi/p17 (Larisch/Salfelder, PACE 2017 winner, tw-exact = Tamaki port) | 2017 | https://github.com/freetdi/p17 | PACE 2017 winner | cloned, NOT built (needs boost + autotools, none installed; its tw-exact is a port of Tamaki's PID, so no new algorithm) |
| Trellis (Fichte, Lodha, Szeider) SAT local improvement | 2017/18 | https://github.com/daajoe/trellis | heuristic, not exact | skipped |
| Berg/Jarvisalo SAT/MaxSAT encodings | ICTAI 2014 | no maintained public code found | pre-2017, weaker than PID | skipped |
| BZTreewidth (Bodlaender/van der Zanden), TreeWidthSolver.jl, QuickBB, libtw | various | github | n/a, old or Bouchitte-Todinca style | not built |

Searches (web, Oct 2026) found no 2024-2026 exact-treewidth solver with public code that claims to beat RTW. HPID/RTW (2023) is the newest. Caution: github.com/junkawahara/treewidth-solvers is an AI-generated benchmark wrapper with no measured results; used only as a lead list.

Notes on twalgor ExactTW: invoked exactly as the README says (`gr td -acsd`, `-Xss1g -Xmx8g`). It is genuinely slow on some instances (ex001, n=262, tw 10: >60 s with and without `-acsd`), while others are fast (ex003 22 s). So the earlier >100 s on ex001 was not misuse; it is a research implementation of the 2019 PID-over-separator-lists algorithm.
Notes on arboretum: reports "OK" with width optimum+1 on ex014 (19 vs 18) and ex156 (12 vs 11), so it is not exact; its 169 "OK" count has 167 correct.
RTW uses `-XX:ActiveProcessorCount=1`; it writes a `.twc` certificate which `run.py` converts to `.td` before validation. Disagreement check: every other OK width agrees across solvers.

## Solved at 30 s

| solver | exact (200) | bonus (100) |
|---|---|---|
| **GPU pipeline (HPG, results_hpg/gpu_1800.tsv, 1800 s)** | 200 | n/a |
| **CPU pipeline septw (results_ablation, 30 s)** | 199 | (septw_bonus: 21) |
| Tamaki17 (30 s) | 173 | 7 |
| twalgor ExactTW (-acsd) | 163 | 16 |
| RTW (2023) | 160 | **55** |
| arboretum | 167 (169 claimed) | 10 |
| Jdrasil | 154 | 8 |
| UpLow22 (earlier) | 128 | not run |
| septw 1-thread (README) | 134 | 21 |

Union of all five external solvers: 192/200 exact, 61/100 bonus (RTW alone solves 41 bonus instances no other external solver does). Exact: RTW solves 10 that Tamaki17 misses, Tamaki17 solves 23 that RTW misses.

## Verdict

- Exact set: strongest baseline is still Tamaki17 (173), with ExactTW/arboretum/RTW in the 160s. Our 199 (CPU) / 200 (GPU) beat every one at equal 30 s budget (GPU number is at 1800 s; compare Tamaki17 HPG 1800 s = 200/200).
- Bonus set: the strongest baseline we must beat is RTW (55 at 30 s vs 21 for septw and 7 for Tamaki17; paper claims 98/100 at 10000 s). Claiming a win over state of the art on bonus requires beating RTW, and at 30 s single-thread we do not.
- So the fair statement is: ahead on the PACE exact set, behind RTW on the bonus set at 30 s (check at 1800 s with `hpg_sota.sbatch`).

## HPG 1800 s runs (not submitted)

`hpg_sota.sbatch` (same pattern and PATH/LD_LIBRARY_PATH exports as `hpg_cpu.sbatch`): `sbatch hpg_sota.sbatch rtw bonus`, `... rtw exact`, `... tamaki17 exact`. Sync `tools/build/rtw`, `tools/build/tw2022`, `tools/build/jdrasil`, `tools/jdk`, `tools/arboretum/src/target/release/arboretum-cli`, and `data/plain` to `/blue/fairbanksj/itaykadosh/gpu/pace/`. Jobs of the same name serialize (singleton), submit one at a time. 7 JVM workers x 30 GB.
