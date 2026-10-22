# GPU fill-reducing orderings

GPU minimum fill (round-based, batched) and GPU nested dissection with a CPU+GPU refinement split.
Prototype code (CUDA.jl, Julia 1.12); scripts hard-code HiPerGator paths.

## Core (about 8.5k lines)

- `Rounds.jl`: shared types (`RoundsAlg`, scores) and the CPU round-based reference.
- `GRounds.jl`: GPU minimum fill / minimum degree by rounds (distance-2 Luby selection, Wing-Huang deficiency updates, supervariables, dense tail on the CPU).
- `GSmall.jl`: one-launch modes and `grounds_batch` (one block per graph, many graphs per launch, device-resident input/output).
- `nd/GND.jl`: GPU nested dissection (twin compression, coarsening, separators, leaves ordered by `grounds_batch`).
- `nd/hybfm.jl`: FM refinement of the coarse levels on CPU threads, overlapped with the GPU.
- `nd/policy.jl`: size-based choice between GPU ND and a serial ordering.
- `nd/kway.jl`: one-shot k-way top partition (compared, not used by default).
- `cudss/ndtree.jl`: the ND partition tree that cuDSS needs alongside a user permutation.

Recommended ND config: `gndd:256:hyb=1;rmax=4` (see `notes/ND_NOTES.md`, "Closing the AMD gap").

## Experiments

- `nd/ndbench.jl`, `nd/hpg/`: ND benchmarks; `nd/results3/tables.md`.
- `small/`: batched and end-to-end runs.
- `cudss/`: end-to-end solves with cuDSS.
- `pbase/`: ParAMD / ParMETIS baselines.
- `mfstrat/`: serial MF strategies 0-4.
- `harvest/`: result tables.
- `notes/`: design notes and changelogs.
