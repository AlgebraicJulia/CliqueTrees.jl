# gpupid: GPU positive-instance-driven Bouchitté–Todinca

Prototype (CUDA.jl, Julia 1.12) of Tamaki's PID-BT exact treewidth algorithm (ESA 2017; JCO 2019), weighted: a bag weighs the sum of log2 of its index dimensions.
This is not the CliqueTrees solver called PIDBT, which is `septw` (Tamaki SEA 2019, `sepenum` + `sepdp`); `septw` is the CPU baseline here.

- `src/bitsets.jl`: fixed-size bitsets for device code
- `src/pidref.jl`: CPU reference of the same search (per-round counts match the GPU)
- `src/devfuns.jl`, `src/sortjoin.jl`: device functions; the I x O block join, as a sort-join with pruning
- `src/gpupid.jl`: the level-synchronous search (rounds of join, classify, hash dedup), chunked under a memory budget
- `scripts/run_gpu.jl`: solve (search auto, order rbfs); `scripts/run_certify.jl`: one decision just below a given width; `scripts/cpu_base.jl`: CPU septw/HBT baselines
- `test/test_ref.jl`, `test/test_gpu.jl`: reference and GPU against CliqueTrees
- `hpg/`: HiPerGator sbatch files and instance lists

Some scripts still hard-code local or HiPerGator paths.

Results so far (B200, atoms of 100–600 vertices, one decision at the HBT width): 28 of 77 certified, the largest with 592 vertices; a median of 52x over CPU septw on the 16 both solve; widths always agree.
42 runs ran out of memory, in the pre-dedup candidate buffers of the join, which are deduplicated only once per round.
