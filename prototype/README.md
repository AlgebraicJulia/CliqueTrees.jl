# Prototypes

Research code for the tensor-network contraction paper. Not part of the package; scripts hard-code
local and HiPerGator paths (`/path/to/tn-work` stands for the local working directory).
The library code lives in `src/` (`pipeline.jl`, `bt/`) and `ext/CUDAExt` (GPU septw, queue).

## Algorithms

- `gpupid/`: GPU positive-instance-driven treewidth (PID-BT).
- `gpusep/`: GPU septw kernels (sepenum, sepdp) behind the CUDA extension.
- `gpuord/`: GPU orderings inside HBT.
- `rounds/`: GPU round-based minimum fill / degree (the ordering paper has the newer version on `itay/gpu-nd`).
- `gpucost/`: GPU exact-cost scoring of contraction trees and batched subtree reconfiguration.
- `gpuanneal/`: GPU contraction-tree annealer.
- `pipeline/`: pipeline scale tests.

## Experiments (`experiments/`)

- `pace/`: PACE 2017 treewidth, exact and bonus tracks.
- `exactness/`: width vs contraction cost, real contraction on a B200 (torch and cuTensorNet executors).
- `costsel/`: cost-aware choice among near-minimum-width decompositions (`pmccost`).
- `baselines/`: cuTensorNet path finder, TensorOrder/QTensor-style decompositions.
- `einsumbench/`: einsum benchmark harness; `ctgplugin/`: cotengra optimizer plugin.
- `frontier/`: Sycamore slicing experiments; `lightcone/`: light-cone QAOA.
- `fill/`: fill-in comparisons.
