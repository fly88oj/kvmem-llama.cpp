# llama.cpp patch replay

`llama-kvmem-current.patch` is the cumulative diff against pinned `a25c986` (tag `b11189`,
the llama.cpp release LM Studio's official ROCm runtime 2.46.0 ships).
It includes the existing KVMem hooks, multimodal batch, MTP, media
parser and mtmd helper extensions, plus FP32 GDN Record/Fold for ReplaySSM.
It also fixes reasoning-budget initialization from a template's generation prefix,
carries interleaved-SWA support (gemma3/gemma4 incl. QAT variants, capture hooks
in `src/models/gemma4.cpp`/`qwen35.cpp`) and the `logical_pos`/`embd_nextn`
channel across the b111xx `llama_batch_ext` architecture. The shared
`LLAMA_KVMEM` integration block in `src/CMakeLists.txt` (CUDA and HIP) is part
of this patch as well; the HIP-only increments live in the delta below.
`scripts/apply-patches.sh` applies it
without creating commits and checks for an already applied tree.

`reasoning-budget-upgrade.patch` upgrades the v0.15.0 ReplaySSM tree.
`replayssm-upgrade.patch` upgrades the preceding multimodal/query-replay tree.
`multimodal-upgrade.patch` upgrades the KVMem working tree recorded before
the 2026-09-14 implementation to the same current code. The script checks applicability before
changing files. Unrelated local changes are preserved; conflicting changes
require review.

The numbered `0001` through `0004` files are historical patches, retained for
reference. They are superseded by the cumulative diff: the old series did
not cleanly replay on the current pin and must not be applied together with it.

`kvmem-hip-port-cmake.patch` is the AMD/HIP delta: it adds only the `GGML_HIP`
increment (mutual-exclusion guard, HIP stage-in branch, `NOT GGML_HIP` cudart
guard) to the `LLAMA_KVMEM` block that the cumulative patch already put in
`llama.cpp/src/CMakeLists.txt`, and applies **on top of** the cumulative patch
(the cumulative patch itself is backend-agnostic and HIP-free).
`scripts/windows/build-hip.ps1` applies both in order; regenerate the delta with
`python scripts/windows/emit-hip-patch.py` after editing that CMakeLists. See
[docs/amd-hip-port.md](../docs/amd-hip-port.md).

To check a clean extraction without changing the active submodule:

```bash
mkdir -p /tmp/kvmem-llama-patch-check
git -C llama.cpp archive a25c986 | tar -x -C /tmp/kvmem-llama-patch-check
KVMEM_LLAMA_DIR=/tmp/kvmem-llama-patch-check scripts/apply-patches.sh
KVMEM_LLAMA_DIR=/tmp/kvmem-llama-patch-check scripts/apply-patches.sh
```
