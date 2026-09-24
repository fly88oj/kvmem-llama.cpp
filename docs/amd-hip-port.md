# KVMem on AMD GPUs (HIP / ROCm)

> **Language:** English (this file) · [简体中文](amd-hip-port.zh-CN.md)
>
> Scope: running the **same KVMem code path** (bounded GPU slot pool + pinned host
> tier + query retrieval + GPU stage-in + MTP/ReplaySSM) natively on AMD GPUs,
> following the same strategy llama.cpp uses for its own ROCm backend — hand the
> CUDA sources to the ROCm toolchain instead of rewriting them.

The main [`README.md`](../README.md) targets NVIDIA CUDA. This document covers the
AMD **HIP/ROCm** port, validated on **Radeon RX 9070 XT (RDNA4 / gfx1201)** on
native Windows. The Linux/WSL2 route works the same way (`build-hip.ps1` is a thin
wrapper around `cmake -DGGML_HIP=ON`).

---

## 1. Tested platform (reproduction conditions)

Everything under "GPU results" in §6 was measured on exactly this setup. Numbers
will differ on other hardware, drivers, or OS builds.

| Component | Value |
|---|---|
| GPU | Radeon RX 9070 XT, 16 GiB (RDNA4, `gfx1201`) |
| CPU / RAM | 16C/32T, 47 GiB |
| OS | Windows 11 (native, not WSL) |
| ROCm | 7.14.0 wheel (HIP 7.2 runtime), installed into `.venv-rocm` |
| GPU driver | AMD Adrenalin 26.8.x |
| Compiler | ROCm clang/clang++ (from the wheel), MSVC 14.5x Windows SDK headers |
| Build | CMake + Ninja Multi-Config, `AMDGPU_TARGETS=gfx1201` |
| Model | Qwopus3.5-9B-Coder-MTP `Q4_K_S` GGUF (Qwen3.5 GDN hybrid) for smoke/needle |

---

## 2. What the port changes

All changes live **in this repository** (the `kvmem/` policy library is untouched).
The `llama.cpp` submodule stays at pin `b81c99b`; the port is delivered as patches
plus out-of-tree files so a vanilla submodule build stays bit-identical to upstream.

| File | Purpose |
|---|---|
| `compat/hipify/cuda_runtime.h`, `cuda_fp16.h`, `cuda_bf16.h` | Forward the CUDA header names to their HIP equivalents. Only the HIP configure puts this dir on the include path, so CUDA builds are unaffected. |
| `compat/hipify/kvmem_hip_defs.h` | Maps every `cuda*` symbol used by `src/adapter/*` and the stage-in kernel onto `hip*` (mirrors ggml's `vendors/hip.h`, and fills gaps such as `cudaHostAlloc` and `cudaPointerAttributes`). Each define is `#ifndef`-guarded so it can coexist with `vendors/hip.h`. |
| `llama.cpp/src/CMakeLists.txt` (via `patches/kvmem-hip-port-cmake.patch`) | The `GGML_HIP` branch: compiles `llama-kvmem-stagein.cu` with ROCm clang (Windows: `LANGUAGE CXX` + `hip::device`; Linux: native `LANGUAGE HIP`), and injects the compat include dir. |
| `patches/llama-kvmem-current.patch` | The cumulative KVMem feature diff vs the pin (memory tiers, MTP, multimodal, reasoning budget, GDN kernels). Backend-agnostic; shared with the CUDA build. |
| `scripts/windows/rocm-install.ps1` | Installs the ROCm wheel (`pip --index-url https://repo.amd.com/rocm/whl-multi-arch/ rocm[libraries,devel]==7.14.0` + `rocm-sdk init`). Same method as llama.cpp's official `windows-setup-rocm` action. |
| `scripts/windows/patch-rocm-headers.py` | Idempotent fix for a ROCm-on-Windows toolchain bug (llama.cpp issue #22570: MSVC 14.5x `<cmath>` collides with clang's HIP math forward-declares for `isgreater`/`isless`/…). Guarded by `KVMEM_HIP_SKIP_MATH_FWD`; the build script enables it. |
| `scripts/windows/revert-guards.py` | Cleanly reverses `patch-rocm-headers.py` for **both** patched clang headers. |
| `scripts/windows/build-hip.ps1` | One-shot: patch headers (idempotent) → apply the cumulative KVMem patch → apply the HIP cmake delta → configure (ROCm clang, `-DAMDGPU_TARGETS=gfx1201`) → build → ctest → stage ROCm runtime DLLs next to the exe. |
| `scripts/windows/start-server-hip.ps1` | Launches `llama-kvmem-server` with KVMem defaults (no manual PATH setup). |
| `scripts/windows/smoke.ps1`, `needle.ps1` + `tests-data/gen_needle.py` | Baseline-vs-KVMem smoke test and needle-in-a-haystack retrieval check. |

---

## 3. Prerequisites

1. **AMD GPU** with a ROCm-supported target. `gfx1201` = RX 9070 XT / 9070. For
   other cards pass the matching `-GpuTarget` (e.g. `gfx1100` for RX 7900).
2. **Windows 11 x64** (or Linux/WSL2), **Visual Studio 2022 C++ Build Tools**
   (for the Windows SDK headers/libs), **CMake** and **Ninja** on `PATH`
   (`winget install Kitware.CMake Ninja.Ninja`).
3. **Python 3.10–3.12** (the ROCm wheels do not yet support 3.13+).
   `rocm-install.ps1` picks a compatible interpreter automatically.
4. ~**8 GiB** of free disk for the ROCm venv, and enough host RAM for the pinned
   KV arena you request with `--kvmem-cpu-gb` (the recipes below use 6–12 GiB).
5. A **GGUF model**. The smoke/needle scripts default to a Qwen3.5-9B MTP quant;
   any llama.cpp-compatible GGUF works for a first run.

---

## 4. Beginner quick start (copy-paste, 3 steps)

Open **PowerShell** in the repository root (the folder that contains `scripts\`).
Run these three commands one at a time and wait for each to finish.

```powershell
# Step 1 - install the ROCm toolchain into a project-local venv (one time, ~7.5 GB).
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\rocm-install.ps1

# Step 2 - build. Defaults to gfx1201 (RX 9070 XT). Pass -GpuTarget for other cards.
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\build-hip.ps1

# Step 3 - run a quick correctness check (tell it where your model is).
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\smoke.ps1 -Model 'D:\models\your-model.gguf'
```

**What you should see**

- Step 1 ends with `ROCm installed at: <path>`.
- Step 2 ends with `Done. Server: ...\build-hip\bin\Release\llama-kvmem-server.exe`.
- Step 3 prints two sections and ends with `smoke test PASSED (both runs exit=0)`.

**Start the chat server** (then open <http://127.0.0.1:18200/> in your browser):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-server-hip.ps1 -Model 'D:\models\your-model.gguf'
```

Press `Ctrl+C` in the terminal to stop the server. You can also point any
OpenAI-compatible client at `http://127.0.0.1:18200/v1`.

> **No PATH setup needed.** `build-hip.ps1` copies every required ROCm DLL
> (including the transitive `rocsolver.dll`: ggml-hip → hipblas → rocblas →
> rocsolver) next to the executables, so they run from a double-click or a bare
> command line. **Do not delete or move `.venv-rocm`** — rocBLAS resolves its
> Tensile kernel library through an absolute path baked into that venv.

---

## 5. Build and run (details)

### Build options

`build-hip.ps1` parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `-GpuTarget` | `gfx1201` | Value passed to `AMDGPU_TARGETS`/`GPU_TARGETS`. |
| `-BuildDir` | `build-hip` | Output directory. |
| `-RocmRoot` | *(auto)* | Explicit ROCm root; otherwise resolved from `.venv-rocm` then `HIP_PATH`. |
| `-Jobs` | `min(16, cores)` | Parallel compile jobs (clang HIP TUs peak ~2 GB each). |
| `-HostOnly` | *(off)* | Build the CPU-only KVMem host library + host tests (no GPU toolchain needed). |

Products: `build-hip\bin\Release\llama-kvmem-server.exe` and `llama-kvmem-cli.exe`.

### Run the server with the 27B long-context recipe

```powershell
powershell -File scripts\windows\start-server-hip.ps1 `
  -Model 'D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf' `
  -Ctx 262144 -Budget 36864 -GenReserve 16384 -Mtp 3
```

The scripts take the model path from `-Model` or the `KVMEM_MODEL` environment
variable (no personal paths are hard-coded). `-Mtp 0` disables multi-token
prediction; `-Mtp 3` enables `--spec-type draft-mtp --spec-draft-n-max 3`.

### Web UI

The server looks for the static UI two levels above the executable
(`build-hip\bin\share\kvmem\ui`). Build it once with Node.js 22+:

```powershell
python scripts\build-webui.py --output ..\build-hip\bin\share\kvmem\ui
```

If `index.html` is missing the server still starts, but `/` returns 404 (the API
is unaffected). Use `--ui-dir PATH` to point elsewhere, or `-NoUi` to disable.

---

## 6. Verification and test results

Results are split by **what hardware they need**. Groups A and B are reproducible
on any machine with a compiler + git; group C requires the RX 9070 XT (or
comparable) and is reported from the developer's measurements.

### A. Host KVMem library tests — backend-independent ✅

These exercise the CPU-side KVMem library (store, pinned tier, runtime, raw KV
store, NVMe-disabled guard). They do **not** need a GPU.

**Reproduce:**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\build.ps1 -HostOnly -BuildDir build-host
```

`build.ps1 -HostOnly` configures with `-DKVMEM_BUILD_LLAMA=OFF`, builds the five
host test targets, and runs them under CTest.

**Result (5/5 passed, exit code 0 each):**

| Test | Result |
|---|---|
| `kvmem_store_test` | PASS (`OK`) |
| `pinned_kv_tier_test` | PASS (`OK`) |
| `nvme_disabled_test` | PASS (`NVMe requests rejected; memory-only tests run separately`) |
| `kvmem_runtime_test` | PASS (`OK`) |
| `raw_kv_store_test` | PASS |

The runtime test emits structured diagnostics such as
`KVMEM_TIERS cpu_bytes=16384 cpu_slots=16 nvme_bytes=0 …` and
`KVMEM_TRACE mandatory_trim policy=topk kept=1 dropped=1 budget=4`, confirming the
tier bookkeeping and trim policies behave as specified.

### B. Patch-replay integrity — reproducible anywhere ✅

The submodule is patch-managed, so a clean checkout must reproduce the working
tree exactly. **Reproduce** (bash; see [`patches/README.md`](../patches/README.md)):

```bash
tmp=$(mktemp -d)
git -C llama.cpp archive b81c99b | tar -x -C "$tmp"
git -C "$tmp" init -q
git -C "$tmp" apply patches/llama-kvmem-current.patch     # cumulative KVMem base
git -C "$tmp" apply patches/kvmem-hip-port-cmake.patch    # HIP cmake delta
git diff --no-index llama.cpp/src "$tmp/src"              # expect: no output
```

**Result:** the cumulative patch applies cleanly to the pristine pin `b81c99b`,
the HIP delta applies on top, and the replayed tree is **byte-identical** to the
working tree (verified across `llama-kv-cache.cpp`, `llama-kv-cells.h`,
`llama-graph.cpp`, `speculative.cpp`, `llama-model.cpp`, `src/CMakeLists.txt`).
The base patch is HIP-free; the HIP wiring is isolated in the delta patch.

### C. GPU results on RX 9070 XT — requires that hardware

Measured 2026-09-22 on the platform in §1. **Reproduce** with the scripts noted
per row; these need the AMD GPU and cannot be re-run on a machine without it.

1. **Build:** 593/593 targets, zero errors (includes all ggml-cuda kernels
   code-generated for `gfx1201`, the `FA_ALL_QUANTS` template instantiations, the
   15 kernels in `llama-kvmem-stagein.cu`, and the injected `gdn_fold_f32` /
   `gdn_conv_fold_f32`).
2. **CTest:** 12/12 passed (`kvmem_store`, `pinned_tier`, `runtime`,
   `raw_kv_store`, `gdn_replay`, `server_options`, …); `kvmem-mtp-kv-test` skips
   by design when no model is supplied.
3. **Device path:** logs show `ROCm0 compute buffer …` and `ROCm_Host compute
   buffer …` (HIP device + pinned host buffers both active), and CUDA-Graph reuse
   (`graphs reused=47`), so the HIP graph path is exercised.
4. **Baseline inference** (Qwopus3.5-9B-Coder-MTP Q4_K_S): prefill ≈ 134–152
   tok/s (4K cold-start segment); decode ≈ **85–87 tok/s**.
5. **KVMem equivalence (needle-in-a-haystack)** — `scripts\windows\needle.ps1`:
   a 2314-token document with the needle placed mid-document (outside the
   sink/recent window), run with `--kvmem-budget 512 --kvmem-cpu-gb 6 --kv-dtype q8_0`:
   - `KVMEM_TIERS cpu_bytes=6442450944 cpu_slots=11565` — host tier active;
   - `KVMEM_STAGE retrieval_ms=31.98 replay_n=64` — retrieval + query replay ran;
   - `KVMEM_TRACE gdn_ckpt/gdn_restore bytes=52691548` — the GDN recurrent-state
     checkpoint/fold kernels (previously CUDA-only) run on `gfx1201`;
   - **model output `amber-jaguar-4417` — recall correct**;
   - long-document prefill ≈ 2925 tok/s (cache-reuse segment), decode ≈ 78–80 tok/s.
6. **KVMem + MTP** (`--spec-type draft-mtp --spec-draft-n-max 3`): needle recall
   correct again (`retrieval_ms=36.44`, `replay_n=63`); MTP coexists with the slot
   pool on the HIP path.
7. **OpenAI-compatible server** (KVMem + q8_0 KV + CPU spill): `/health`,
   `/v1/models`, `/v1/chat/completions` all OK, responses include
   `prompt_cache_hit/miss_tokens` and coherent `reasoning_content`.
   Note: server-side GDN ReplaySSM has an upstream model allow-list ("requires …
   Qwen 27B with MTP"); for a non-27B hybrid model run the server with
   `--spec-type none`. This is upstream policy, unrelated to the AMD port (the CLI
   path is unrestricted).

### D. KVMem on/off scaling (RX 9070 XT, Qwopus3.5-9B-MTP Q4_K_S, both f16 KV)

Reproduce with `tests-data\bench-kvmem.ps1` (four context tiers) +
`tests-data\bench-256k.ps1`. VRAM is the Windows PDH `GPU Process
Memory\Dedicated Usage` per-PID peak. KVMem rows use budget=4096,
gen_reserve=1024, block=128, CPU arena 12 GiB. Tier names are the `-c` config;
the real prompt length is the engine's logged `prompt_n`.

| `-c` config | Mode | prompt tok | prefill tok/s | decode tok/s | VRAM peak | RAM peak | needle |
|---|---|---:|---:|---:|---:|---:|---|
| 8K   | plain  | 2527  | **3651** | 83.7 | 5830 MB | 4.9 GB | YES |
| 8K   | KVMem  | 2591  | 3319 | **84.5** | 5830 MB | 17.4 GB | YES |
| 32K  | plain  | 9934  | **3435** | 83.7 | 6691 MB | 5.0 GB | YES |
| 32K  | KVMem  | 9998  | 3256 | 83.1 | **5816 MB** | 17.7 GB | YES |
| 128K | plain  | 40724 | 2193 | 72.6 | 9896 MB | 5.1 GB | YES |
| 128K | KVMem  | 40788 | **3248** (+48%) | **86.4** (+19%) | **5821 MB** (−41%) | 18.7 GB | YES |
| 256K | plain  | 40724 | 1016 | 67.4 | 13244 MB (83% of card) | 5.2 GB | YES |
| 256K | KVMem  | 40788 | **3258** (3.2×) | **86.1** (+28%) | **5816 MB** (−56%) | 18.6 GB | YES |

**Takeaways**

1. For short contexts (≤10K prompt) plain is slightly ahead (prefill +8–10%, same
   VRAM) — no need for KVMem when the working set fits.
2. The crossover is ~40K tokens; beyond it KVMem wins across the board (the
   bounded window keeps prefill/decode flat while plain decays 3651→1016 tok/s).
3. KVMem VRAM is ~constant at ≈5.8 GB (the slot pool does not grow with `-c`); at
   a 256K virtual workspace it saves 7.4 GB vs plain — the quantitative basis for
   the "27B + 256K on a 16 GB card" recipe.
4. The cost is +13 GB host RAM (pinned arena; fine on a 48 GB machine) and a
   retrieval-failure risk — all four tiers hit the needle, so budget 4096 is
   lossless for a 40K workspace.
5. Model load is 2.7× faster at the 256K tier with KVMem (16.2 s vs 43.1 s; plain
   must pre-allocate the full KV).

### E. Tuning matrix (RX 9070 XT)

Each candidate was run for real on the `bench_128k` prompt (40.7K tokens), ctx
139264, greedy, `-n 128`, with needle recall as the quality gate; one warmup then
the second run is reported. Metrics: `KVMEM_PERF prompt_toks` (prefill) and
`KVMEM_GEN_WALL toks` (end-to-end decode). Reproduce with
`tests-data\opt-matrix-a.ps1`, `opt-matrix-c.ps1`, `opt-eval-b.ps1`, `opt-final.ps1`.

| # | Candidate | Type | prefill | decode(wall) | VRAM | needle | Verdict |
|---|---|---|---:|---:|---:|---|---|
| — | baseline f16 KV, no MTP | — | 3202 | 72.4 | 5821 | YES | reference |
| A2 | **q8_0 KV + MTP3** | flags | **3218** | **87.5** | **5742** | YES | ✅ **adopted (final)** |
| A3 | + ubatch 2048 | flags | 3232 | 75.1 | 6055 | YES | ❌ |
| A4 | block-tokens 32 | flags | 2906 | 77.3 | 5742 | YES | ❌ |
| A5 | budget 8192 | flags | 2817 | 79.7 | 5842 | YES | ❌ |
| C1 | MTP draft-n-max 4 | flags | 3104 | 65.6 | 5793 | YES | ❌ acceptance dropped |
| C2 | block-tokens 256 | flags | 3159 | 77.2 | 5742 | YES | ❌ |
| C3 | budget 2048 | flags | 3325 | 75.6 | 5700 | YES | ❌ decode dropped |
| C4 | harvest-v | flags | 3138 | 84.1 | 5772 | YES | ❌ no gain beyond noise |
| A7/B | q8_0 K + q4_0 V | flags | 3213 | 86.2 | 5720 | YES | ➖ fallback (VRAM-tight; single quality sample) |
| B | `GGML_HIP_NO_VMM=OFF` + `NATIVE=ON` | rebuild | 1537 | 72.1 | 5819 | YES | ❌ **VMM harmful: f16 prefill −52%** |
| C | `GGML_CUDA_FORCE_CUBLAS=ON` | rebuild | 658 | 72.6 | 6200 | YES* | ❌ **−80% prefill + rocBLAS errors** |
| — | hipBLASLt (`GGML_HIPBLAS`) | rebuild | — | — | — | — | 🚫 excluded: pin `b81c99b` has no integration; bumping the pin breaks the fork's patch anchors |
| — | FA on/off | flags | — | — | — | — | 🚫 excluded: code evidence — q8_0 V requires FA enabled (`llama-context.cpp`); `auto` already does this |
| — | thread `-t` tuning | flags | — | — | — | — | 🚫 excluded: with `-ngl 99` full offload the CPU only samples; B2 vs A2 differ <2% (noise) |

\* Group C only produced correct results after borrowing LM Studio's Tensile
library; the wheel's rocBLAS lacks `gfx1201` kernel dispatch — a ROCm-on-Windows
maturity issue, consistent with the research-phase TheRock matrix findings.

**Final spec:** binaries in `build-hip\bin\Release\` (default flags: `NO_VMM=ON`,
adaptive mmq/hipBLAS, `GRAPHS=ON`, `FA_ALL_QUANTS=ON`). Recommended for long
context (≥30K): `--kvmem -ctk q8_0 -ctv q8_0 --spec-type draft-mtp
--spec-draft-n-max 3 --kvmem-block-tokens 128 --kvmem-budget 4096+ (scale with
VRAM) --kvmem-cpu-gb 12`. For short context (<10K): omit `--kvmem` (plain
llama.cpp prefills faster). Net gain vs the first baseline: **decode +21%
(72.4→87.5)**, VRAM −79 MB, prefill flat, needle all-hit (no quality regression).

---

## 6b. 128K vs LM Studio + optimization study (2026-09-23, matched methodology)

### F. 128K head-to-head vs LM Studio

Same machine (RX 9070 XT), same model, same `bench_128k.txt` prompt (**40724
tokens**), ctx 139264, greedy, **n=256 steady-state**, median of 3 (see the
methodology note below). LM Studio = its Vulkan runtime, f16 full-KV, MTP
draft-n 3, ~14.6 GB VRAM. KVMem-HIP = this port, q8_0 KV, MTP3, budget 4096,
block 128, ~5.8 GB VRAM. Reproduce: `tests-data/bench_server.py` (LM Studio) and
`tests-data/opt-decode.ps1` (KVMem).

| @128K (40.7K prompt) | prefill tok/s | decode tok/s (warm) | VRAM |
|---|---:|---:|---:|
| LM Studio (Vulkan, f16 full-KV, MTP3) | 2387 | 102-104 | 14.6 GB |
| **KVMem-HIP (q8_0, MTP3, bounded window)** | **3050-3220** | **108-111** | **5.8 GB** |
| **Gap (KVMem vs LM Studio)** | **+28-35%** | **+4-7%** | **-60%** |

Takeaways:
- **Decode is essentially tied.** Both use MTP3 and both sit near the ~125 tok/s
  memory-bandwidth roofline for a 5.1 GB Q4_K_S model on this card (KVMem 86-89%,
  LM Studio 82-83%). KVMem's bounded window does not speed decode much because at
  40K context with `n_head_kv=4` the decode bottleneck is weight matmul, not KV
  attention.
- **KVMem wins prefill by ~30%** (bounded window => less attention work while
  staging the 40K prompt) and **VRAM by 60%** (5.8 vs 14.6 GB).
- **VRAM is the decisive robustness difference.** LM Studio's 14.6 GB nearly
  saturates the 16 GB card; any extra desktop VRAM pressure spills to shared
  memory and decode collapses (we measured 102 -> 23.5 tok/s under contention with
  a stray 5.8 GB process). KVMem's 5.8 GB is unaffected - and is what enables 256K
  on a 16 GB card.

> **Measurement methodology (important).** A single n=64 generation badly
> under-measures decode: GPU clocks and MTP acceptance ramp over the first ~1-2 s,
> so the same KVMem config reads ~70 tok/s at n=64 but ~108 at n=256. Always
> benchmark decode at **n>=256, median of >=3 runs**. The short-generation tables
> in section 6D/6E therefore understate steady-state decode by ~20-30%; the
> corrected steady-state figure for the shipped recipe is **~108-111 tok/s**.

### G. Optimization study - all runtime leads tested

Swept one variable at a time from the baseline on `bench_128k.txt`
(`tests-data/opt-sweep.ps1` for prefill/retrieval/VRAM; `tests-data/opt-decode.ps1`
for decode at n=256, median of 3):

| Lead | Result | Verdict |
|---|---|---|
| KV dtype q4_0/q5_0 V vs q8_0 | decode 106.5/106 vs 108.4; prefill flat | no gain (KV not the bottleneck); keep q8_0 for quality |
| MTP off vs n-max 3 | 80.7 vs 108.4 | **MTP = +34%, the dominant lever (keep on)** |
| MTP n-max 2 vs 3 | 105.4 vs 108.4 | n-max 3 slightly better |
| `--spec-kv-dtype q4_0` | 108.5 vs 108.4 | no change |
| budget 4096 / 6144 / 8192 | prefill 3218 / 2988 / 2818; retrieval 114 / 156 / 191 ms | **4096 optimal** |
| block-tokens 128 / 64 | prefill 3218 / 3090; retrieval 114 / 151 ms | **128 optimal** |
| needle recall | 3/3 in every config | quality preserved |

**Conclusion: the shipped recipe (q8_0/q8_0, MTP3, budget 4096, block 128) is the
runtime optimum - no tested variant beats it.** The same optima hold at the 32K
tier (re-swept: baseline prefill 3232 and best-in-class decode; budget 4096 and
block 128 again beat 6144/8192 and 64; MTP off costs -22% decode), so the recipe
is tier-robust across 32K-128K. Below a ~10K prompt KVMem's window exceeds the
prompt, so plain (no `--kvmem`) prefills faster - use KVMem for >=30K. The
retrieval re-layout (`layout_d2h` + `layout_h2d` ~ 1.9 ms) and total
`retrieval_ms` (~114-125 ms, one-time per turn, ~1% of the 13 s prefill) are not
decode bottlenecks, so the sync-collapse / stage-in-overlap micro-optimizations
are low-value here (their measured ceiling is <=1.5% of a one-time path).

**Kernel-level leads - investigated to closure:**
- **rocWMMA FlashAttention: ALREADY ACTIVE (not a pending gain).** Source analysis
  plus `clang++ -dM -E --offload-arch=gfx1201` (which prints `#define __GFX12__ 1`)
  confirm the chain `__GFX12__` -> `RDNA4` (`vendors/hip.h`) -> `AMD_WMMA_AVAILABLE`
  (`common.cuh`) -> the WMMA MMA-FA kernel in `fattn-mma-f16.cuh` is compiled in,
  and `ggml_cuda_get_best_fattn_kernel` returns `BEST_FATTN_KERNEL_MMA_F16` for
  gfx1201 during **prefill** (large `Q->ne[1]`). For **decode** (`Q->ne[1]=1`,
  `gqa_ratio_eff=8`, product 8, not > 8) it correctly uses the tile/vec kernel:
  WMMA cannot fill its 16-wide tile at effective batch 8, and decode is
  memory-bandwidth-bound regardless. The newer-llama.cpp `GGML_HIP_ROCWMMA_FATTN`
  flag is redundant here - at pin `b81c99b` the path is arch-gated and already on
  for RDNA4. So the biggest suspected lever is already exploited; no headroom.
- **rocBLAS/Tensile gfx1201 kernels**: the wheel's rocBLAS lacks gfx1201 Tensile
  dispatch (why `FORCE_CUBLAS` measured -80% prefill); no community prebuilt logic
  pack covers gfx1201 yet (only <=gfx1150), so mmq is the active matmul path.
  Self-tuning Tensile for gfx1201 is the one remaining large prefill lever, gated
  behind ROCm maturing gfx1201 support (major effort).
- **GGML_NATIVE / AVX512**: structurally cannot help - it only adds `-march=native`
  to ggml-cpu, which is idle under `-ngl 99` full offload, and it does not touch the
  ggml-cuda/HIP kernels or the KVMem adapter (compiled as part of `llama`). It also
  trades away binary portability, so it stays OFF.
- **AMD HRX backend** (Lemonade 11.9, 2026-09): AMD claims +30-50% prefill and
  +10% non-MTP decode vs Vulkan/HIP. Not in llama.cpp yet - watch for upstream
  integration.

### H. Build validation of the reviewed source (2026-09-23)

The reviewed/fixed source was rebuilt from scratch (`build-hip.ps1 -BuildDir
build-hip2`, gfx1201): **593/593 targets, 13/13 CTest passed** (`kvmem-mtp-kv-test`
skips without the 27B model). Benchmarked with the fixed binaries (n=256, median of
3, 128K): **decode 111.9 tok/s** (tight 110.3-112.5), **prefill 3071 tok/s** -
matching or slightly above the pre-fix build (108.4 / 3047), so the correctness and
micro-optimization fixes introduce **no regression**. This rebuild also caught and
fixed a regression in `patch-rocm-headers.py`: a prior `newline=""` change broke its
regex on the wheel's **CRLF** clang headers (build failed with "guard regex matched
nothing"). It now reads with universal newlines and writes with platform newlines,
so patch->revert is a byte-exact CRLF round-trip (verified: 289->301->289 and
850->856->850 CRLF, no residue).

### I. Extended knob sweep - the remaining runtime flags (2026-09-23)

A second sweep (`tests-data/opt-sweep2.ps1`) tested the knobs the first sweep
missed, one at a time from the baseline at 128K:

| Knob | prefill | decode | retrieval_ms | needle | Verdict |
|---|---:|---:|---:|---|---|
| baseline (ubatch 512) | 3035 | 98.6 | 103 | YES | reference |
| `-ub/-b` 1024 / 2048 | 3107 / 3109 | 92.9 / 94.1 | ~102 | YES | +2% prefill, decode down; not worth it |
| `--kvmem-recent-tokens 2048` | 3041 | 96.5 | **79 (-23%)** | YES | retrieval-latency win only |
| `--kvmem-recent-tokens 4096` | 2973 | 79.3 | 23 (-78%) | **NO** | breaks recall (recent fills the 4096 budget) |
| `--kvmem-gpu-high 0.99` | 3040 | 98.2 | 112 | YES | neutral |
| `--kvmem-query-last 128` | 3029 | 94.3 | 84 | YES | neutral |
| `--kvmem-sink-tokens 512` | 3056 | 103.2 | 92 | YES | neutral-to-mild retrieval win |

A rigorous re-check (n=256, median of 3) of a combined refinement
(`--kvmem-recent-tokens 1024 --kvmem-sink-tokens 512`) gave decode **111.0 vs
baseline 115.0** - i.e. no throughput gain (within noise), needle 3/3 both.

**Retrieval cost breakdown** (`KVMEM_RETR_SUM`, baseline, `retrieval_ms=102`):
`admit 37.6 + stage_out 34.2` (70%) `+ mtp 9.9 + copy/set/occupy ~6 + score 7.9
(8%)`. Consequences: (a) retrieval is dominated by block admit/stage-out DMA and
bookkeeping, not scoring; (b) a source-level fix to `score_retrieval()` - the
per-head query sum `qh` is redundantly recomputed (and re-allocated) inside the
per-block loop though it is block-independent, so hoisting it out would cut
`score_ms` roughly in half - saves only ~5 ms of a path that is ~1% of prefill,
so it is **documented but not worth a rebuild**; (c) `--kvmem-recent-tokens` /
`--kvmem-sink-tokens` cut retrieval latency (fewer blocks re-staged) but not
throughput.

**Final verdict: the baseline recipe remains the throughput optimum; no untested
runtime knob remains.** The only safe refinement is `--kvmem-recent-tokens ~1024`
(must stay well under `--kvmem-budget`; 4096 broke recall) plus
`--kvmem-sink-tokens 512`, which cut per-turn retrieval latency ~10-23% for
multi-turn/agent use at no throughput or quality cost.

---

## 7. Recommended settings

| Situation | Command sketch |
|---|---|
| Long context (≥30K), 16 GiB card | `--kvmem -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3 --kvmem-block-tokens 128 --kvmem-budget 4096 --kvmem-cpu-gb 12` |
| Short context (<10K) | omit `--kvmem` — plain path prefills faster |
| VRAM-tight | `-ctk q8_0 -ctv q4_0` (fallback; single quality sample so far) |
| 27B + 256K workspace | `-Ctx 262144 -Budget 36864 -GenReserve 16384 -Mtp 3` |

---

## 8. Known differences and limitations vs CUDA

- **Precision canaries not run on Windows/RDNA4** for the official IQ3_S 27B /
  256K / MTP3 recipe. The CUDA side once produced garbage from IQ3 due to an nvcc
  version bug; the same caution applies to any new toolchain. After changing
  compilers, run the health checks in [`scripts/windows/README.md`](../scripts/windows/README.md)
  before trusting IQ3/IQ4 long-context recipes.
- **Upstream limits, backend-independent:** one generation cannot exceed
  `--kvmem-gen-reserve`; multi-GPU is unsupported. The NVMe/SSD spill tier is
  enabled by default in HIP builds (`build-hip.ps1` passes
  `-DKVMEM_ENABLE_NVME=ON`); see
  [kvmem-nvme-disk-tier-windows.md](kvmem-nvme-disk-tier-windows.md).
- **VMM:** `GGML_HIP_NO_VMM` defaults ON (conservative allocation). Watch the logs
  if you hit allocation failures under heavy VRAM fragmentation. HIP unified
  memory is unsupported on Windows, but KVMem uses explicit pinned copies and does
  not rely on it.
- **"Vulkan is faster than HIP on Windows"** does not apply to this fork: the KVMem
  adapter and stage-in kernels are bound to the CUDA programming model, so Vulkan
  would need a full rewrite. HIP is the minimal-change equivalence route.
- **Untested combinations:** the five unvalidated mixed `-ctk/-ctv` quant pairs,
  and the `--kv-dtype f32` FA path (HIP kernel coverage differs slightly from CUDA).

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `cannot find rocsolver.dll` on launch | ROCm DLLs were not staged. Re-run `build-hip.ps1` (it copies them next to the exe), or pass `-RocmRoot` explicitly. |
| clang errors about `isgreater`/`isless` redefinition | The ROCm header patch did not apply. Run `python scripts\windows\patch-rocm-headers.py <.venv-rocm ROCm root>`; the build script does this automatically and enables `-DKVMEM_HIP_SKIP_MATH_FWD`. To undo: `revert-guards.py <root>`. |
| `ROCm not found` | Run `rocm-install.ps1` first, or pass `-RocmRoot <path>` / set `HIP_PATH`. |
| Server starts but `/` is 404 | The Web UI was not built into `build-hip\bin\share\kvmem\ui`. See §5 → Web UI, or pass `--ui-dir`. |
| Garbage output from an IQ3/IQ4 quant | Validate the toolchain first (§8). Do a health check with a small Q8 model before long-context runs. |
| Slow first run / kernel dispatch errors | The wheel's rocBLAS may lack your `gfx` kernel dispatch; confirm `-GpuTarget` matches your card (`gfx1201` for RX 9070 XT). |

---

## 10. Reproducing the patches from scratch

If you reassemble the tree elsewhere:

```bash
# 1. cumulative KVMem feature patch (backend-agnostic)
scripts/apply-patches.sh
# 2. this port's cmake wiring (applies on top of step 1)
git -C llama.cpp apply ../patches/kvmem-hip-port-cmake.patch
```

`compat/`, `scripts/windows/*hip*`, and `tests-data/` are direct files in this
tree. To regenerate the HIP cmake delta after editing `llama.cpp/src/CMakeLists.txt`:
`python scripts/windows/emit-hip-patch.py`.

---

## 11. Code-review hardening in this revision

This revision passed a three-round code review (converged: 9 → 7 → 0 findings).
Notable fixes, all reflected in `patches/llama-kvmem-current.patch`:

- **KV-cache mask correctness (critical):** hole detection in
  `set_input_kq_mask_impl` used the stream-wide `get_used()`, which counts other
  sequences' cells and could hide real holes in a multi-sequence batch, enabling a
  stale attention mask. It now uses a per-sequence cell count
  (`llama_kv_cells::seq_pos_count`) and conservatively disables the copy-and-patch
  shortcut on any count/span mismatch, with an `int64_t` span (no overflow).
- **Robustness:** the multimodal MTP `begin()` warning no longer fires before any
  sync history exists; `getenv` trace/dump flags are cached instead of read per
  token/per layer; the out-of-tree factory include is guarded by `LLAMA_KVMEM`;
  dead code in `kvmem_capture_v` removed.
- **Tooling:** `revert-guards.py` reverses both patched clang headers byte-exactly
  and preserves LF; the Windows test scripts are portable (`$PSScriptRoot`-relative,
  `-Model`/`$env:KVMEM_MODEL`, no hard-coded developer paths) and fail loudly.

---

## 12. Further reading

- [`README.md`](../README.md) — project overview (CUDA), KVMem design, server API.
- [`scripts/windows/README.md`](../scripts/windows/README.md) — Windows CUDA build/run/validation.
- [`patches/README.md`](../patches/README.md) — patch-replay mechanics.
- [`docs/architecture.md`](architecture.md) — KVMem architecture.
- [`docs/deep-research-amd-rdna4-port.md`](deep-research-amd-rdna4-port.md) — the research that preceded this port.
- KVMem upstream Issue #44 (AMD ROCm/HIP adaptation) tracks community interest.
