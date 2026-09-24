# 在 AMD GPU 上运行 KVMem（HIP / ROCm）

> **语言：** [English](amd-hip-port.md) · 简体中文（本文件）
>
> 范围：让**同一套 KVMem 代码逻辑**（有界 GPU 槽位池 + 锁页主存分层 + 查询检索 +
> GPU stage-in + MTP/ReplaySSM）在 AMD GPU 上原生运行。路线与 llama.cpp 自身 ROCm
> 后端一致——把 CUDA 源码交给 ROCm 工具链编译，而不是重写。

根目录 [`README.md`](../README.md) 面向 NVIDIA CUDA。本文覆盖 AMD **HIP/ROCm** 移植，
已在 **Radeon RX 9070 XT（RDNA4 / gfx1201）**、原生 Windows 上验证。Linux/WSL2 路线
同理（`build-hip.ps1` 只是 `cmake -DGGML_HIP=ON` 的薄封装）。

---

## 1. 实测平台（复现条件）

§6 中所有“GPU 结果”均在下列环境实测。其他硬件/驱动/系统版本的数据会不同。

| 组件 | 取值 |
|---|---|
| GPU | Radeon RX 9070 XT，16 GiB（RDNA4，`gfx1201`） |
| CPU / 内存 | 16 核 32 线程，47 GiB |
| 操作系统 | Windows 11（原生，非 WSL） |
| ROCm | 7.14.0 wheel（HIP 7.2 运行时），安装于 `.venv-rocm` |
| 显卡驱动 | AMD Adrenalin 26.8.x |
| 编译器 | ROCm clang/clang++（来自 wheel），MSVC 14.5x Windows SDK 头 |
| 构建 | CMake + Ninja Multi-Config，`AMDGPU_TARGETS=gfx1201` |
| 模型 | Qwopus3.5-9B-Coder-MTP `Q4_K_S` GGUF（Qwen3.5 GDN 混合架构），用于冒烟/needle |

---

## 2. 移植改动清单

所有改动都在**本仓库内**（未改动 `kvmem/` 策略库）。`llama.cpp` 子模块保持在 pin
`b81c99b`；移植以“补丁 + 树外文件”交付，因此未打补丁的纯净子模块构建与上游逐位一致。

| 文件 | 作用 |
|---|---|
| `compat/hipify/cuda_runtime.h`、`cuda_fp16.h`、`cuda_bf16.h` | 把 CUDA 头文件名转发到 HIP 等价物。仅 HIP 配置会把该目录加入 include 路径，CUDA 构建零影响。 |
| `compat/hipify/kvmem_hip_defs.h` | 把 `src/adapter/*` 与 stage-in 内核用到的全部 `cuda*` 符号映射到 `hip*`（镜像 ggml 的 `vendors/hip.h`，并补齐 `cudaHostAlloc`、`cudaPointerAttributes` 等缺口）。每个 `#define` 都有 `#ifndef` 守卫，可与 `vendors/hip.h` 共存。 |
| `llama.cpp/src/CMakeLists.txt`（经 `patches/kvmem-hip-port-cmake.patch`） | `GGML_HIP` 分支：用 ROCm clang 编译 `llama-kvmem-stagein.cu`（Windows 用 `LANGUAGE CXX` + `hip::device`，Linux 用原生 `LANGUAGE HIP`），并注入 compat include 目录。 |
| `patches/llama-kvmem-current.patch` | 相对 pin 的 KVMem 累积特性补丁（内存分层、MTP、多模态、推理预算、GDN 内核）。与后端无关，CUDA 构建共用。 |
| `scripts/windows/rocm-install.ps1` | 安装 ROCm wheel（`pip --index-url https://repo.amd.com/rocm/whl-multi-arch/ rocm[libraries,devel]==7.14.0` + `rocm-sdk init`），与 llama.cpp 官方 `windows-setup-rocm` action 同源。 |
| `scripts/windows/patch-rocm-headers.py` | 幂等修复 ROCm-on-Windows 已知工具链 bug（llama.cpp issue #22570：MSVC 14.5x `<cmath>` 与 clang HIP math forward-declares 的 `isgreater`/`isless` 重载冲突）。带 `KVMEM_HIP_SKIP_MATH_FWD` 守卫，构建脚本自动启用。 |
| `scripts/windows/revert-guards.py` | 干净地回退 `patch-rocm-headers.py` 对**两个** clang 头的改动。 |
| `scripts/windows/build-hip.ps1` | 一键：打头补丁（幂等）→ 应用 KVMem 累积补丁 → 应用 HIP cmake delta → 配置（ROCm clang，`-DAMDGPU_TARGETS=gfx1201`）→ 构建 → ctest → 把 ROCm 运行时 DLL staging 到 exe 同目录。 |
| `scripts/windows/start-server-hip.ps1` | 以 KVMem 默认参数启动 `llama-kvmem-server`（无需手动配 PATH）。 |
| `scripts/windows/smoke.ps1`、`needle.ps1` + `tests-data/gen_needle.py` | 基线 vs KVMem 冒烟测试，以及 needle-in-a-haystack 检索验证。 |

---

## 3. 前置条件

1. **AMD GPU**，且 target 受 ROCm 支持。`gfx1201` = RX 9070 XT / 9070。其他卡请传
   对应的 `-GpuTarget`（如 RX 7900 系用 `gfx1100`）。
2. **Windows 11 x64**（或 Linux/WSL2）、**Visual Studio 2022 C++ Build Tools**
   （提供 Windows SDK 头/库）、`PATH` 上的 **CMake** 与 **Ninja**
   （`winget install Kitware.CMake Ninja.Ninja`）。
3. **Python 3.10–3.12**（ROCm wheel 尚不支持 3.13+）。`rocm-install.ps1` 会自动挑选
   兼容的解释器。
4. 约 **8 GiB** 空闲磁盘给 ROCm venv；以及你用 `--kvmem-cpu-gb` 申请的锁页 KV arena
   所需的主存（下面配方用 6–12 GiB）。
5. 一个 **GGUF 模型**。冒烟/needle 脚本默认用 Qwen3.5-9B MTP 量化；首次跑通用任意
   llama.cpp 兼容 GGUF 即可。

---

## 4. 小白快速上手（复制粘贴，三步）

在仓库根目录（含 `scripts\` 的目录）打开 **PowerShell**，逐条运行并等待每条结束。

```powershell
# 第 1 步 - 把 ROCm 工具链装进项目本地 venv（一次性，约 7.5 GB）。
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\rocm-install.ps1

# 第 2 步 - 构建。默认 gfx1201（RX 9070 XT）；其他卡传 -GpuTarget。
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\build-hip.ps1

# 第 3 步 - 跑一次正确性检查（告诉它你的模型在哪）。
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\smoke.ps1 -Model 'D:\models\your-model.gguf'
```

**你应该看到**

- 第 1 步以 `ROCm installed at: <路径>` 结束。
- 第 2 步以 `Done. Server: ...\build-hip\bin\Release\llama-kvmem-server.exe` 结束。
- 第 3 步打印两段并以 `smoke test PASSED (both runs exit=0)` 结束。

**启动聊天服务器**（然后浏览器打开 <http://127.0.0.1:18200/>）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\start-server-hip.ps1 -Model 'D:\models\your-model.gguf'
```

在终端按 `Ctrl+C` 停止服务器。也可用任意 OpenAI 兼容客户端指向
`http://127.0.0.1:18200/v1`。

> **无需配置 PATH。** `build-hip.ps1` 会把所有需要的 ROCm DLL（含传递依赖
> `rocsolver.dll`：ggml-hip → hipblas → rocblas → rocsolver）复制到 exe 同目录，因此
> 双击或裸命令行都能直接运行。**请勿删除或移动 `.venv-rocm`**——rocBLAS 通过该 venv
> 内固化的绝对路径解析其 Tensile 内核库。

---

## 5. 构建与运行（细节）

### 构建参数

`build-hip.ps1` 参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `-GpuTarget` | `gfx1201` | 传给 `AMDGPU_TARGETS`/`GPU_TARGETS` 的值。 |
| `-BuildDir` | `build-hip` | 输出目录。 |
| `-RocmRoot` | *（自动）* | 显式 ROCm 根；否则先从 `.venv-rocm` 解析，再看 `HIP_PATH`。 |
| `-Jobs` | `min(16, 核数)` | 并行编译数（clang HIP 编译单元峰值约 2 GB/个）。 |
| `-HostOnly` | *（关）* | 只构建 CPU 版 KVMem 主存库 + host 测试（无需 GPU 工具链）。 |

产物：`build-hip\bin\Release\llama-kvmem-server.exe` 与 `llama-kvmem-cli.exe`。

### 用 27B 长上下文配方启动服务器

```powershell
powershell -File scripts\windows\start-server-hip.ps1 `
  -Model 'D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf' `
  -Ctx 262144 -Budget 36864 -GenReserve 16384 -Mtp 3
```

脚本从 `-Model` 或环境变量 `KVMEM_MODEL` 取模型路径（不硬编码任何个人路径）。
`-Mtp 0` 关闭多 token 预测；`-Mtp 3` 启用 `--spec-type draft-mtp --spec-draft-n-max 3`。

### Web UI

服务器在 exe 上两级目录查找静态 UI（`build-hip\bin\share\kvmem\ui`）。用 Node.js 22+
构建一次：

```powershell
python scripts\build-webui.py --output ..\build-hip\bin\share\kvmem\ui
```

若缺少 `index.html`，服务器照常启动，但 `/` 返回 404（API 不受影响）。用
`--ui-dir PATH` 指定其他目录，或 `-NoUi` 关闭。

---

## 6. 验证与测试结果

结果按**所需硬件**分组。A、B 组在任何有编译器 + git 的机器上都能复现；C 组需要
RX 9070 XT（或同级），数据取自开发者实测。

### A. KVMem 主存库测试——与后端无关 ✅

覆盖 CPU 侧 KVMem 库（store、pinned tier、runtime、raw KV store、NVMe-disabled 守卫），
**不需要 GPU**。

**复现：**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\build.ps1 -HostOnly -BuildDir build-host
```

`build.ps1 -HostOnly` 以 `-DKVMEM_BUILD_LLAMA=OFF` 配置，构建五个 host 测试目标，并在
CTest 下运行。

**结果（5/5 通过，各自 exit=0）：**

| 测试 | 结果 |
|---|---|
| `kvmem_store_test` | 通过（`OK`） |
| `pinned_kv_tier_test` | 通过（`OK`） |
| `nvme_disabled_test` | 通过（`NVMe requests rejected; memory-only tests run separately`） |
| `kvmem_runtime_test` | 通过（`OK`） |
| `raw_kv_store_test` | 通过 |

runtime 测试输出结构化诊断，如
`KVMEM_TIERS cpu_bytes=16384 cpu_slots=16 nvme_bytes=0 …` 与
`KVMEM_TRACE mandatory_trim policy=topk kept=1 dropped=1 budget=4`，证明分层记账与裁剪
策略符合预期。

### B. 补丁重放完整性——任何机器可复现 ✅

子模块由补丁管理，因此纯净检出必须逐字节复现工作树。**复现**（bash，另见
[`patches/README.md`](../patches/README.md)）：

```bash
tmp=$(mktemp -d)
git -C llama.cpp archive b81c99b | tar -x -C "$tmp"
git -C "$tmp" init -q
git -C "$tmp" apply patches/llama-kvmem-current.patch     # KVMem 累积基线
git -C "$tmp" apply patches/kvmem-hip-port-cmake.patch    # HIP cmake delta
git diff --no-index llama.cpp/src "$tmp/src"              # 期望：无输出
```

**结果：** 累积补丁可干净地应用到纯净 pin `b81c99b`，HIP delta 叠加其上，重放出的树与
工作树**逐字节一致**（已核对 `llama-kv-cache.cpp`、`llama-kv-cells.h`、
`llama-graph.cpp`、`speculative.cpp`、`llama-model.cpp`、`src/CMakeLists.txt`）。基线补丁
不含 HIP，HIP 接线隔离在 delta 补丁中。

### C. RX 9070 XT 上的 GPU 结果——需要该硬件

2026-09-22 于 §1 平台实测。**复现**用各行标注的脚本；这些需要 AMD GPU，无法在没有它的
机器上重跑。

1. **构建**：593/593 目标零错误（含全部 ggml-cuda 内核为 `gfx1201` 的 codegen、
   `FA_ALL_QUANTS` 模板实例、`llama-kvmem-stagein.cu` 的 15 个内核，以及注入的
   `gdn_fold_f32` / `gdn_conv_fold_f32`）。
2. **CTest**：12/12 通过（`kvmem_store`、`pinned_tier`、`runtime`、`raw_kv_store`、
   `gdn_replay`、`server_options` 等）；无模型时 `kvmem-mtp-kv-test` 按设计 skip。
3. **设备路径**：日志出现 `ROCm0 compute buffer …` 与 `ROCm_Host compute buffer …`
   （HIP 设备 + 锁页主存缓冲均工作），并有 CUDA-Graph 复用（`graphs reused=47`），
   说明 HIP graph 路径被启用。
4. **基线推理**（Qwopus3.5-9B-Coder-MTP Q4_K_S）：prefill ≈ 134–152 tok/s（4K 冷启动段）；
   decode ≈ **85–87 tok/s**。
5. **KVMem 等效性（needle-in-a-haystack）**——`scripts\windows\needle.ps1`：2314-token
   文档，needle 置于中段（避开 sink/recent 窗口），`--kvmem-budget 512 --kvmem-cpu-gb 6
   --kv-dtype q8_0`：
   - `KVMEM_TIERS cpu_bytes=6442450944 cpu_slots=11565`——主存层激活；
   - `KVMEM_STAGE retrieval_ms=31.98 replay_n=64`——检索换入 + query replay 执行；
   - `KVMEM_TRACE gdn_ckpt/gdn_restore bytes=52691548`——GDN 递归状态 checkpoint/fold
     内核（原 CUDA-only 路径）在 `gfx1201` 上运行；
   - **模型输出 `amber-jaguar-4417`——召回正确**；
   - 长文档 prefill ≈ 2925 tok/s（缓存复用段），decode ≈ 78–80 tok/s。
6. **KVMem + MTP**（`--spec-type draft-mtp --spec-draft-n-max 3`）：needle 再次召回正确
   （`retrieval_ms=36.44`、`replay_n=63`）；MTP 与槽位池在 HIP 路径无冲突。
7. **OpenAI 兼容服务器**（KVMem + q8_0 KV + CPU spill）：`/health`、`/v1/models`、
   `/v1/chat/completions` 全部正常，响应含 `prompt_cache_hit/miss_tokens` 与连贯的
   `reasoning_content`。注：服务器端 GDN ReplaySSM 有上游模型白名单（“requires … Qwen
   27B with MTP”）；非 27B 混合模型跑服务器需加 `--spec-type none`。这是上游策略，与 AMD
   移植无关（CLI 路径不受限）。

### D. KVMem 开/关扩展性（RX 9070 XT，Qwopus3.5-9B-MTP Q4_K_S，两组均 f16 KV）

用 `tests-data\bench-kvmem.ps1`（四档 ctx）+ `tests-data\bench-256k.ps1` 复现。显存为
Windows PDH `GPU Process Memory\Dedicated Usage` 按 PID 采样峰值。KVMem 组 budget=4096、
gen_reserve=1024、block=128、CPU arena 12 GiB。档位名是 `-c` 配置，真实 prompt 长度以引擎
日志 `prompt_n` 为准。

| `-c` 配置 | 模式 | prompt tok | prefill tok/s | decode tok/s | 显存峰值 | RAM 峰值 | needle |
|---|---|---:|---:|---:|---:|---:|---|
| 8K   | plain  | 2527  | **3651** | 83.7 | 5830 MB | 4.9 GB | YES |
| 8K   | KVMem  | 2591  | 3319 | **84.5** | 5830 MB | 17.4 GB | YES |
| 32K  | plain  | 9934  | **3435** | 83.7 | 6691 MB | 5.0 GB | YES |
| 32K  | KVMem  | 9998  | 3256 | 83.1 | **5816 MB** | 17.7 GB | YES |
| 128K | plain  | 40724 | 2193 | 72.6 | 9896 MB | 5.1 GB | YES |
| 128K | KVMem  | 40788 | **3248**（+48%） | **86.4**（+19%） | **5821 MB**（−41%） | 18.7 GB | YES |
| 256K | plain  | 40724 | 1016 | 67.4 | 13244 MB（占卡 83%） | 5.2 GB | YES |
| 256K | KVMem  | 40788 | **3258**（3.2×） | **86.1**（+28%） | **5816 MB**（−56%） | 18.6 GB | YES |

**结论**

1. 短上下文（≤10K prompt）plain 略优（prefill +8~10%，显存持平）——工作集装得下时不必开
   KVMem。
2. 交叉点约 40K token；超过后 KVMem 全面反超（有界窗口使 prefill/decode 不随上下文增长而
   衰减，plain 则单调下降 3651→1016 tok/s）。
3. KVMem 显存恒定 ≈5.8 GB（槽位池不随 `-c` 增长）；256K 虚拟工作区下比 plain 省 7.4 GB——
   这正是“16 GB 卡跑 27B + 256K”配方成立的定量基础。
4. 代价是主存 +13 GB（锁页 arena，48 GB 机器可承受）与检索失效风险——四档 needle 全部命中，
   budget 4096 对 40K 工作区无质量损失。
5. 256K 档 KVMem 模型加载快 2.7×（16.2 s vs 43.1 s；plain 需预分配全量 KV）。

### E. 调优矩阵（RX 9070 XT）

每个候选都在 `bench_128k` prompt（实测 40.7K token）、ctx 139264、greedy、`-n 128` 下实跑，
以 needle 召回作为质量门；预热一次后取第二次。指标：`KVMEM_PERF prompt_toks`（prefill）与
`KVMEM_GEN_WALL toks`（端到端 decode）。用 `tests-data\opt-matrix-a.ps1`、
`opt-matrix-c.ps1`、`opt-eval-b.ps1`、`opt-final.ps1` 复现。

| # | 方案 | 类型 | prefill | decode(wall) | 显存 | needle | 裁决 |
|---|---|---|---:|---:|---:|---|---|
| — | 基线 f16 KV 无 MTP | — | 3202 | 72.4 | 5821 | YES | 基准 |
| A2 | **q8_0 KV + MTP3** | 参数 | **3218** | **87.5** | **5742** | YES | ✅ **采纳（最终）** |
| A3 | + ubatch 2048 | 参数 | 3232 | 75.1 | 6055 | YES | ❌ |
| A4 | block-tokens 32 | 参数 | 2906 | 77.3 | 5742 | YES | ❌ |
| A5 | budget 8192 | 参数 | 2817 | 79.7 | 5842 | YES | ❌ |
| C1 | MTP draft-n-max 4 | 参数 | 3104 | 65.6 | 5793 | YES | ❌ 接受率下降 |
| C2 | block-tokens 256 | 参数 | 3159 | 77.2 | 5742 | YES | ❌ |
| C3 | budget 2048 | 参数 | 3325 | 75.6 | 5700 | YES | ❌ decode 降 |
| C4 | harvest-v | 参数 | 3138 | 84.1 | 5772 | YES | ❌ 噪声内无收益 |
| A7/B | q8_0 K + q4_0 V | 参数 | 3213 | 86.2 | 5720 | YES | ➖ 备选（显存敏感时；质量样本单一） |
| B | `GGML_HIP_NO_VMM=OFF` + `NATIVE=ON` | 重编译 | 1537 | 72.1 | 5819 | YES | ❌ **VMM 有害：f16 prefill −52%** |
| C | `GGML_CUDA_FORCE_CUBLAS=ON` | 重编译 | 658 | 72.6 | 6200 | YES* | ❌ **−80% prefill + rocBLAS 错误** |
| — | hipBLASLt（`GGML_HIPBLAS`） | 重编译 | — | — | — | — | 🚫 排除：pin `b81c99b` 无集成，升 pin 会破坏 fork 补丁锚点 |
| — | FA on/off | 参数 | — | — | — | — | 🚫 排除：代码证据——q8_0 V 强制要求 FA enabled（`llama-context.cpp`），`auto` 已生效 |
| — | 线程 `-t` 调优 | 参数 | — | — | — | — | 🚫 排除：`-ngl 99` 全卸载下 CPU 仅采样；B2 vs A2 差异 <2%（噪声） |

\* C 组借用 LM Studio 的 Tensile 库后才出正确结果；wheel 自带 rocBLAS 缺 `gfx1201` 内核分发——
属 ROCm-on-Windows 平台成熟度问题，与调研阶段 TheRock 矩阵结论一致。

**最终规格：** 二进制在 `build-hip\bin\Release\`（默认 flags：`NO_VMM=ON`、mmq/hipBLAS
自适应、`GRAPHS=ON`、`FA_ALL_QUANTS=ON`）。长上下文（≥30K）推荐：`--kvmem -ctk q8_0
-ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3 --kvmem-block-tokens 128
--kvmem-budget 4096+（随显存） --kvmem-cpu-gb 12`。短上下文（<10K）：不加 `--kvmem`
（纯 llama.cpp 路径 prefill 更快）。相对初版基线累计收益：**decode +21%（72.4→87.5）**、
显存 −79 MB、prefill 持平、needle 全命中（无质量回归）。

---

## 6b. 128K 对比 LM Studio + 优化研究（2026-09-23，同口径实测）

### F. 128K 与 LM Studio 正面对比

同机（RX 9070 XT）、同模型、同 `bench_128k.txt`（**40724 token**）、ctx 139264、
greedy、**n=256 稳态**、取 3 次中位数（方法学见下）。LM Studio = 其 Vulkan 运行时、
f16 全量 KV、MTP draft-n 3、~14.6 GB 显存；KVMem-HIP = 本移植、q8_0 KV、MTP3、
budget 4096、block 128、~5.8 GB 显存。复现：`tests-data/bench_server.py`（LM Studio）
与 `tests-data/opt-decode.ps1`（KVMem）。

| @128K（40.7K prompt） | prefill tok/s | decode tok/s（稳态） | 显存 |
|---|---:|---:|---:|
| LM Studio（Vulkan, f16 全量 KV, MTP3） | 2387 | 102-104 | 14.6 GB |
| **KVMem-HIP（q8_0, MTP3, 有界窗口）** | **3050-3220** | **108-111** | **5.8 GB** |
| **差距（KVMem vs LM Studio）** | **+28-35%** | **+4-7%** | **-60%** |

要点：
- **decode 基本持平**。两者都用 MTP3，都接近本卡上 5.1GB Q4_K_S 模型的 ~125 tok/s
  显存带宽屋顶线（KVMem 86-89%，LM Studio 82-83%）。KVMem 的有界窗口对 decode 提速有限，
  因为 40K 上下文 + `n_head_kv=4` 时 decode 瓶颈是权重矩阵乘，而非 KV 注意力。
- **KVMem prefill 快 ~30%**（有界窗口 => stage-in 40K prompt 时注意力工作量更小），
  **显存少 60%**（5.8 vs 14.6 GB）。
- **显存是决定性的鲁棒性差异**。LM Studio 的 14.6 GB 几乎顶满 16 GB 卡；只要桌面多占
  一点显存就会溢出到共享内存、decode 崩塌（我们实测到在一个游离 5.8GB 进程争抢下
  102 -> 23.5 tok/s）。KVMem 的 5.8 GB 不受影响——这正是 16GB 卡能跑 256K 的原因。

> **测量方法学（重要）**。单次 n=64 生成会严重低估 decode：GPU 频率与 MTP 接受率在
> 前 ~1-2 秒才爬升，同一 KVMem 配置 n=64 读到 ~70 tok/s，n=256 读到 ~108。decode 基准
> 务必用 **n>=256、取 >=3 次中位数**。因此 6D/6E 的短生成表格低估稳态 decode ~20-30%；
> 最终配方的修正稳态值为 **~108-111 tok/s**。

### G. 优化研究——所有运行时线索已测

在 `bench_128k.txt` 上从基线逐项变动（`tests-data/opt-sweep.ps1` 测 prefill/检索/显存；
`tests-data/opt-decode.ps1` 以 n=256、3 次中位数测 decode）：

| 线索 | 结果 | 裁决 |
|---|---|---|
| KV dtype q4_0/q5_0 V vs q8_0 | decode 106.5/106 vs 108.4；prefill 持平 | 无收益（KV 非瓶颈）；为质量保留 q8_0 |
| MTP 关 vs n-max 3 | 80.7 vs 108.4 | **MTP = +34%，主导杆杆（保持开）** |
| MTP n-max 2 vs 3 | 105.4 vs 108.4 | n-max 3 略优 |
| `--spec-kv-dtype q4_0` | 108.5 vs 108.4 | 无变化 |
| budget 4096 / 6144 / 8192 | prefill 3218 / 2988 / 2818；检索 114 / 156 / 191 ms | **4096 最优** |
| block-tokens 128 / 64 | prefill 3218 / 3090；检索 114 / 151 ms | **128 最优** |
| needle 召回 | 所有配置 3/3 | 质量无损失 |

**结论：出厂配方（q8_0/q8_0、MTP3、budget 4096、block 128）就是运行时最优——没有任何
实测变体能超过它。** 同一最优在 32K 档复扫依然成立（基线 prefill 3232、decode 同类最佳；
budget 4096、block 128 再次胜过 6144/8192 与 64；关 MTP 损 -22% decode），因此配方在
32K-128K 跨档鲁棒。prompt 低于 ~10K 时 KVMem 窗口超过 prompt，纯路径（不加 `--kvmem`）
prefill 更快——KVMem 用于 >=30K。检索重排（`layout_d2h` + `layout_h2d` ~ 1.9 ms）与
`retrieval_ms` 总时（~114-125 ms，每轮一次性，约占 13s prefill 的 1%）都不是 decode 瓶颈，
因此同步塔缩 / stage-in 重叠类微优化在此价值很低（其实测上限 <=1.5% 且仅作用于一次性路径）。

**内核级线索——已查到头：**
- **rocWMMA FlashAttention：已经生效（不是待获取的收益）。** 源码分析 +
  `clang++ -dM -E --offload-arch=gfx1201`（输出 `#define __GFX12__ 1`）确认链路
  `__GFX12__` -> `RDNA4`（`vendors/hip.h`）-> `AMD_WMMA_AVAILABLE`（`common.cuh`）->
  `fattn-mma-f16.cuh` 的 WMMA MMA-FA 内核已编入，且 `ggml_cuda_get_best_fattn_kernel`
  在 **prefill**（`Q->ne[1]` 大）时为 gfx1201 返回 `BEST_FATTN_KERNEL_MMA_F16`。
  **decode**（`Q->ne[1]=1`、`gqa_ratio_eff=8`、乘积 8，不 >8）则正确地用 tile/vec 内核：
  WMMA 在有效 batch 8 时填不满 16 宽 tile，且 decode 本就受显存带宽限制。新版
  llama.cpp 的 `GGML_HIP_ROCWMMA_FATTN` 开关在此多余——pin `b81c99b` 下该路径按 arch
  门控、RDNA4 已自动启用。所以最大的疑似杆杆已被利用，无额外空间。
- **rocBLAS/Tensile gfx1201 内核**：wheel 的 rocBLAS 缺 gfx1201 的 Tensile 分发（所以
  `FORCE_CUBLAS` 实测 -80% prefill）；目前无社区预编译 logic 包覆盖 gfx1201（仅 <=gfx1150），
  故 mmq 是当前矩阵乘路径。自行为 gfx1201 调优 Tensile 是剩下唯一的大 prefill 杆杆，
  但被 ROCm 对 gfx1201 的成熟度阻塞（工量大）。
- **GGML_NATIVE / AVX512**：结构上无法生效——它只给 ggml-cpu 加 `-march=native`，
  而 `-ngl 99` 全卸载下 ggml-cpu 基本闲置，且它不影响 ggml-cuda/HIP 内核与 KVMem 适配层
  （后者作为 `llama` 一部分编译）。它还牺牲二进制可移植性，故保持 OFF。
- **AMD HRX 后端**（Lemonade 11.9，2026-09）：AMD 宣称相比 Vulkan/HIP prefill +30-50%、
  非-MTP decode +10%。尚未进入 llama.cpp——关注上游集成。

### H. 对已评审源码的构建验证（2026-09-23）

将已评审/修复的源码从头重建（`build-hip.ps1 -BuildDir build-hip2`，gfx1201）：
**593/593 目标、13/13 CTest 全部通过**（`kvmem-mtp-kv-test` 无 27B 模型时 skip）。
用修复后二进制实测（n=256、3 次中位数、128K）：**decode 111.9 tok/s**（紧凑 110.3-112.5）、
**prefill 3071 tok/s**——持平或略优于修复前构建（108.4 / 3047），因此正确性与微优化修复
**无性能回退**。本次重建还捕获并修复了 `patch-rocm-headers.py` 的一个回退：之前的
`newline=""` 改动在 wheel 的 **CRLF** clang 头上使正则失配（构建报“guard regex matched
nothing”）。现改为通用换行读取 + 平台换行写入，patch->revert 为字节级 CRLF 往返
（已验：289->301->289 与 850->856->850 CRLF，无残留）。

### I. 扩展旋钮扫描——剩余运行时开关（2026-09-23）

第二轮扫描（`tests-data/opt-sweep2.ps1`）在 128K 上逐项测试了首轮遗漏的开关：

| 开关 | prefill | decode | retrieval_ms | needle | 裁决 |
|---|---:|---:|---:|---|---|
| 基线（ubatch 512） | 3035 | 98.6 | 103 | YES | 参考 |
| `-ub/-b` 1024 / 2048 | 3107 / 3109 | 92.9 / 94.1 | ~102 | YES | prefill +2%、decode 降；不划算 |
| `--kvmem-recent-tokens 2048` | 3041 | 96.5 | **79 (-23%)** | YES | 仅降检索延迟 |
| `--kvmem-recent-tokens 4096` | 2973 | 79.3 | 23 (-78%) | **NO** | 破坏召回（recent 占满 4096 预算） |
| `--kvmem-gpu-high 0.99` | 3040 | 98.2 | 112 | YES | 中性 |
| `--kvmem-query-last 128` | 3029 | 94.3 | 84 | YES | 中性 |
| `--kvmem-sink-tokens 512` | 3056 | 103.2 | 92 | YES | 中性偏轻微降检索 |

组合精调（`--kvmem-recent-tokens 1024 --kvmem-sink-tokens 512`）严格复测（n=256、3 次中位数）：
decode **111.0 vs 基线 115.0**——无吞吐收益（噪声内），needle 均 3/3。

**检索耗时分解**（`KVMEM_RETR_SUM`，基线，`retrieval_ms=102`）：`admit 37.6 + stage_out 34.2`（70%）
`+ mtp 9.9 + copy/set/occupy ~6 + score 7.9（8%）`。结论：(a) 检索主要是块 admit/stage-out 的 DMA 与
记账，不是打分；(b) `score_retrieval()` 里 per-head 查询和 `qh` 在按块循环内被冗余地重算+重分配
（它与块无关），提出循环可省约一半 `score_ms`——但只省 ~5ms（该路径约占 prefill 1%），**记录但不值得
重编译**；(c) `--kvmem-recent-tokens`/`--kvmem-sink-tokens` 降检索延迟而非吞吐。

**最终结论：基线配方仍是吞吐最优；无未测运行时开关。** 唯一安全的精调是 `--kvmem-recent-tokens ~1024`
（必须远小于 `--kvmem-budget`；4096 会破坏召回）+ `--kvmem-sink-tokens 512`，为多轮/agent 场景降每轮
检索延迟 ~10-23%，且无吞吐/质量代价。

---

## 7. 推荐设置

| 场景 | 命令要点 |
|---|---|
| 长上下文（≥30K），16 GiB 卡 | `--kvmem -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3 --kvmem-block-tokens 128 --kvmem-budget 4096 --kvmem-cpu-gb 12` |
| 短上下文（<10K） | 省略 `--kvmem`——纯路径 prefill 更快 |
| 显存紧张 | `-ctk q8_0 -ctv q4_0`（备选；目前质量样本单一） |
| 27B + 256K 工作区 | `-Ctx 262144 -Budget 36864 -GenReserve 16384 -Mtp 3` |

---

## 8. 与 CUDA 版的已知差异与限制

- **未在 Windows/RDNA4 上做精度 canary**（官方 IQ3_S 27B / 256K / MTP3 配方）。CUDA 侧曾因
  nvcc 版本 bug 导致 IQ3 乱码；换任何新工具链都要同样谨慎。更换编译器后，请先按
  [`scripts/windows/README.md`](../scripts/windows/README.md) 的 validation 流程做健康检查，
  再信任 IQ3/IQ4 长上下文配方。
- **上游固有限制（与后端无关）：** 单轮生成不能超过 `--kvmem-gen-reserve`；不支持多 GPU；
  NVMe 层未实现（见根 README）。
- **VMM：** `GGML_HIP_NO_VMM` 默认 ON（保守分配）。显存碎片严重导致分配失败时留意日志。
  HIP 统一内存 Windows 不支持，但 KVMem 用显式锁页拷贝，不依赖它。
- **“Windows 上 Vulkan 比 HIP 快”** 的社区结论不适用本 fork：KVMem 适配层与 stage-in 内核
  绑定 CUDA 编程模型，Vulkan 需完全重写；HIP 是最小改动的等效路线。
- **未测试组合：** 五种未验证的 `-ctk/-ctv` 混合量化对，以及 `--kv-dtype f32` 的 FA 路径
  （HIP 内核支持面与 CUDA 略有差异）。

---

## 9. 故障排查

| 现象 | 原因 / 解决 |
|---|---|
| 启动报 `cannot find rocsolver.dll` | ROCm DLL 未 staging。重跑 `build-hip.ps1`（会复制到 exe 同目录），或显式传 `-RocmRoot`。 |
| clang 报 `isgreater`/`isless` 重定义 | ROCm 头补丁未生效。运行 `python scripts\windows\patch-rocm-headers.py <.venv-rocm 的 ROCm 根>`；构建脚本会自动执行并启用 `-DKVMEM_HIP_SKIP_MATH_FWD`。撤销用 `revert-guards.py <根>`。 |
| `ROCm not found` | 先跑 `rocm-install.ps1`，或传 `-RocmRoot <路径>` / 设 `HIP_PATH`。 |
| 服务器启动但 `/` 404 | Web UI 未构建进 `build-hip\bin\share\kvmem\ui`。见 §5 → Web UI，或传 `--ui-dir`。 |
| IQ3/IQ4 量化输出乱码 | 先验证工具链（§8）。长上下文前先用小 Q8 模型做健康检查。 |
| 首次运行慢 / 内核分发错误 | wheel 的 rocBLAS 可能缺你的 `gfx` 内核分发；确认 `-GpuTarget` 与显卡匹配（RX 9070 XT 为 `gfx1201`）。 |

---

## 10. 从头复现补丁

若在别处重新组装本树：

```bash
# 1. KVMem 累积特性补丁（与后端无关）
scripts/apply-patches.sh
# 2. 本移植的 cmake 接线（叠加在第 1 步之后）
git -C llama.cpp apply ../patches/kvmem-hip-port-cmake.patch
```

`compat/`、`scripts/windows/*hip*`、`tests-data/` 为本树直接文件。修改
`llama.cpp/src/CMakeLists.txt` 后重新生成 HIP cmake delta：
`python scripts/windows/emit-hip-patch.py`。

---

## 11. 本次修订的代码评审加固

本次修订通过三轮代码评审（收敛：9 → 7 → 0 findings）。主要修复均已反映在
`patches/llama-kvmem-current.patch`：

- **KV-cache mask 正确性（关键）：** `set_input_kq_mask_impl` 的空洞检测原用全流
  `get_used()`，它会把其他 sequence 的 cell 计入，可能在多序列 batch 中掩盖真实空洞，从而
  启用陈旧注意力 mask。现改用按序列的 cell 计数（`llama_kv_cells::seq_pos_count`），并在
  计数/跨度不一致时保守地关闭 copy-and-patch 捷径，跨度用 `int64_t`（无溢出）。
- **健壮性：** 多模态 MTP `begin()` 的告警不再在尚无 sync 历史时误触发；`getenv` 的
  trace/dump 标志改为缓存读取，而非每 token/每层读取；树外 factory 头 include 由
  `LLAMA_KVMEM` 守卫；移除 `kvmem_capture_v` 中的死代码。
- **工具链：** `revert-guards.py` 现逐字节精确回退两个 clang 头并保留 LF；Windows 测试脚本
  可移植（`$PSScriptRoot` 相对、`-Model`/`$env:KVMEM_MODEL`、无硬编码个人路径）且失败即报错。

---

## 12. 延伸阅读

- [`README.md`](../README.md)——项目总览（CUDA）、KVMem 设计、服务器 API。
- [`scripts/windows/README.md`](../scripts/windows/README.md)——Windows CUDA 构建/运行/验证。
- [`patches/README.md`](../patches/README.md)——补丁重放机制。
- [`docs/architecture.md`](architecture.md)——KVMem 架构。
- [`docs/deep-research-amd-rdna4-port.md`](deep-research-amd-rdna4-port.md)——本移植前的深度调研。
- KVMem 上游 Issue #44（AMD ROCm/HIP 适配）跟踪社区需求。
