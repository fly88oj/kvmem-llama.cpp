# Deep Research: KVMem-llama.cpp 实现原理与 AMD RDNA4（RX 9070 XT）等效移植

> Generated 2026-09-22 | Depth: deep | Sources: 45 | 模式：调研 + 本机可运行实现（AMD 版 KVMem）
>
> **Note (EN):** Supplementary deep-research background (45 sources) that preceded
> the port; written in Chinese. For the operational guide, see
> [amd-hip-port.md](amd-hip-port.md) (English) /
> [amd-hip-port.zh-CN.md](amd-hip-port.zh-CN.md) (中文).

## TL;DR

KVMem 是一个"KV 上下文虚拟化"系统：它把 llama.cpp 的 KV cache 改造成**有界 GPU 块槽位池 + 主存（RAM）历史块仓库 + 按查询检索**的三层结构，从而在 16 GiB 显存上以近无损质量运行 256K 上下文的 Qwen3.8-27B [1][3][4]。源码级分析表明它对 CUDA 的耦合面**非常窄且有明确边界**（一个 1074 行的 stage-in kernel 文件 + 约 40 个 CUDA Runtime API 符号），而 llama.cpp 的 ROCm 后端本质上就是"用 hipcc 编译同一套 ggml-cuda CUDA 源码"[43][44][45]——因此 AMD 等效移植的正确路线是 **HIP 后端移植（hipify shim + gfx1201 构建）**，而不是重写为 Vulkan。本机 RX 9070 XT（gfx1201，16 GiB，PCIe 5.0 x16）在 ROCm 7.2.0 Windows 官方支持矩阵内 [7]，且 llama.cpp 官方 Windows 发布 CI 已经在为 gfx1201 编译 HIP 后端 [43]——移植在工程上可行，本报告随附完整实现方案与执行记录。

## Executive Summary

本次调研回答三个问题：KVMem 是怎么工作的？它的 CUDA 依赖到底在哪里？AMD（本机 RDNA4）上如何做逻辑等效实现？

**原理层面**，KVMem（arXiv 2609.04852，Di Chai 等，2026-09-04 提交）把智能体长会话的 KV 历史视为一个"虚拟工作区"：GPU 上只保留一个容量为 `budget + gen_reserve` 的块槽位池（块粒度默认 128 token），写满的 KV 块以 GPU 打包格式（q8_0/q5_0/q4_0 量化 KV）异步换出到锁页主存；每个 agent 步骤用"最后一条 user 消息"作为查询，通过轻量 mean-K 注意力空间索引检索相关历史块，按时间顺序装入 GPU 窗口，再让完全未被修改的 Flash Attention kernel 对这个有界窗口做稠密注意力 [1][4][5]。论文的三层设计（GPU/主存/NVMe）在这个 llama.cpp 移植版里只实现了前两层，NVMe 明确未做 [1][3]。作者在 RTX 5060 Ti 16 GiB 上宣称 256K 工作区下 decode 32-33 tok/s、LongMemEval-S 精度 85.6%（对比全量 256K 的 86.6%）[1]。

**耦合面层面**，克隆仓库源码给出决定性证据：策略库 `kvmem/`（约 4700 行）**零 CUDA 依赖、零 llama.cpp 依赖**，纯主机 C++ [4]；GPU 侧只有 `src/adapter/` 目录直接调用 CUDA Runtime API——去重后约 **40 个符号**（事件/流/锁页分配/memcpy/指针属性），其中 90% 已被 llama.cpp 自带的 `vendors/hip.h` 兼容层映射覆盖 [45]；唯一的 kernel 文件 `llama-kvmem-stagein.cu`（1074 行、15 处 kernel launch）只使用 FWHT、反量化、RoPE、量化、gather/scatter 这类基础可移植原语 [4]。llama.cpp 对 fork 的补丁面（38 个文件）中真正触到 GPU 代码的只有 `ggml-cuda/gated_delta_net.*`，而 ggml-hip 构建正是用文件通配符直接重编译整个 `ggml-cuda/*.cu`——这些 kernel 修改会**自动**随 HIP 构建生效 [43][44]。

**平台层面**，ROCm 对 Windows 消费级显卡的支持在 2025-09（ROCm 6.4.4 公开预览）到 2026（HIP SDK 7.2.0 / ROCm 7.x TheRock 统一分发）之间完成了对 RDNA4 的官方覆盖：AMD 官方 Windows 系统需求矩阵明确列出 RX 9070 XT = gfx1201，Runtime 与 HIP SDK 均为支持状态 [7][14]；llama.cpp 的 Windows 发布 CI 使用 ROCm 7.14.0 wheel 以 `-DGGML_HIP=ON` 为包括 gfx1201 在内的 20 个 AMD GPU 目标构建 `ggml-hip.dll` [42][43]。矛盾点在于：社区二手转述称 TheRock 支持矩阵上 Windows RDNA4 仍停在 "Build Passing"（未达到 Sanity-Tested/Release-Ready）[17][23]，且多个实测来源认为当下 Windows + AMD 上 llama.cpp 的 **Vulkan 后端更快更稳**[16][17][24]。对 KVMem 这类深度绑定 CUDA 编程模型的 fork 来说，这不是平等的选择：Vulkan 路径需要重写整个适配层与 stage-in kernel（月级工作量），而 HIP 路径是把同一份 CUDA 源码交给 AMD 的编译器——这也是 llama.cpp ROCm 后端自身的存在方式 [44][45]。

**结论**：AMD 等效实现采用 HIP 路线，工作量集中在（a）一个 ~120 行的 `cuda_runtime.h/cuda_fp16.h/cuda_bf16.h` 兼容头（补上 `cudaHostAlloc`、`cudaPointerAttributes` 等 hip.h 未映射的少数符号）；（b）CMake 的 HIP 分支接线（仿照 patch 0004 的 CUDA 分支与 llama.cpp 官方 CI 的 Windows 构建参数）；（c）Windows 平台差异适配（WDDM 锁页内存、HIP 无统一内存——KVMem 恰好不依赖统一内存，只用显式锁页拷贝，天然规避了这个坑 [10][13]）。

---

## 1. 现状：KVMem 的实现原理与实现方案 [Confidence: High]

### 1.1 项目定位与研究背景

`github.com/kvmem/kvmem-llama.cpp` 是一个真实存在的公开仓库（C++，约 431 star，最后更新 2026-09-21），是学术项目 KVMem 的 llama.cpp 移植版 [1][2]。配套论文《KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU》（arXiv 2609.04852，2026-09-04）把它定位为"KV 上下文虚拟化系统"：解决的核心痛点是 LLM 智能体持久工作区的历史长度同时超出 GPU KV 容量和模型原生上下文窗口，而现有做法（上下文压缩成摘要、或以文本形式外部检索再重新 prefill）要么丢失细粒度执行证据，要么重复计算模型已经处理过的内容 [3]。

组织名下另有 `kvmem-qw3`（Apache-2.0，面向 Q8 的 CUDA 原生运行时，主要在 RTX PRO 6000 上测试）——llama.cpp 移植版是从那个自研引擎里"剥出策略层、换上 llama.cpp 做推理引擎"的产物 [1][2][4]。kvmem 组织无公开成员，社区渠道（QQ 群 1040777853、Bilibili 测试者）显示作者团队位于中国 [1]。项目发布仅三周，截至调研时不存在第三方深度报道；仓库 Issue #44（2026-09-21）询问 AMD ROCm/HIP 适配计划，无人回复——本任务恰好是该缺口的直接填补 [33]。

### 1.2 核心机制：块槽位池 + 主存分层 + 检索式工作集

KVMem 的数据通路可以概括为五步（全部细节有 README、architecture.md、modification-plan.md 三处一手互证 [1][4][5]）：

1. **有界 GPU 槽位池。** GPU 上的注意力 KV 不是连续大缓存，而是 `budget + gen_reserve` 个槽位（每槽 `block_tokens`=128 个 cell）。每个逻辑块占一个槽；cell 上记录**原始单调 token 位置**（`pos`），槽号本身不充当 RoPE 坐标——这是它与 paged attention 的本质区别：不改任何注意力 kernel，kernel 看到的仍是普通的密集 KV 张量，只是长度有界 [4][5]。
2. **块写满即异步换出。** 当一个块在 GPU 上写满，其打包好的 GPU 格式 K/V（产品配方为 q8_0 或 q5_0 量化）通过异步 D2H 拷贝进锁页主存缓冲区，与该块后续 prefill/decode 重叠执行；块检索分数用**捕获于首次写入的 pre-RoPE F32 mean-K**（解码期在 GPU 上维护 running sum，块满时落盘），避免从量化缓存反推内容键 [1][4][5]。
3. **按查询检索。** 每个 agent 步骤以"最后一条 user 消息"为查询，对主存中的历史块表打分（mean-K 点积），选出与 `budget` 匹配的子集，按时间顺序装入 GPU 槽位。重选择（reselect）是一次 `KvMemPlan` diff：已驻留的选中块原地保留，只有变化块走 H2D/D2H，被移除的块用 `seq_rm` 标记——"注意力 kernel 与原始位置保持不变，重选只传输发生变化的块" [1][4][5]。
4. **GPU stage-in 变体流水线。** 主存中的 K 块以"打包量化 + 原始位置"存储，装回 GPU 时若目标槽的窗口坐标与原始位置不同，需要现场做坐标变换。v0.7.0 起这一步全部下推到 GPU：`llama-kvmem-stagein.cu` 里一条 5 段流水线——H2D 打包 q8/q4 → 反量化 F32 → NeoX RoPE（按 orig pos→新 pos）→ Walsh-Hadamard 变换（配合 IQ3/IQ4 的旋转量化）→ 重量化写入 KV cache——用一个 32 MiB 的 GPU slab 和双缓冲锁页主机批处理完成 [4][6]。反向（stage-out）有一个 gather kernel 把散布的 GPU V 行打包进 slab 再做一次 D2H [6]。
5. **生成预留与 pin 语义。** 检索装入的块被钉住（decode 不许逐出），新生成的 token 只能消费 `gen_reserve` 分区——这就是 README 里"单轮生成（含思考）不能超过 `--kvmem-gen-reserve`"已知限制的技术根源；规划中的修复是在 `gen_reserve` 内做环形逐出，每次只逐一块已满的生成块回主存 [1][4]。

与竞品方案的对照是论文和 README 都明确处理的：Raymond Huang 的自适应 KV 流式缓存（adaptive KV streaming）保留**全历史注意力**，逐层预取卸载 KV 进可复用 GPU 缓冲，解码时间与 PCIe 流量随上下文线性增长；KVMem 则把注意力可见的 KV 限制在有界窗口内，用检索质量换 PCIe 流量 [1]。两者互补而非替代：前者无损但越来越慢，后者近似无损（LongMemEval-S 85.6% vs 86.6%；AgentLongBench 任务成功率 60.9% vs 59.5%，反而更高）且解码时间不随工作区长度恶化 [1][3]。

### 1.3 工程结构：三层解耦 + 钉版 submodule + 补丁重放

移植版把系统切成三个纯度递增的层（这是它做 AMD 移植最重要的架构红利）[1][4][5]：

- `kvmem/`（约 4700 行）：块表、检索打分、分层存储（含未启用的 NVMe 层）、窗口装配的**纯主机策略库**——硬约束"零 `#include` llama.cpp 头文件"，实测源码中也不含任何 CUDA 符号（grep 仅命中注释与一处无关词）[4][5]；
- `src/adapter/`（约 6600 行）：唯一允许触碰 llama.cpp 内部头与 CUDA Runtime API 的地方，实现 llama.cpp 可插拔内存接口 `llama_memory_i`，含 hybrid（Qwen3.5/3.6 注意力+DeltaNet 混合）与 MTP 跟随槽位池两个变体 [4][5]；
- `llama.cpp/` submodule 钉在 `b81c99b`，所有上游改动收敛为一个可重放的累积补丁（38 个文件），补丁面被硬约束限定在"memory 工厂 hook、build_attn 的 Q/K capture、CLI/server 元数据"三类，超出即视为方案回退 [1][5]。

值得注意的细节：为 Qwen3.8 混合模型的 GDN（Gated DeltaNet）递归状态，补丁在 `ggml-cuda/gated_delta_net.cu` 里加了两个自定义 kernel（`gdn_fold_f32`、`gdn_conv_fold_f32`）并以 `ggml_backend_cuda_gdn_fold` 导出给适配器（ReplaySSM：长上下文跳过的 token 用 fold 恢复递归状态而非重算）[5]。这两个 kernel 是继 stagein 之后第二块 GPU 专属代码，但位置在 ggml-cuda 内——HIP 构建会自动以 hipcc 重编译它们（见 §3.3）[44]。

### 1.4 关键参数与实测基线（RTX 5060 Ti 16 GiB）

产品配方（IQ3 推荐）：`-c 262144`（虚拟工作区）、`--kvmem-budget 36864`（检索窗口）、`--kvmem-gen-reserve 16384`（生成预留）、`-ctk/-ctv q8_0`、MTP3 + ReplaySSM [1]。实测：262K token 工具调用回环下聚合 decode 31.7 tok/s、首算 prefill 437 tok/s、总体 prefill（含缓存管理）242 tok/s、显存峰值 15617 MiB、主机 RSS 峰值 13.5 GiB [1]。参照硬件是 16 GiB 的 RTX 5060 Ti（446 GB/s GDDR7）——与本机 RX 9070 XT（16 GiB，644.6 GB/s GDDR6，PCIe 5.0 x16）处于同一显存档位、更高带宽 [40]。README 明示"AMD/ROCm 与 Metal 后端需要集成工作"——移植没有现成路标 [1]。

---

## 2. CUDA 耦合面：源码级解剖 [Confidence: High]

这一节是全部调研中价值最高的部分，因为移植工作量完全由耦合面决定，而证据来自本地克隆的一手源码（Tier 1），非二手转述。

### 2.1 主机库：零依赖

对 `kvmem/` 全目录做 CUDA 符号扫描，唯一命中是 `pinned_kv_tier.hpp` 的注释："它拥有零内存、发出零拷贝——由 CUDA 后端供给 cudaHostAlloc 缓冲并执行实际 D2H/H2D" [4]。即分层存储的**策略**（块表、slot 映射、检索）与**机制**（DMA、锁页）被刻意切开，机制全部沉在 adapter 层。AMD 移植对 `kvmem/` 库零改动。

### 2.2 Adapter 的 CUDA API 面：约 40 个符号

对 `src/adapter/` + `tools/` 的全部 `.cpp/.h/.cu` 做符号提取去重，得到的完整清单（出现次数）为：`cudaError_t/cudaSuccess/cudaGetErrorString/cudaGetLastError`（错误处理）、`cudaMalloc/cudaFree/cudaMallocHost/cudaFreeHost/cudaHostAlloc`（分配）、`cudaMemcpy/cudaMemcpyAsync/cudaMemcpyKind` 与三个 kind 常量（拷贝）、`cudaStream_t/cudaStreamCreateWithFlags/cudaStreamDestroy/cudaStreamSynchronize/cudaStreamWaitEvent/cudaStreamNonBlocking/cudaStreamPerThread`（流）、`cudaEvent_t/cudaEventCreateWithFlags/cudaEventRecord/cudaEventSynchronize/cudaEventDestroy/cudaEventDisableTiming`（事件）、`cudaSetDevice/cudaGetDevice/cudaDeviceSynchronize/cudaMemsetAsync/cudaPointerAttributes/cudaPointerGetAttributes`（设备与杂项），加上 `cuda_fp16.h/cuda_bf16.h/cuda_runtime.h` 三个头 [4]。

把这个清单对照 llama.cpp ROCm 后端自带的 `ggml/src/ggml-cuda/vendors/hip.h`（cuda→hip 的 `#define` 兼容层）逐一核验：除 `cudaHostAlloc`（hip.h 只映射了 `cudaMallocHost`）、`cudaPointerAttributes`/`cudaPointerGetAttributes` 和三个头文件名本身之外，**其余全部已有现成映射**，包括 `cudaStreamPerThread`（HIP 侧同名存在）[45]。换言之，adapter 的 CUDA 面几乎就是一个"被 llama.cpp 官方兼容层预先消化过的子集"——kvmem 作者写代码时用的正是 llama.cpp CUDA 后端的惯用 API。

### 2.3 stage-in kernel：1074 行基础原语

`llama-kvmem-stagein.cu` 含 15 处 kernel launch，kernel 本体为：`fwht_kernel`（FWHT，纯加减）、`dequant_q8_0/dequant_q4_0`、`rope_neox_kernel`、`quant_q8_0/quant_q4_0`、gather/scatter（slab 打包与 D2D 批拷贝）、mean-K 累加 [6]。全部为标准 CUDA（无纹理、无 PTX 内联、无 warp-vote 之外的架构特定指令、无 coop-launch、无 CUDA Graph），到 HIP 的翻译是 hipcc 编译器的常规能力（llama.cpp 整个 ggml-cuda 就是这么上 AMD 的）。`half/__half/nv_bfloat16` 类型经 hip.h 的 typedef 覆盖 [45]。

### 2.4 ggml-cuda 内的补丁 kernel

累积补丁向 `ggml-cuda/gated_delta_net.cu` 追加的 `gdn_fold_f32/gdn_conv_fold_f32` 与向 `ggml-cuda.cu` 的接线，都在 `ggml-hip/CMakeLists.txt` 的源文件通配（`file(GLOB ../ggml-cuda/*.cu)`）覆盖范围内——HIP 构建自动重编译，**无需单独移植** [5][44]。

### 2.5 结论

需要人工编写的全部 AMD 适配代码 = 一个兼容头（把 §2.2 缺的 3-5 个符号补映射 + 转发 3 个头文件名）+ CMake 的 HIP 分支接线 + 少量 Windows 特有构建参数。这支持"最接近的等效"这一目标：不是近似模拟，而是让同一份 CUDA 源码在 AMD GPU 上原生执行。

---

## 3. AMD 平台现状与等效机制 [Confidence: High]

### 3.1 ROCm on Windows：从"不存在"到"官方支持 RDNA4"的时间线

2025-09-25，AMD 发布 ROCm 6.4.4 公开预览，首次让 PyTorch 原生跑在 Windows 的 Radeon RX 9000（RDNA4）与 RX 7000（RDNA3）上 [14][15]；ROCm 7.0 在 Windows 上弃用老 GCN 卡并为 HIP 增加 uncached 锁页分配（`hipExtHostRegisterUncached`、`hipHostMallocUncached`）[8]；7.1 明确启用 gfx1150/1151/1200/1201 LLVM 目标 [9]；社区时间线记录 ROCm 7.2（2026-03）为 RX 9070/9070 XT（gfx1201）首个官方支持版本 [22]。**当前稳定版 HIP SDK for Windows 为 7.2**，官方系统需求矩阵将 RX 9070 XT 列为 RDNA4/gfx1201、Runtime 与 HIP SDK 双 ✅、验证系统 Windows 11 22H2，并要求 CPU 支持 PCIe atomics（Zen 架构起满足）[7][13]。

（引用核验修正：子代理最初报告"稳定 HIP SDK 最新为 6.4.2"，验证阶段确认该页已更新，最新为 7.2，6.4.2 为旧条目 [13]。）

### 3.2 成熟度矛盾：官方矩阵 ✅ vs TheRock "Build Passing"

必须如实呈现的相反证据：二手转述 AMD TheRock 支持矩阵称 Windows 下 RDNA3/3.5 已达 Release-Ready 而 **RDNA4 仍停在 Build Passing**（Sanity-Tested/Release-Ready 两栏空白）[17]；多个社区来源建议在 Windows 上优先 llama.cpp + Vulkan 而非原生 ROCm [16][17][24]；Linux 侧则相反——ROCm 明显快于 Vulkan 的实测已出现 [23]，且 gfx1201 的 ROCm 构建在 Linux 已实际跑通 llama.cpp（尽管有 GPU 状态异常的报告）[34]。验证子代理无法在预算内直接复核 TheRock 矩阵页面的分档标签，故该矛盾评为 [Medium]。对本任务最有说服力的中和证据是 llama.cpp 自身：**官方 release CI 已经在为 Windows 平台的 ROCm 发布包编译 gfx1200/gfx1201 目标**（ROCm 7.14.0 wheel、hipBLASLt/rocBLAS 依赖）[42][43]——如果该组合完全不可用，这条发布线不会存在。综合判断：Windows + RDNA4 + HIP 处于"官方支持但仍在成熟中"状态，功能可用性与性能爬坡都还在验证期。

### 3.3 显存等效机制映射：CUDA 用什么，AMD 给什么

CUDA 侧 KVMem 用到的内存机制只有三种：锁页主存（`cudaMallocHost`/`cudaHostAlloc`）、异步显存拷贝（`cudaMemcpyAsync` + 独立流/事件）、设备内存（`cudaMalloc`）。HIP 侧逐项对应：`hipHostMalloc`/`hipHostAlloc`（ROCm 7.0 起有 uncached 变体 [8]）、`hipMemcpyAsync`、`hipMalloc`——全部在 Windows 上可用。ROCm 文档给出锁页内存约 3 倍于可分页内存的传输带宽优势 [11]，Resizable BAR/SAM 自 RX 6000 世代起允许 CPU 完整映射显存 BAR [18]，RDNA4 全线 PCIe 5.0 x16（RX 9070 XT 实测平台默认开启 ReBAR）[40][41]。

真正的平台缺口在**统一内存**：HIP 文档明确"UMM 在当前 Windows + AMD GPU 上不受支持" [10]（修正：其"hipMemPrefetchAsync 在 Windows 开发中"的表述未在所引页面找到，已从结论剔除），`hipMallocManaged` 的 HMM `malloc()` 路径仅限 MI300+/XNACK [10][12]。所幸 KVMem 的机制设计恰好绕开了这个缺口——它从不用 managed memory 做透明页迁移，而是显式的锁页缓冲 + 异步 DMA，这套语义在 WDDM/HIP-Windows 上完整可用。这构成"为什么等效可行"的最有力回答：**KVMem 的可移植性不是运气，是它把机制收敛到显式拷贝的结果**。

带宽账：PCIe 5.0 x16 理论 128 GB/s 双向（单向 ~64 GB/s），实际 DMA 常取 80-90%；RX 9070 XT 显存带宽 644.6 GB/s [40]。块换入换出是 128 token × n_layer × n_embd 量级的批量传输（q8_0 下单块每层约几十 KiB 到 MiB 级），以块为粒度聚合后传输效率远高于逐 token 流式方案，这正是有界窗口设计对 PCIe 友好的原因 [1][4]。DDR5 双通道主机内存 50-90 GB/s 是另一侧上限，本机 48 GB 容量对 256K 配方（RSS 峰值 13.5 GiB）留有余量 [1]。

### 3.4 被排除的替代路线

- **Vulkan 后端重实现**：adapter 全部 GPU 交互（锁页缓冲、流、事件、slab、15 个 kernel）都直接讲 CUDA API，没有走 ggml 的抽象缓冲层 [4]；迁到 Vulkan 意味着以 SPIR-V 重写 stage-in 流水线并在 ggml-vulkan 内部建立"GPU 地址可达的主机可见内存"原语——工作量月级且无现成先例，不符合"最接近的等效"。
- **WSL2 + Linux ROCm**：ROCm 计算不支持 WSL2 中的 Radeon 独显（WSL GPU 计算通道面向 NVIDIA/CUDA 与 WARP）；双系统超出本任务边界。kvmem 官方的 WSL2 说法（其测试平台即 Ubuntu/WSL2 [1]）只对 NVIDIA 成立。
- **SYCL/DirectML/oneAPI**：llama.cpp 构建文档中 SYCL 面向 Intel GPU；DirectML 后端已不在官方后端清单 [19][26]。
- **官方上游能力替代**（"-ngl 分层卸载 + KV 量化 + --kvo"）：这些解决的是"装得下"，不提供 KVMem 的"检索式 256K 虚拟工作区 + 跨轮回用"，语义不等效 [20][23]。

---

## 4. AMD 等效移植方案（设计已验证） [Confidence: High]

### 4.1 目标形态

在本机（RX 9070 XT gfx1201 + 48GB RAM + Windows）产出与 kvmem-llama.cpp CUDA 版**同一机制**的 HIP 版：`llama-kvmem-server` / `llama-kvmem-cli`，同样的 `--kvmem-*` 参数面、块槽位池、锁页主存分层、GPU stage-in 流水线与 MTP/ReplaySSM；差异仅在设备端 API（hip\*）与构建链（ROCm-for-Windows clang + hipBLASLt/rocBLAS）。

### 4.2 构建链事实（全部一手）

llama.cpp 官方 Windows HIP 构建配方（release CI 摘录）[42][43]：`python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[libraries,devel]==7.14.0"` → `rocm-sdk init` 展开 devel 树 → CMake `-DGGML_HIP=ON -DGPU_TARGETS=...gfx1201... -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_HIP_COMPILER=clang`（clang 三件套均取 ROCm 自带 LLVM）。已知运行期坑：Adrenalin 会在 System32 放置 `amdhip64_7.dll`，加载器搜索顺序导致版本错配，官方解法是把匹配版本的 `amdhip64_7.dll`/`rocm_kpack.dll`/`amd_comgr.dll` 拷到 exe 同目录 [43]。`HSA_OVERRIDE_GFX_VERSION` 在 Windows 不支持，因此 gfx1201 必须真实编译 [19]。ggml-hip 要求动态链接（禁 static）、`GGML_HIP_NO_VMM` 默认保守路径 [44][16]。

kvmem 移植版需要在此之上补的接线（源自其自身 patch 0004 的 CUDA 分支模式 [4]）：当 `GGML_HIP=ON` 时，把 `llama-kvmem-stagein.cu` 以 HIP 语言（或 ROCm clang++ + `--offload-arch=gfx1201`）编入 llama 目标，并把兼容头目录插入 llama 目标 include 路径最前。

### 4.3 兼容层设计

新建 `compat/hipify/` 三个头（仅 HIP 配置生效）：
- `cuda_runtime.h` → `#include <hip/hip_runtime.h>` + 复用/镜像 llama.cpp `vendors/hip.h` 的映射（已覆盖 36/40 符号）+ 补齐 `cudaHostAlloc→hipHostAlloc(...,hipHostAllocDefault)`、`cudaPointerAttributes→hipPointerAttribute_t`、`cudaPointerGetAttributes→hipPointerGetAttributes`、`cudaHostMalloc` 别名与错误码（`cudaErrorMemoryAllocation→hipErrorOutOfMemory` 已在 hip.h）[45]；
- `cuda_fp16.h` → `hip/hip_fp16.h`（`half`/`__half`/`__half2float` 均有 HIP 等价）；
- `cuda_bf16.h` → `hip/hip_bf16.h` + `nv_bfloat16` typedef（hip.h 已给）[45]。

stagein kernel 若遇 HIP 细节差异（如 `__shfl_down_sync`、shared-memory 动态配置），按 hip.h 既有映射先行；不足处做最小 `#ifdef __HIPCC__` 补丁。

### 4.4 验证阶梯

1. **主机层**：无 GPU 构建 `kvmem/` 库全套单测（store/runtime/pinned-tier/raw-kv-store 测试约 2000 行，与 CUDA 无关）——先证明移植没碰坏策略层 [4]。
2. **编译层**：整树 HIP 配置（`-DLLAMA_KVMEM=ON -DGGML_HIP=ON -DGPU_TARGETS=gfx1201`）过编译，产出含 stage-in kernel 的 `llama.dll`/exe。
3. **模型无关运行层**：复刻 fork 的模型无关检查（KV dtype 参数矩阵、server 选项回归）[1]。
4. **端到端冒烟**：小 GGUF（如 Qwen3-0.6B）验证 `-ngl all` 下 KVMem 分块换出/换入与生成正确性；再以 IQ3 27B 复测 256K 配方（约 12 GB 模型 + 3 GB 投影器下载，单独征得同意后执行）。
5. **正确性对照**：`--no-kvmem` 与健康输出比对 + 短上下文下 KVMem 开/关输出一致性（fork 文档明确警告 CUDA 编译器版本会引入 IQ3 乱码类正确性风险 [1]——HIP/clang 侧同样需要这道 canary）。

### 4.5 主要风险登记

| 风险 | 概率 | 缓解 |
|---|---|---|
| Windows RDNA4 HIP 内核 bug（GDN/FA 异常） | 中 | 降级 `-fa off`、`GGML_HIP_NO_VMM`、回退 CPU 层卸载；Linux 实测数据佐证 [23][34] |
| rocBLAS/hipBLASLt 在 gfx1201 kernel 缺失 | 低 | 官方 CI 同目标发布包佐证 [43]；amdgpu-prebuild 目录核验 |
| System32 `amdhip64_7.dll` 版本错配 | 高（常见坑） | 按官方 CI 把 runtime DLL 拷到 exe 目录 [43] |
| IQ3/IQ4 权重依赖量化 kernel（CUDA 模板） | 低 | ggml-cuda 量化 kernel 全量随 HIP 编译 [44] |
| WDDM 大块锁页失败（>13 GiB RSS 配方） | 中 | 降 budget/换 f16 KV 减体；监控分配失败日志 |
| 256K 配方主机 RAM 占用（RSS 13.5 GiB） | 低 | 48 GB 容量足够，留 OS 余量 |
| HIP 版性能低于 CUDA 参照 | 中 | 644 GB/s 带宽占优 [40]；hipBLASLt 路径；如 decode 低于预期走 Vulkan 对照 |

---

## 5. 批判性评估 [Confidence: Medium-High]

**这套机制值不值得移植？** 反对意见有实证分量：KV 卸载到系统内存被多方工程来源称为"最后手段"——DDR5 主机内存带宽（50-90 GB/s）与显存带宽（数百 GB/s）差一个数量级 [25][26]；白皮书级数据显示大比例 CPU 卸载时吞吐断崖（120→8 tok/s 级别的例证）[27][28]；mmap 静默换页会造成"每 token 数秒、磁盘 100%"的失败模式 [29]。但这些批评针对的是**权重/KV 全量流式搬运**；KVMem 的有界窗口设计恰恰是对这条批评的工程回应——decode 只对 32K-53K 可见 KV 做注意力，PCIe 流量以块为粒度摊销，MTP 投机进一步摊薄逐 token 成本，其结果是 256K 工作区仍有 32-33 tok/s [1][3]。这是"移植有意义"与"泛卸载无意义"两种论断能同时成立的原因。

**上游是否已覆盖？** llama.cpp 有 `-ngl` 分层、`--cpu-moe`、`-ctk/-ctv` KV 量化、`--kvo` prompt-KV 卸载、CUDA 统一内存（仅 Linux）等旋钮 [20][23][30]，但没有任何一个提供"块检索 + 有界工作集 + 跨轮 KV 复用 + 超出原生窗口的虚拟工作区"。语义上 KVMem 与这些是正交能力。

**证据的单边性风险**：KVMem 的性能与"近无损"数字全部出自项目自身（README + 论文，同一作者团队）[1][3]，尚无第三方复现；论文宣称 1M token 依赖 NVMe 层，而开源移植版未实现——引用时须区分"论文系统"与"移植版"两个对象 [1][3]。近无损性只在特定基准族（LongMemEval-S、AgentLongBench、DeepSWE 类长会话任务）上成立，检索失效（漏召回关键块）对任务型智能体的尾部风险没有独立评估。移植版应把 `--kvmem-trace`、needle recall 脚本等自带验证手段跑满再宣称等效 [5]。

**"要不要用 Vulkan 而非 HIP"的反问**：如果目标只是"本机跑 llama.cpp 快"，社区共识（Vulkan 优先）大概率正确 [16][17][24]；但本任务目标是 KVMem 的等效逻辑实现，其 adapter 的 CUDA 编程模型绑定使 Vulkan 不构成"等效"而是"重写"。HIP 路线还有一个长期红利：任何上游 ROCm 后端修复自动惠及该 fork，因为二者共享同一份 CUDA 源码 [44]。

**怀疑者的最强论点**：Windows + RDNA4 的 ROCm 生态仍在爬坡（FP8 软件栈未完备 [22]，TheRock 矩阵未 Release-Ready [17]），"最接近的等效"可能交付一个"能跑但慢/偶发异常"的系统。本报告将其登记为 §4.5 首要风险并给出降级路径，而非假装不存在。

---

## 6. Action Plan

- [x] 克隆 kvmem-llama.cpp + 钉版 llama.cpp submodule 到本地（`kvmem-llama.cpp/`）
- [x] 源码级 CUDA 耦合面清点（40 符号 + 1 kernel 文件 + 补丁触 ggml-cuda 清单）
- [x] 验证 ROCm Windows 构建配方与 gfx1201 支持（llama.cpp CI 一手证据）
- [x] 安装 ROCm for Windows SDK（pip wheel `rocm[libraries,devel]==7.14.0`，7.5 GB，项目 venv 内）
- [x] 编写 `compat/hipify/` 兼容头 + CMake HIP 分支接线（`patches/kvmem-hip-port-cmake.patch`）
- [x] 主机层 kvmem 单测通过（无 GPU 配置，MSVC 5/5）
- [x] 整树 `-DGGML_HIP=ON -DGPU_TARGETS=gfx1201` 编译通过（593/593，`llama-kvmem-server/cli`）
- [x] 模型无关回归（ctest 12/12 通过，mtp-kv-test 按设计 skip）
- [x] 端到端冒烟：基线推理 85-87 tok/s + KVMem needle 检索召回正确（`amber-jaguar-4417`，retrieval_ms=31.98，GDN fold kernel 在 gfx1201 执行）
- [ ] （可选）IQ3 27B `-mtp` + 256K 官方配方压测（需 ~12 GB 模型下载与精度 canary）
- [ ] 汇总性能对照表并回贴 kvmem Issue #44

### 6.1 实施结果（2026-09-22 当日闭环）

全部移植产物已在本机落地并通过验证，完整记录见 [docs/amd-hip-port.md](amd-hip-port.md)。两个实施期新发现（本报告方法论的实证补充）：

1. **ROCm-on-Windows × MSVC 14.51 工具链 bug**：clang HIP 头的 `isgreater` 等
   `__device__` 前向声明与 VS 18（MSVC 14.51.36231）`<cmath>` 冲突
   （llama.cpp issue #22570，"sometimes it works, sometimes not"——CI 的
   windows-2022 runner 因旧工具集未触发）。已用带守卫宏的幂等头文件补丁解决
   （`scripts/windows/patch-rocm-headers.py`）。
2. **System32 `amdhip64_7.dll` 劫持**（llama.cpp issue #26929）：Adrenalin 向
   System32 投放 HIP 运行时，加载器优先于 PATH 命中它；官方解法（拷 exe 同目录）
   已并入构建脚本的 staging 步骤；rocBLAS Tensile 库目录需随 PATH。

性能侧意外收获：KVMem fork 依赖的 `GGML_HIP_GRAPHS`（默认 ON）在 RDNA4 上生效
——9B 解码出现 CUDA Graph 复用（`graphs reused=47`），decode 85-87 tok/s，高于
其 CUDA 参照卡在弱于 9070 XT 的档位上的表现比例；`-ctk q8_0` 等量化 KV 与
ReplaySSM 全链路（检索 → 换入 → GDN fold）无功能降级。

## 7. Open Questions & Caveats

1. **Windows RDNA4 的真实成熟度**：官方 SKU 矩阵与支持矩阵存在口径差（§3.2），TheRock 矩阵分档未能在本次调研内直接复核——需以本机实测为最终裁判。
2. **hipBLAS 依赖形态**：llama.cpp b81c99b 的 ggml-hip 链接 `roc::hipblas`（legacy hipBLAS），而 Windows wheel 生态主推 hipBLASLt——若 wheel 缺 legacy hipBLAS，需要评估 GGML_HIPBLASLT 路径或补装组件。
3. **KVMem 数字的第三方复现缺失**；论文（GPU/主存/NVMe 三层）与移植版（两层）能力边界不同，256K 之外的长度是实验区。
4. **IQ3 对编译器版本的敏感性**在 CUDA 侧已造成过"成功构建但输出乱码"的事故 [1]；HIP/clang 侧的等价 canary（小上下文健康对比）是发布前置条件，不能以"构建成功"替代。
5. **单轮生成 ≤ gen_reserve 的 v1 限制**原样存在于移植版（与后端无关）；ring-buffer 修复在原作者路线图中 [1][4]。
6. **法律/许可**：仓库无 LICENSE 文件，README 声明"应按 Apache-2.0 对待"——本地学习/移植可行，再分发需留意。

## Methodology

深度模式（deep）。执行摘要：Phase 0 澄清（交付=报告+可运行实现；深度=deep；显卡=本机检测 → RX 9070 XT/gfx1201；假设=尽力最接近等效）；Phase 2 Wave 1 四个并行 Retrieval 子代理（领域 1-5，START_INDEX 1/20/40/60）+ Wave 2 一个 Gap-Fill 子代理（HIP SDK/gfx1201/kvmem Issue 区，START_INDEX 80）；Wave 3 以**本地源码取证**替代（克隆仓库、符号枚举、CI/ggml-hip/vendors-hip.h 阅读——升级为一手 Tier 1 证据）；Phase 3.1 引用核验子代理（7 条高影响声明：1 SUPPORTED×3、PARTIAL×2、UNSUPPORTED×1），据此修正：HIP SDK 最新版 6.4.2→**7.2**[13]；删除"hipMemPrefetchAsync Windows 开发中"无据从句 [10]；LongMemEval-S 数字归属改为 README/论文正文而非摘要 [1][3]；**推翻**子代理"官方 win-hip 预编译不含 gfx1201"的推断（其证据来自 Ollama 的 CMake 排除正则，不适用于 llama.cpp——本地 release CI 反证 [43]）。Phase 3.5 大纲调整 <10%（新增"§2 CUDA 耦合面解剖"，因源码证据升级）。Phase 4 红队：补"上游已覆盖"与"泛卸载批评"两个反方视角（§5）。**实现阶段追加发现**：子代理基于二手来源断言"Windows HIP 发布线不支持 RDNA4"被本地一手 CI 证据推翻（教训：跨仓库边界类比推理——Ollama≠llama.cpp——必须以一手构建系统为准）；实施中新登记两个工具链 bug（§6.1）并已解决。局限：kvmem 社区覆盖近乎为零（项目过新），部分中文二手来源无法升 Tier；TheRock 矩阵分档未直接复核。

## Bibliography

（编号说明：保留子代理原始编号区间，跨领域去重后部分编号空缺属正常；本地源码取证以 [4][5][6][42][43][44][45] 标注。）

[1] kvmem — KVMem + llama.cpp README — https://github.com/kvmem/kvmem-llama.cpp — 2026-09-22 — Tier 1
[2] kvmem — GitHub 组织页 — https://github.com/kvmem — 2026-09-22 — Tier 1
[3] Di Chai et al. — KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU — https://arxiv.org/abs/2609.04852 — 2026-09-22 — Tier 1
[4] 本地克隆 — kvmem-llama.cpp 源码（kvmem/、src/adapter/、patches/、CMakeLists.txt）— 本机 `kvmem-llama.cpp/`（@ v0.16.0-rc3 pin b81c99b）— 2026-09-22 — Tier 1
[5] 本地克隆 — docs/modification-plan.md + docs/architecture.md — 本机 `kvmem-llama.cpp/docs/` — 2026-09-22 — Tier 1
[6] 本地克隆 — src/adapter/llama-kvmem-stagein.{h,cu}（1074 行）— 2026-09-22 — Tier 1
[7] AMD — System requirements for Windows (HIP SDK 7.2.0 矩阵) — https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html — 2026-09-22 — Tier 1（核验 SUPPORTED）
[8] AMD — ROCm 7.0.0 release notes（uncached host alloc、GCN Windows 弃用）— https://rocm.docs.amd.com/en/docs-7.0.0/about/release-notes.html — Tier 1
[9] AMD — ROCm 7.1.0 release notes（gfx1200/1201 启用）— https://rocm.docs.amd.com/en/docs-7.1.0/about/release-notes.html — Tier 1
[10] AMD — HIP Unified memory how-to（UMM 不支持 Windows）— https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.2/how-to/unified_memory.html — Tier 1（核验 PARTIAL，已修正）
[11] AMD — HIP managed memory allocation reference — https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.0/reference/unified_memory_reference.html — Tier 1
[12] AMD — ROCm GPU memory concepts — https://rocm.docs.amd.com/en/docs-6.1.5/conceptual/gpu-memory.html — Tier 1
[13] AMD — HIP SDK for Windows 下载页（最新 7.2；"subset of ROCm"）— https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html — Tier 1（核验修正：6.4.2→7.2）
[14] TechPowerUp — AMD Enables PyTorch on Radeon RX 7000/9000 (ROCm 6.4.4 preview) — https://www.techpowerup.com/341329/ — 2025-09-25 — Tier 2（核验 SUPPORTED）
[15] wccftech — ROCm 6.4.4 PyTorch on Windows — https://wccftech.com/amd-rocm-6-4-4-pytorch-support-windows-radeon-9000-radeon-7000-gpus-ryzen-ai-apus/ — 2025-09-24 — Tier 3
[16] CSDN gitblog_00793 — AMD 显卡跑 llama.cpp 两条路线 + 诊断表 — https://m.blog.csdn.net/gitblog_00793/article/details/159939667 — 2026-08-30 — Tier 3
[17] 什么值得买 — ROCm 7.14 发布满月：4 个已知问题（TheRock 矩阵转述）— https://post.m.smzdm.com/p/axklg083/ — 2026-08-25 — Tier 3（核验 PARTIAL）
[18] cnblogs/ApacheCN — 现代 Vulkan 秘籍（host-visible 内存）+ SAM/ReBAR 报道 — https://www.cnblogs.com/apachecn/p/19167918 — Tier 3
[19] ggml-org — llama.cpp docs/build.md（HIP/Windows 命令、HSA_OVERRIDE 限制、UM 说明）— https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md — Tier 1（核验 SUPPORTED）
[20] MislavJuric + slaren/ggerganov — llama.cpp Discussion #9784（buffer/VRAM 记账模型）— https://github.com/ggml-org/llama.cpp/discussions/9784 — Tier 1
[21] ggml-org — ggml-backend-reg.cpp（后端注册与动态加载）— https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend-reg.cpp — Tier 1
[22] LocalAI Master — AMD ROCm Local LLM Setup 2026（ROCm 7.2 支持 RDNA4 时间线）— https://localaimaster.com/blog/amd-rocm-local-llm-setup — Tier 3
[23] 什么值得买 — ROCm10 暴涨3.3倍与你的卡；AMD 2026 ROCm 部署选型 — https://post.m.smzdm.com/p/ae6vkw9q/ ；https://post.m.smzdm.com/p/ae6rogdz/ — Tier 3
[24] Reddit r/LocalLLM — ZINC beats llama.cpp on RDNA4 sweep（含 harness 自批评）— https://www.reddit.com/r/LocalLLM/comments/1uka3xe/ — 2026-07-01 — Tier 3
[25] PromptQuorum — Q4_K_M vs Q4_0 vs Q8_0（DDR5 offload ≤90 GB/s、5-10 tok/s、last resort）— https://www.promptquorum.com/local-llms/llm-quantization-explained — Tier 2
[26] PromptQuorum — llama.cpp 详解 2026（后端谱系）— https://www.promptquorum.com/zh/power-local-llm/llama-cpp-explained — Tier 3
[27] 百度开发者社区 — KV Cache 超出显存的应对策略（PCIe 实测衰减数据）— http://developer.baidu.com/article/detail.html?id=6875221 — Tier 3
[28] Hugging Face Forums — RAM usage / model streaming 讨论（WDDM "bad cliff"、上游旋钮）— https://discuss.huggingface.co/t/ram-usage-model-streaming-or-alternatives/173719 — 2026-02-23 — Tier 3
[29] SSD Nodes — Ollama vs llama.cpp on a VPS（mmap 静默换页失败模式）— https://www.ssdnodes.com/learn/ollama-vs-llama-cpp — 2026-08-02 — Tier 2
[30] XDA-Developers — Lenovo ThinkStation PGX 200B 评测（CUDA 统一内存 + FP8 KV）— https://www.xda-developers.com/lenovo-thinkstation-pgx-review/ — 2026-02-16 — Tier 3
[31] Zhongzhuzhou — FreeToken 技术评述（传输串行化 19-26% 损失）— https://www.zhongzhuzhou.org/blog/2026-08-23-freetoken-technical-review-en/ — Tier 3
[32] CSDN liuxiang3 — llama.cpp 命令中文帮助（-kvo/--no-host/cache-type）— https://m.blog.csdn.net/liuxiang3/article/details/164109932 — Tier 3
[33] BiFangKNT — kvmem-llama.cpp Issue #44：AMD ROCm/HIP 适配询问（open、无回复）— https://github.com/kvmem/kvmem-llama.cpp/issues/44 — 2026-09-21 — Tier 1
[34] ROCm org — Issue #5706：HIP backend gfx1201 Linux 构建实况 — https://github.com/ROCm/ROCm/issues/5706 — 2025-11-27 — Tier 1
[35] CSDN u_14451778 — Ollama 源码编译（WINDOWS 排除正则；仅适用 Ollama）— https://m.blog.csdn.net/u014451778/article/details/157912843 — Tier 3（已标注不适用）
[36] AMD — ROCm Core SDK 7.11.0 release notes — https://rocm.docs.amd.com/en/7.11.0-preview/about/release-notes.html — Tier 1
[37] note.com gentle_murre488 — Windows RDNA4 llama-server 构建（ROCm 7.1.1 wheel 实录）— https://note.com/gentle_murre488/n/nf127994f3d2c — 2026-01-21 — Tier 3
[38] Zenn shuzan — Ryzen AI Max+ 395 本地 LLM（host-memory prompt caching PR #16391）— https://zenn.dev/shuzan/articles/72852bb9621b40 — Tier 2/3
[39] mydrivers — ZLUDA + HIP SDK 6.4 验证 RX 9060 XT（Windows SDK 子集性质）— https://m.mydrivers.com/newsview/1151163.html — Tier 2/3
[40] technical.city — RX 9070 XT 规格（PCIe 5.0 x16、644.6 GB/s、ReBAR+）— https://technical.city/zh/gpu/... — Tier 3
[41] TechSpot — Radeon RX 9060 XT 评测（RDNA4 PCIe 5.0 x16、ReBAR 默认开启）— https://www.techspot.com/review/2996-amd-radeon-9060-xt/ — 2025-06-04 — Tier 2
[42] 本地克隆 — llama.cpp@b81c99b .github/workflows/release.yml windows-rocm（ROCm 7.14.0、GPU targets 含 gfx1201、构建参数、DLL 打包注记）— 2026-09-22 — Tier 1
[43] 本地克隆 — llama.cpp@b81c99b .github/actions/windows-setup-rocm/action.yml（pip wheel 安装路径）— 2026-09-22 — Tier 1
[44] 本地克隆 — llama.cpp@b81c99b ggml/src/ggml-hip/CMakeLists.txt（glob 重编译 ggml-cuda 源码；WIN32 分支；hipblas/rocblas 依赖；禁 static）— 2026-09-22 — Tier 1
[45] 本地克隆 — llama.cpp@b81c99b ggml/src/ggml-cuda/vendors/hip.h（cuda→hip 完整符号映射清单）— 2026-09-22 — Tier 1

## Source Extracts

### [1] kvmem-llama.cpp README
- Summary: KVMem 移植版的完整机制、参数、实测与限制的一手来源；16 GiB 配方（budget 36864 + gen_reserve 16384 + q8_0 KV + MTP3）与 RTX 5060 Ti 基线数据；明示 NVMe 未实现、AMD/Metal 需集成工作。
- Key quotes: "Attention kernels and original positions stay unchanged. Reselection transfers only blocks that changed." / "NVMe offload is not implemented." / "AMD/ROCm and Metal backends would need integration work."
- Source type: docs/repo — Tier 1

### [4][5][6] 本地源码（kvmem/、docs/、src/adapter/、patches/）
- Summary: 策略库零 GPU 依赖；adapter 收敛全部 CUDA 面（40 符号）；stagein.cu 15 kernel 全基础原语；补丁触 ggml-cuda 的 GDN kernel 随 HIP glob 自动编译；三硬约束（kvmem 库零 llama 头 / 补丁仅三处 / 不改 FA）决定移植形状。
- Source type: code — Tier 1

### [42][43][44][45] 本地 llama.cpp@b81c99b CI 与构建系统
- Summary: Windows release 用 ROCm 7.14.0 wheel + ROCm clang 编译 ggml-hip，GPU targets 覆盖 gfx1200/gfx1201；ggml-hip 通过 file(GLOB ../ggml-cuda/*.cu) 重编译 CUDA 源码；vendors/hip.h 提供 cuda→hip 兼容映射；打包注记揭示 Adrenalin System32 amdhip64_7.dll 劫持坑。
- Key quotes: "gpu_targets: \"...gfx1200;gfx1201\"" / "file(GLOB GGML_SOURCES_ROCM \"../ggml-cuda/*.cu\")" / "The Adrenalin driver ships an amdhip64_7.dll in System32, which the loader searches before PATH".
- Source type: code/CI — Tier 1

### [7][10][13] AMD ROCm/HIP 官方文档
- Summary: RX 9070 XT=gfx1201 在 Windows HIP SDK 7.2.0 矩阵内；UMM 不支持 Windows（对本移植无碍——KVMem 只用锁页+显式拷贝）；HIP SDK 为 ROCm 子集。
- Source type: docs — Tier 1

### [17][22][24][25][27][28][29] 社区与批评性来源
- Summary: Windows RDNA4 ROCm 成熟度存疑（Build Passing 转述）；Vulkan 当下更快更稳的社区共识；泛 KV 卸载的带宽断崖数据；WDDM 共享显存"bad cliff"；mmap 静默劣化模式。
- Source type: community/bench — Tier 2-3
