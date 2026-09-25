# KVMem AMD/HIP 完整性能对比矩阵

> 硬件：Radeon RX 9070 XT 16GB（gfx1201/RDNA4）+ 47GB RAM + NVMe；
> 软件：`build-hip2`（ROCm 7.14 wheel 工具链，FA 开启，`-ctk/-ctv q8_0`，
> 分支 `feature/kvmem-swa-support`，含 NVMe Win32 移植 + SWA 支持 + capture 接线）。
> 日期：2026-09-25。除标注外均为 `llama-kvmem-cli`/`server` 冷启动、watchdog 保护下实测。

## 1. 模型 × 路径总览

| 模型 | 架构路径 | 上下文 | prefill tok/s | decode tok/s | 峰值 VRAM | 峰值 RAM | needle |
|---|---|---|---|---|---|---|---|
| Qwopus3.5-9B Q4_K_S（plain+MTP3） | KVMem 检索 | 40.7K@128K | **3077** | **111.9**（n=256 中位） | 7.9 GB | ~37 GB 计 | ✅ HIT |
| 同上 | LM Studio（Vulkan, f16 全量 KV, MTP3） | 40.7K@128K | 2387 | 102-104 | **14.6 GB** | n/a | ✅ |
| Qwen3.8-27B IQ3_S（GDN hybrid，MTP 有害→关闭） | KVMem + NVMe 16GB | 40.8K@256K | 264.5 | 25.5 | 15.4 GB | 34.8 GB | ✅（深填另测） |
| 同上 | KVMem + NVMe | **244K token 深填充@256K** | 206（干净背景） | ~25 | 15.4 GB | min-free 5.2GB | ✅ 200K 深处召回，**10.9GB 落盘** |
| Bonsai-27B Q1_0（GDN） | KVMem | 32K | 冒烟通过 | — | 安全 | — | （EOS 特性，未做问答） |
| **gemma-4-12B-it-QAT Q4_0（interleaved SWA，新）** | **stock（无 KVMem）** | 10K@36K | 654 | ~60 | **15.7 GB（36K 即近爆）** | 26.4 GB | — |
| 同上 | **KVMem budget 4096** | 10K@36K | ~650 | ~60 | **9.1 GB（−43%）** | 26.2 GB | — |
| 同上 | KVMem budget 4096 | 40.7K@128K | **1527** | **55.6** | **11.8 GB** | 30.9 GB | ✅（server） |
| 同上 | KVMem budget 4096 | 40.7K@**256K** | **1539** | 同上 | **10.7 GB**（budget 封顶，几乎不随 ctx 增长；与 128K 同量级） | 30.8 GB | — |
| 同上 | **server chat 端到端**（39.5K prompt + needle 问答） | 128K | **26.2 s 全请求** | — | — | — | **✅ HIT** |
| 同上 | stock server 同请求 | 128K | **162.6 s 全请求（6.2× 慢）** | — | 爆险 | — | ✅ |

**gemma 关键结论**：stock 在 36K 上下文就吃掉 15.7GB VRAM（128K 物理不可行——此前实测 139K 直接溢出被看门狗终止）；
KVMem SWA 路径把 128K/256K 都压到 ~11GB，且端到端长上下文请求快 **6.2 倍**（避免 KV 挤爆 VRAM 的 offload 往返）。

## 2. gemma budget 敏感性（128K ctx，40.7K prompt，q8_0 KV）

| budget+gen_reserve | prefill t/s | decode t/s | VRAM |
|---|---|---|---|
| **4096+1024（最优）** | **1527** | **55.6** | 11.8 GB |
| 16384+4096 | 687 | 40.2 | 14.5 GB |
| 32768+8192 | 252 | — | 15.7 GB |

与 Qwen 两轮 sweep 的结论一致：**budget 4096 最优**。gemma 的 global 层 KV 极小
（8 层 × 1 头 × 512 dim ≈ 1.1MB/128-token 块），检索往返便宜，GPU 窗口小反而让
prefill/decode 的注意力带宽占用更低。

## 3. 正确性与回归证据（本轮 SWA/capture 改动后）

| 验证 | 结果 |
|---|---|
| gemma4 一致性金标准（temp 0，`--tokens-only`，KVMem vs stock 逐 token） | **24/24 相同**（reuse 接线后复验不变） |
| gemma4 128K needle（server chat 模板，budget 4096 触发逐出+检索） | **HIT**（capture 接线后；接线前 MISS——根因 gemma4.cpp 无 capture hooks，mean-K 从未被采集，检索失明） |
| HIP ctest | **13/13** |
| host ctest（MSVC，NVMe ON） | **5/5**（kvmem lib 本轮零改动，重建复验） |
| Qwopus-9B plain+MTP 回归（128K，needle） | prefill **3077**（基线 3047-3071）+ **HIT** |
| Qwen3.8-27B GDN+NVMe 回归（40K@256K） | prefill **264.5** / decode **25.53**（基线区间 257-270 / 25.2-25.9） |
| patch 完整性 | 重新生成的 `llama-kvmem-current.patch` 对工作树 **`apply --reverse --check` 字节级往返一致** |
| 小预算下限保护 | budget 512 → base cells 自动抬到 1152（= n_swa+n_ubatch），正常完成 |

## 4. 推荐配方（按模型类型）

| 模型类型 | 配方 |
|---|---|
| Qwen3.5/3.6 系（含 MTP） | `-ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3 --kvmem-budget 4096 --kvmem-block-tokens 128`（≥30K 上下文；<10K 用默认） |
| Qwen3.8-27B IQ3_S（GDN，无 MTP） | 上述去掉 spec + `--kvmem-cpu-gb 2 --kvmem-nvme-gb 16`（深填充场景）；**禁用 MTP**（decode 14 vs 26 有害） |
| gemma3/gemma4（SWA，含 QAT） | `--kvmem-budget 4096 --kvmem-gen-reserve 1024 --kvmem-block-tokens 128 --kvmem-cpu-gb 4 -ctk q8_0 -ctv q8_0 -fa on`；VRAM 受 budget 封顶，256K 与 128K 同量级（~10.7-11.8 GB，差异来自 compute buffer 随 ctx/生成长度波动） |
| KV-sharing 变体（gemma E2B/E4B） | reuse 回调已接线，**但缺实机验证**——拿到模型后先跑一致性金标准再采信 |
| 所有模型 | gemma 类必须经 server/chat 模板使用（CLI 裸文本 instruct 模型会早 EOS，非移植缺陷） |

## 5. 已知边界

- QAT/int4/int3 等**权重量化与 KVMem/NVMe/SWA 分层完全正交**（工作在 KV 块层面），无需任何适配。
- gemma4 的 SWA 层不进 KVMem 检索（物理上远端已滑出窗口）；长程记忆全部由 8 个 global 层承担——检索打分只捕获 global 层 mean-K。
- MTP/`--spec-type` 与 SWA 模型组合暂无意义（gemma 无 MTP 头）。
- 256K 深填充（>150K 有效 KV 落 budget）在 47GB RAM 机器上贴容量边缘：`--kvmem-nvme-gb` 是必要的第二层。
