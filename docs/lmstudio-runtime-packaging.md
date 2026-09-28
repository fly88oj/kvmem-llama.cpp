# LM Studio Runtime 打包（KVMem AMD/HIP 引擎扩展包）

> 目标：把本仓库的 KVMem HIP 版 llama.cpp 做成 **LM Studio 直接可用**的
> runtime extension pack，在 LM Studio UI/CLI 里用 KVMem 的分层 KV 能力。
> 日期：2026-09-25，全程在 RX 9070 XT (gfx1201) + LM Studio 0.4.25 实测。

## 1. 机制（LM Studio 0.4.x runtime extension pack）

LM Studio 把推理 runtime 安装在
`%USERPROFILE%\.lmstudio\extensions\backends\<name>-<version>\`，由
`backend-manifest.json` 描述。0.4.x 的 llama.cpp 包分两层：

- **LM Studio 引擎层**（`llm_engine*.dll/.node`、`lmstudiocore.dll`）——官方
  预编译物，通过 `engine-protocol-server-artifacts.json` 列出的文件集以标准
  llama-server 协议对接下层；
- **llama 栈**（`llama.dll`、`llama-common.dll`、`llama-server.exe` +
  `llama-server-impl.dll`、`ggml*.dll`、`mtmd.dll`）——**这正是一个标准
  shared llama.cpp 构建的产物集**。

打包方法 = **克隆官方 ROCm 包目录 + 只覆盖 llama 栈**（社区 fork 通行做法，
与 Spark-X2.5 等先例一致）；ROCm 依赖 DLL 走官方 vendor 包
（`extensions/backends/vendor/win-llama-rocm-vendor-v6`），构建产物里
stage 的同名 DLL 一并带上保证自包含。

## 2. 构建流程（两步）

```powershell
# ① 共享库 HIP 构建（产物与 artifacts 清单对齐；统一 ggml.dll，
#    GGML_BACKEND_DL=OFF——adapter 直调 backend 入口无法跨插件 DLL 边界）
powershell scripts/windows/build-hip.ps1 -LmsShared -BuildDir build-hip-lms

# ② 打包并安装（克隆官方 2.46.0 → 覆盖 → manifest 版本自动 bump 到 2.46.1）
powershell scripts/windows/make-lms-extension.ps1 -Install

# ③ 选择引擎（或 LM Studio 里 Ctrl+Shift+R）
lms runtime select "llama.cpp-win-x86_64-amd-rocm-avx2@2.46.1"
```

配套的必要修改（都在仓库内）：
- `build-hip.ps1 -LmsShared`：shared+`GGML_BACKEND_DL=OFF`+server/app target，
  白名单构建目标（kvmem 自带测试 exe 依赖 llama 内部 C++ 类，shared 树链不动，
  与 pack 无关）；
- 根 `CMakeLists.txt`：`LLAMA_BUILD_SERVER/APP` 从 FORCE OFF 改为可 opt-in
  （pack 需要 stock `llama-server` target；kvmem 自身构建默认行为不变）；
- `kvmem/CMakeLists.txt`：kvmem 库显式 STATIC（shared 树中被吸收进 llama.dll）；
- **`KVMEM_*` 环境变量通道**（见 §3）。

## 3. KVMem 参数注入：`KVMEM_*` 环境变量

LM Studio 只会传它自己认识的 llama-server 参数，`--kvmem-*` 无法透传。为此
adapter 新增 env 回退通道：闸门 `llama_memory_kvmem_maybe_create` 改走
`llama_kvmem_get_params()` 访问器（它是宿主碰到的第一个 KVMem 入口，首次调用
即一次性应用 `KVMEM_*` 环境变量）。而 KVMem 感知的宿主（llama-kvmem-cli /
llama-kvmem-server）现在**无条件**在启动时调用 `llama_kvmem_set_params()`
（即使未加 `--kvmem`），置位 latch 标志后 ambient 环境变量永不可能改写 CLI
默认——机器上 `setx` 过 `KVMEM_*` 的用户，跑不带 `--kvmem` 的基线测试仍然严格
纯 stock。

| 变量 | 含义（对应 CLI 参数） |
|---|---|
| `KVMEM_ENABLE=1` | 开启 KVMem（`--kvmem`）——总开关，不设/0 即纯 stock |
| `KVMEM_BUDGET=4096` | GPU 工作集 tokens（`--kvmem-budget`） |
| `KVMEM_GEN_RESERVE=1024` | 生成预留（`--kvmem-gen-reserve`） |
| `KVMEM_BLOCK_TOKENS=128` | 块大小（`--kvmem-block-tokens`） |
| `KVMEM_METHOD=retrieval\|recency` | 检索策略（`--kvmem-method`） |
| `KVMEM_SINK_TOKENS` / `KVMEM_RECENT_TOKENS` | 同 CLI |
| `KVMEM_CPU_GB=4` / `KVMEM_NVME_GB=2` / `KVMEM_NVME_DIR` | 二级/三级缓存（`--kvmem-cpu-gb/--kvmem-nvme-gb/--kvmem-nvme-dir`） |
| `KVMEM_GPU_RATIO/HIGH/LOW`、`KVMEM_HARVEST_V`、`KVMEM_RAW_K_NVME`、`KVMEM_MTP_STATE`（0=snapshots/1=auto/2=replay，数字形式） | 同 CLI |

设置方式（用户级、对新启动的 LM Studio 生效）：

```powershell
setx KVMEM_ENABLE 1;  setx KVMEM_BUDGET 4096;  setx KVMEM_GEN_RESERVE 1024
setx KVMEM_BLOCK_TOKENS 128;  setx KVMEM_CPU_GB 4;  setx KVMEM_NVME_GB 2
# 然后**完全退出并重开** LM Studio（setx 不影响已运行进程的 env）
```

## 4. 前提与限制（实测确认）

- **模型必须单并行**：KVMem 要求 `n_seq_max=1`；LM Studio 的 Parallel 必须
  手动设为 1（GUI 加载设置 / `lms load --parallel 1`）。**注意新版 runtime
  （2.43+）默认 Parallel=4**——默认值下 KVMem 整个被旁路（日志
  `KVMem requires n_seq_max=1` 后回退 stock），256K 上下文时 f16 KV ≈16.8GB
  远超 VRAM，ggml host fallback 把 RAM 打满（27B IQ3_S + 24GB 空闲实测
  `freeRAM=0.3GB`——这就是"选 KVMem runtime 仍占满内存"的真因组合：
  Parallel=4 × f16 KV × MTP 默认 on）。
- **带 nextn 头的模型（如 Qwen3.8-27B UD）必须在 LM Studio 关闭 MTP 草稿**：
  app 默认加载 MTP draft context，其 KV 不受 KVMem 管理且与主 ctx 同 ctx 尺寸，
  256K 下直接 `failed to create MTP context: failed to allocate compute pp
  buffers` 加载失败（`lms load --no-speculative-draft-mtp` 或 GUI 关闭；
  KVMem 的 MTP 走自有 `--spec-type draft-mtp`，不走该通道）。
- **LM Studio 的 SWA 模型 VRAM 估算过保守**（把 gemma4 的 40 个滑窗层按全量
  KV 计入，131K 估 21GB；实际双缓存 ≈1.2GB），大 ctx 弹窗警告可忽略；
  KVMem 生效后实测 131072 ctx 仅 8.9GB VRAM。
- **关闭 runtime 自动更新**（settings → `autoUpdateExtensionPacks`/
  `autoDeleteExtensionPacks` = false），否则官方包自动升版时可能覆盖选择/
  清理旧包目录。
- **必须在 LM Studio 里打开 Flash Attention + KV 量化设 8-bit（q8_0）**：
  LM Studio 默认传 `--flash-attn off --cache-type-k/v f16`，该组合会触发
  rocBLAS 的 Tensile GEMM 路径，而 **gfx1201 缺预编译 Tensile 库**（上游已知
  gap；实测官方 ROCm 2.41.0 包在同样参数下同样 0xC0000409 启动即崩——非本
  pack 问题；官方 2.46.0 亦未随包提供 gfx1201 Tensile 库，该要求保持不变）。
  FA on + q8_0 同时避开该路径并匹配 KVMem 最优实测配方
  （pack 上验证：40K prompt 131K ctx 预填 2125 t/s、decode 55.6 t/s、完整
  needle 命中无崩溃）。
- **已知 app 侧限制**：app 的 engine-protocol 转发对超长 prompt 的 `/v1/chat/completions`
  可能提前断连；基础对话正常，超长文档建议直接跑 pack 的 llama-server.exe（参数级
  验证均健康）。gemma4 为 reasoning 模型：短 max_tokens 下答案在
  `reasoning_content` 字段属正常行为。
- KV 类型：除 GUI 设 8-bit 外，env 通道不控制 KV dtype（由 `--cache-type-*` 参数决定，
  LM Studio 可控）。
- MTP 加速不进 LM Studio 通道（app 有自己的 speculative 参数，仅当模型带
  nextn 且 app 支持时生效）。

## 5. 验收记录（gemma-4-12B-it-QAT / Qwen3.8-27B-IQ3_S，LM Studio 0.4.25 + pack）

| 检查 | 结果 |
|---|---|
| `lms runtime ls` 列出并可选 2.41.1（b81c99b 基线）/ 2.46.1（b11189 同步升级） | ✓ |
| app 日志 `LLM model loaded ... 2.46.1` | ✓ 27B IQ3_S `-c 262144 --no-speculative-draft-mtp` 100% 加载，server 来自 2.46.1 pack 目录 |
| server 进程 = pack 目录的 `llama-server.exe`（engine protocol 拉起） | ✓ |
| `KVMEM_*` env 生效（修复后真实直证） | ✓ 手工同参全量：`KVMEM_TIERS cpu_slots=3855 nvme_slots=1927 slot_bytes=1114112`（managed 布局）+ `%TEMP%\kvmem_nvme` 创建 |
| 40K prompt 端到端（llama-server 直连，FA on + q8_0） | ✓ prefill 2125 t/s、decode 55.6 t/s、正常释放无崩溃 |
| 16K/131K ctx 经 app 加载与短对话 | ✓ 回答正常（reasoning 字段） |
