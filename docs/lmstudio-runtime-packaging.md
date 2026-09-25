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

# ② 打包并安装（克隆 2.41.0 → 覆盖 → manifest 版本自动 bump 到 2.41.1）
powershell scripts/windows/make-lms-extension.ps1 -Install

# ③ 选择引擎（或 LM Studio 里 Ctrl+Shift+R）
lms runtime select "llama.cpp-win-x86_64-amd-rocm-avx2@2.41.1"
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
adapter 新增 env 回退通道：宿主**从未调用** `llama_kvmem_set_params()` 时
（stock llama-server 即如此），首次 `llama_kvmem_get_params()` 读取一次
`KVMEM_*` 环境变量；KVMem 感知的 CLI（llama-kvmem-cli/server）照常以显式参数为准，
**现有行为零影响**（已回归：HIP ctest 13/13 + 纯 llama-server 冒烟）。

| 变量 | 含义（对应 CLI 参数） |
|---|---|
| `KVMEM_ENABLE=1` | 开启 KVMem（`--kvmem`）——总开关，不设/0 即纯 stock |
| `KVMEM_BUDGET=4096` | GPU 工作集 tokens（`--kvmem-budget`） |
| `KVMEM_GEN_RESERVE=1024` | 生成预留（`--kvmem-gen-reserve`） |
| `KVMEM_BLOCK_TOKENS=128` | 块大小（`--kvmem-block-tokens`） |
| `KVMEM_METHOD=retrieval\|recency` | 检索策略（`--kvmem-method`） |
| `KVMEM_SINK_TOKENS` / `KVMEM_RECENT_TOKENS` | 同 CLI |
| `KVMEM_CPU_GB=4` / `KVMEM_NVME_GB=2` / `KVMEM_NVME_DIR` | 二级/三级缓存（`--kvmem-cpu-gb/--kvmem-nvme-gb/--kvmem-nvme-dir`） |
| `KVMEM_GPU_RATIO/HIGH/LOW`、`KVMEM_HARVEST_V`、`KVMEM_RAW_K_NVME`、`KVMEM_MTP_STATE` | 同 CLI |

设置方式（用户级、对新启动的 LM Studio 生效）：

```powershell
setx KVMEM_ENABLE 1;  setx KVMEM_BUDGET 4096;  setx KVMEM_GEN_RESERVE 1024
setx KVMEM_BLOCK_TOKENS 128;  setx KVMEM_CPU_GB 4;  setx KVMEM_NVME_GB 2
# 然后**完全退出并重开** LM Studio（setx 不影响已运行进程的 env）
```

## 4. 前提与限制（实测确认）

- **模型必须单并行**：KVMem 要求 `n_seq_max=1`；LM Studio 的 Parallel 保持 1
  （GUI 加载设置 / `lms load --parallel 1`）。parallel>1 时日志出现
  `KVMem requires n_seq_max=1` 后回退 stock KV（无害但不省内存）。
- **LM Studio 的 SWA 模型 VRAM 估算过保守**（把 gemma4 的 40 个滑窗层按全量
  KV 计入，131K 估 21GB；实际双缓存 ≈1.2GB），大 ctx 弹窗警告可忽略；
  KVMem 生效后实测 131072 ctx 仅 8.9GB VRAM。
- **关闭 runtime 自动更新**（settings → `autoUpdateExtensionPacks`/
  `autoDeleteExtensionPacks` = false），否则官方包自动升版时可能覆盖选择/
  清理旧包目录。
- **已知 app 侧限制**（与本 pack 无关）：app 的 engine-protocol 转发对超长
  prompt 的 `/v1/chat/completions`（1234 端口）会提前断连（"terminated"）；
  基础对话/40K prompt 直连或 GUI 长文本行为待用户日常使用反馈。
- KV 类型：LM Studio 默认 f16 KV（`--cache-type-k/v f16`）；想要 q8_0 的
  KV 减半效果，需在 LM Studio 模型加载设置里把 KV quant 设为 8bit
  （pack 的 KVMem 检索打分对 f16/q8_0 均按既有路径工作）。
- MTP 加速不进 LM Studio 通道（app 有自己的 speculative 参数，仅当模型带
  nextn 且 app 支持时生效）。

## 5. 验收记录（gemma-4-12B-it-QAT，LM Studio 0.4.25 + 2.41.1 pack）

| 检查 | 结果 |
|---|---|
| `lms runtime ls` 列出并可选 2.41.1 | ✓ |
| app 日志 `LLM model loaded ... 2.41.1, contextLength 131072` | ✓ |
| server 进程 = pack 目录的 `llama-server.exe`（engine protocol 拉起） | ✓ |
| `KVMEM_*` env 生效 | ✓ `%TEMP%\kvmem_nvme` 目录创建（NVMe tier 初始化唯一路径） |
| 131072 ctx VRAM | **8,868 MB**（stock 估算 21.4GB/16GB 卡不可行；KVMem 池省有效） |
| 16K ctx 对话（REST 1234） | ✓ 回答正常（gemma4 reasoning 输出正确字符串） |
| 纯 llama-server.exe + env 冒烟（pack 外独立） | ✓ `KVMem configured from KVMEM_* environment` 日志 |
