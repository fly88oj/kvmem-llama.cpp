# 本机(Windows)构建脚本 - 双目标 fork(主干开发)

- `build.bat [server|lmstudio|all|test]` **统一入口**:两个产物一键可选
- `build-wrapper.bat`          目标 llama-kvmem-server(自动切子模块分支 llama/custom-b81c99b47)
- `runtests.bat` / `rebuild.bat` 定制线宿主测试/增量重编
- `build-lmstudio-runtime.bat` 目标 LM Studio 引擎包(自动切 llama/b11347,KVMem 全开+配置内化)
- `package-lmstudio-runtime.sh` runtime 打包(被 bat 调用;LMSTU_VER/SRC/OUT 可覆盖)

路径为本机绝对路径(E:\Works\KVMem、便携 CUDA 工具链),换机需改。
详见仓库外 E:\Works\KVMem\KVMEM-GUIDE.md(编译/NVMe/Runtime 统一指南)。
