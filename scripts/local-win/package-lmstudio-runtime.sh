#!/bin/bash
# package-lmstudio-runtime.sh — assemble the KVMem llama-server into an LM Studio
# engine runtime folder (backend-manifest v4). Does NOT touch installed runtimes.
set -e
SRC="${LMSTU_SRC:-E:\\Works\\KVMem\\kvmem-src\\build-lmstudio\\bin}"
OUT="${LMSTU_OUT:-E:\\Works\\KVMem\\lmstudio-runtime\\llama.cpp-kvmem-win-x86_64-nvidia-cuda13-avx2}"
mkdir -p "$OUT"

# engine binaries + shared libs from the build
cp -u "$SRC"/*.exe "$SRC"/*.dll "$OUT/" 2>/dev/null || true

# CUDA 13.2 runtime DLLs (self-contained, CUDA 13 differs from LM Studio's 12.x)
CE="E:\\Works\\KVMem\\cuda-extract"
cp -u "$CE/cuda_cudart/cudart/bin/x64/cudart64_13.dll" "$OUT/"
cp -u "$CE/libcublas/cublas/bin/x64/cublas64_13.dll" "$OUT/"
cp -u "$CE/libcublas/cublas/bin/x64/cublasLt64_13.dll" "$OUT/"

# MSVC runtime (match stock runtime's self-contained layout)
cp -u C:/Windows/System32/MSVCP140.dll C:/Windows/System32/VCRUNTIME140.dll \
      C:/Windows/System32/VCRUNTIME140_1.dll C:/Windows/System32/VCOMP140.dll "$OUT/" 2>/dev/null || true

cat > "$OUT/backend-manifest.json" <<EOF
{
  "name": "llama.cpp-kvmem-win-x86_64-nvidia-cuda13-avx2",
  "version": "${LMSTU_VER:-0.16.0-rc3}",
  "domains": ["llm", "embedding"],
  "engine": "llama.cpp",
  "extension_type": "engine",
  "target_libraries": [],
  "platform": "win",
  "cpu": { "architecture": "x86_64", "instruction_set_extensions": ["AVX2"] },
  "gpu": { "make": "Nvidia", "framework": "CUDA", "targets": ["8.9", "12.0"], "minimum_driver_version": "58000" },
  "supported_model_formats": ["gguf"],
  "manifest_version": "4",
  "minimum_lmstudio_version": "0.4.0+15",
  "engine_protocol_server": {
    "runtime_kind": "llama-server",
    "executable_relative_path": "llama-server.exe"
  }
}
EOF

cat > "$OUT/display-data.json" <<EOF
[["en",{"langKey":"en","displayName":"KVMem CUDA 13 llama.cpp (Windows)","description":"kvmem-llama.cpp v0.16.0-rc3 engine with KVMem bounded-GPU-window memory (KVMem on by default in this build)","releaseNotes":[{"version":"0.16.0-rc3","releaseNotes":"- kvmem-llama.cpp ${LMSTU_VER:-0.16.0-rc3}\n- stock llama-server with KVMem factory hook + env bridge\n"}]}]]
EOF

# engine-protocol artifacts list: every exe/dll we ship
python - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
files = sorted(f for f in os.listdir(out) if f.lower().endswith((".exe", ".dll")))
doc = {"schema_version": 1, "runtime_kind": "llama-server",
       "executable_relative_path": "llama-server.exe",
       "files": [{"relative_path": f, "executable": f.lower().endswith(".exe")} for f in files]}
open(os.path.join(out, "engine-protocol-server-artifacts.json"), "w").write(json.dumps(doc, indent=2))
print("artifacts:", len(files))
EOF
ls "$OUT" | head -30
echo "PACKAGED -> $OUT"
