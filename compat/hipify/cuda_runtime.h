#pragma once
// KVMem HIP-port shim (AMD/ROCm equivalent of the CUDA runtime header).
//
// Purpose: the KVMem adapter (src/adapter/) calls the CUDA Runtime API
// directly (cudaMemcpyAsync, cudaMallocHost, events, streams). AMD's ROCm
// backend in llama.cpp solves the same problem for ggml-cuda with
// ggml/src/ggml-cuda/vendors/hip.h — a plain `#define cudaXxx hipXxx` map.
// This shim forwards <cuda_runtime.h> to the HIP runtime for translation
// units compiled by hipcc / ROCm clang++ in a GGML_HIP build, and adds the
// few symbols vendors/hip.h does not cover. When a real CUDA toolchain is
// used (no __HIP_PLATFORM_AMD__), it defers to the toolkit header unchanged.
//
// Placed on the include path ONLY for the HIP configuration (see the
// GGML_HIP branch in llama.cpp/src/CMakeLists.txt), so CUDA builds are
// bit-identical.

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

#include "kvmem_hip_defs.h"

#else
// Real NVIDIA CUDA: use the toolkit header. include_next requires the CUDA
// include dir to appear after this shim directory on the search path.
#include_next <cuda_runtime.h>
#endif
