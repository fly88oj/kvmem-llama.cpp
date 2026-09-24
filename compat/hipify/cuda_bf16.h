#pragma once
// KVMem HIP-port shim for <cuda_bf16.h> (see cuda_runtime.h for rationale).
#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_bf16.h>
// nv_bfloat16 / nv_bfloat162 typedefs live in ggml-cuda vendors/hip.h for
// kernel translation units that include common.cuh; the KVMem stage-in unit
// does not, so provide them here under include guards' own #ifndefs in the
// types' definitions (HIP struct names differ, so define only when absent is
// not detectable — rely on hip_bf16.h not shipping nv_* aliases; it does not).
typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;
// llama-kvmem-stagein.cu meank_add_bf16 names the struct __nv_bfloat16
typedef __hip_bfloat16 __nv_bfloat16;

#else
#include_next <cuda_bf16.h>
#endif
