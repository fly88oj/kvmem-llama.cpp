#pragma once
// KVMem HIP-port shim for <cuda_fp16.h> (see cuda_runtime.h for rationale).
#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_fp16.h>

// Legacy `half` typedef: CUDA's cuda_fp16.h exposes it; mirror it when the
// HIP header of the installed ROCm version does not. Re-typedef to the same
// type is legal in C++, so this line is safe either way unless HIP declares
// `half` as a distinct entity (build log would then flag it; remove to fix).
#if !defined(KVMEM_HIP_NO_HALF_TYPEDEF)
typedef __half half;
typedef __half2 half2;
#endif

#else
#include_next <cuda_fp16.h>
#endif
