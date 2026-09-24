#pragma once
// Symbol map for the KVMem CUDA-API adapter under HIP.
//
// Scope: exactly the cuda* surface used by src/adapter/*.cpp, *.h and
// llama-kvmem-stagein.cu (enumerated from the v0.16.0-rc3 tree). Most of it
// already exists in ggml/src/ggml-cuda/vendors/hip.h, but that header is
// only pulled in by ggml-cuda kernel units through common.cuh; plain host
// units (the adapter .cpp files, compiled by the same ROCm clang++ on
// Windows HIP builds) and the stage-in unit do not include it. This file
// repeats the needed entries, each behind an "#ifndef <cuda-name>" guard.
//
// Guard semantics (important): the cuda* names below are typedefs/enums in
// CUDA, NOT preprocessor macros, so "#ifndef cudaX" is only ever false when
// ggml-cuda's vendors/hip.h was included FIRST in the same translation unit
// (that header #defines the same cuda* names as macros). In that order the
// guard skips our #define and avoids a macro-redefinition warning; in the
// opposite order our replacement list is identical to vendors/hip.h, so any
// later redefinition is benign. "#pragma once" keeps this file idempotent.
//
// Windows note: everything mapped here is explicit pinned-host + async-DMA
// API (hipHostMalloc/hipMemcpyAsync/hipEvent*), which is supported on
// Windows HIP for AMD GPUs. Unified/managed memory is NOT used by KVMem, so
// the "UMM unsupported on Windows" limitation does not apply.

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// ---- error handling ----
#ifndef cudaError_t
#define cudaError_t hipError_t
#endif
#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif
#ifndef cudaErrorMemoryAllocation
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#endif
#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif
#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#endif

// ---- device management ----
#ifndef cudaSetDevice
#define cudaSetDevice hipSetDevice
#endif
#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#endif
#ifndef cudaDeviceSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#endif

// ---- memory allocation ----
#ifndef cudaMalloc
#define cudaMalloc hipMalloc
#endif
#ifndef cudaFree
#define cudaFree hipFree
#endif
#ifndef cudaMallocHost
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#endif
#ifndef cudaHostAlloc
// hipHostMalloc is the canonical API; hipHostAlloc only exists as an alias in
// some ROCm versions. Flags are numerically identical (all zero for default).
#define cudaHostAlloc(ptr, size, flags) hipHostMalloc(ptr, size, flags)
#endif
#ifndef cudaHostAllocDefault
#define cudaHostAllocDefault hipHostMallocDefault
#endif
#ifndef cudaHostAllocMapped
#define cudaHostAllocMapped hipHostMallocMapped
#endif
#ifndef cudaHostAllocPortable
#define cudaHostAllocPortable hipHostMallocPortable
#endif
#ifndef cudaHostAllocWriteCombined
#define cudaHostAllocWriteCombined hipHostMallocWriteCombined
#endif
#ifndef cudaFreeHost
#define cudaFreeHost hipHostFree
#endif

// ---- copies ----
#ifndef cudaMemcpy
#define cudaMemcpy hipMemcpy
#endif
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync hipMemcpyAsync
#endif
#ifndef cudaMemcpyKind
#define cudaMemcpyKind hipMemcpyKind
#endif
#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#endif
#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif
#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif
#ifndef cudaMemsetAsync
#define cudaMemsetAsync hipMemsetAsync
#endif

// ---- streams ----
#ifndef cudaStream_t
#define cudaStream_t hipStream_t
#endif
#ifndef cudaStreamCreateWithFlags
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#endif
#ifndef cudaStreamNonBlocking
#define cudaStreamNonBlocking hipStreamNonBlocking
#endif
#ifndef cudaStreamDestroy
#define cudaStreamDestroy hipStreamDestroy
#endif
#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif
#ifndef cudaStreamWaitEvent
#define cudaStreamWaitEvent hipStreamWaitEvent
#endif
#ifndef cudaStreamPerThread
#define cudaStreamPerThread hipStreamPerThread
#endif

// ---- events ----
#ifndef cudaEvent_t
#define cudaEvent_t hipEvent_t
#endif
#ifndef cudaEventCreateWithFlags
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#endif
#ifndef cudaEventDisableTiming
#define cudaEventDisableTiming hipEventDisableTiming
#endif
#ifndef cudaEventRecord
#define cudaEventRecord hipEventRecord
#endif
#ifndef cudaEventSynchronize
#define cudaEventSynchronize hipEventSynchronize
#endif
#ifndef cudaEventDestroy
#define cudaEventDestroy hipEventDestroy
#endif

// ---- pointer attributes (llama-memory-kvmem.cpp GDN replay device probe) ----
#ifndef cudaPointerAttributes
#define cudaPointerAttributes hipPointerAttribute_t
#endif
#ifndef cudaPointerGetAttributes
#define cudaPointerGetAttributes hipPointerGetAttributes
#endif

#endif // __HIP_PLATFORM_AMD__
