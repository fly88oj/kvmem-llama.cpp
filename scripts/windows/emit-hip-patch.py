"""Emit patches/kvmem-hip-port-cmake.patch: the delta our AMD/HIP port adds on
top of the cumulative KVMem patch (llama.cpp/src/CMakeLists.txt only)."""
import difflib
from pathlib import Path

target = Path("llama.cpp/src/CMakeLists.txt")
cur = target.read_text(encoding="utf-8").splitlines(keepends=True)

GUARD_BLOCK = """    if (GGML_CUDA AND GGML_HIP)
        message(FATAL_ERROR "LLAMA_KVMEM: GGML_CUDA and GGML_HIP are mutually exclusive")
    endif()
"""

HIP_BLOCK = """    # KVMem AMD/HIP port: same stage-in kernels compiled for GCN/RDNA by the
    # ROCm toolchain. Mirrors how ggml/src/ggml-hip rebuilds ../ggml-cuda/*.cu
    # (LANGUAGE CXX + hip::device on Windows where CMake has no HIP language;
    # native LANGUAGE HIP elsewhere). compat/hipify maps the CUDA API names
    # used by the adapter onto HIP (see header for scope).
    if (GGML_HIP)
        find_package(hip REQUIRED)
        set(KVMEM_STAGEIN_CU ${LLAMA_KVMEM_ROOT}/src/adapter/llama-kvmem-stagein.cu)
        target_sources(llama PRIVATE ${KVMEM_STAGEIN_CU})
        if (WIN32)
            set_source_files_properties(${KVMEM_STAGEIN_CU} PROPERTIES LANGUAGE CXX)
        else()
            enable_language(HIP)
            set_source_files_properties(${KVMEM_STAGEIN_CU} PROPERTIES LANGUAGE HIP)
        endif()
        target_include_directories(llama PRIVATE ${LLAMA_KVMEM_ROOT}/compat/hipify)
        target_compile_definitions(llama PRIVATE GGML_USE_HIP)
        target_link_libraries(llama PRIVATE hip::device)
    endif()
"""

base = "".join(cur)
assert HIP_BLOCK in base and GUARD_BLOCK in base
base = base.replace(GUARD_BLOCK, "")
base = base.replace(HIP_BLOCK, "")
base = base.replace("if (CUDAToolkit_FOUND AND NOT GGML_HIP)", "if (CUDAToolkit_FOUND)")

diff = difflib.unified_diff(
    base.splitlines(keepends=True), cur,
    fromfile="a/src/CMakeLists.txt", tofile="b/src/CMakeLists.txt")
body = [line for line in diff if not line.startswith(("---", "+++"))]
body = [l if l.endswith("\n") else l + "\n" for l in body]
out = "".join(
    ["diff --git a/src/CMakeLists.txt b/src/CMakeLists.txt\n",
     "--- a/src/CMakeLists.txt\n", "+++ b/src/CMakeLists.txt\n"] + body)
Path("patches/kvmem-hip-port-cmake.patch").write_text(out, encoding="utf-8", newline="\n")
print("wrote patches/kvmem-hip-port-cmake.patch,", len(out.splitlines()), "lines")
