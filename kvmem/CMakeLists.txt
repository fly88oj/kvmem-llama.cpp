option(KVMEM_ENABLE_NVME "Build the NVMe/SSD KV storage tier (POSIX and Win32)" ON)

add_library(kvmem
    src/host/kvmem_store.cpp
    src/host/kvmem_runtime.cpp
    src/host/raw_kv_store.cpp
    src/host/mean_key_retrieval.cpp
    src/host/memory_contract.cpp
    src/host/kv_content.cpp
    src/host/native_kv_payload.cpp
    src/host/host_kv_storage.cpp
    src/host/spill_io.cpp
    src/host/session_memory.cpp
    src/host/rope.cpp
)
if (KVMEM_ENABLE_NVME)
    # Platform I/O primitives for the NVMe tier; only compiled when the tier
    # is built. windows.h / the POSIX headers stay confined to this TU.
    target_sources(kvmem PRIVATE src/host/nvme_platform.cpp)
endif()
set_target_properties(kvmem PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kvmem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kvmem PUBLIC cxx_std_17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # ninfer's /d1trimfile prefix is its own source root and does not match this directory.
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/d1trimfile:${CMAKE_CURRENT_SOURCE_DIR}/>)
endif()
target_compile_definitions(kvmem PUBLIC KVMEM_ENABLE_NVME=$<BOOL:${KVMEM_ENABLE_NVME}>)

if (WIN32)
    # std::getenv is standard C++ and portable; MSVC marks it deprecated in
    # favour of the Windows-only _dupenv_s. Make the same choice llama.cpp makes
    # for its own sources rather than edit portable code.
    target_compile_definitions(kvmem PRIVATE _CRT_SECURE_NO_WARNINGS)
    # On ELF a shared library exports every symbol, which is what
    # `target_link_libraries(llama PRIVATE kvmem)` relies on. PE has no such
    # default, so without this libllama fails to link with undefined
    # kvmem::* symbols. Let CMake generate the export list.
    set_target_properties(kvmem PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

function(kvmem_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE kvmem)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

kvmem_add_test(backend_rebind_test)
kvmem_add_test(kvmem_store_test)
kvmem_add_test(pinned_kv_tier_test)
kvmem_add_test(memory_contract_test)
kvmem_add_test(native_kv_payload_test)
kvmem_add_test(host_kv_storage_test)
kvmem_add_test(spill_io_test)
kvmem_add_test(tiered_kv_storage_test)
kvmem_add_test(native_residency_test)
kvmem_add_test(kv_content_test)
kvmem_add_test(mean_key_retrieval_test)
# The runtime suite's only NVMe case is guarded by the KVMEM_ENABLE_NVME
# option, and the Win32 platform layer satisfies the tier contract, so it
# runs everywhere.
kvmem_add_test(kvmem_runtime_test)
# The Win32 platform layer satisfies the tier's positional-I/O contract, so
# the NVMe suites run on every platform that builds the tier.
if (KVMEM_ENABLE_NVME)
    kvmem_add_test(nvme_kv_tier_test)
else()
    kvmem_add_test(nvme_disabled_test)
endif()
kvmem_add_test(raw_kv_store_test)
