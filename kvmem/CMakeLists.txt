option(KVMEM_ENABLE_NVME "Build the NVMe/SSD KV storage tier (POSIX and Win32)" ON)

set(KVMEM_SOURCES
    src/host/kvmem_store.cpp
    src/host/kvmem_runtime.cpp
    src/host/raw_kv_store.cpp
    src/host/rope.cpp
)
if (KVMEM_ENABLE_NVME)
    # Platform I/O primitives for the NVMe tier; only compiled when the tier
    # is enabled (an all-guarded TU would trigger MSVC LNK4221).
    list(APPEND KVMEM_SOURCES src/host/nvme_platform.cpp)
endif()
# Windows builds are always static here (the root defaults BUILD_SHARED_LIBS
# OFF; the LM Studio shared tree needs kvmem absorbed into llama.dll and an
# unexported Windows SHARED lib would strand its symbols). Keep the
# BUILD_SHARED_LIBS-driven type elsewhere so the Linux CUDA prebuilt packaging
# keeps shipping libkvmem.so alongside libllama.so.
if (WIN32)
    add_library(kvmem STATIC ${KVMEM_SOURCES})
else()
    add_library(kvmem ${KVMEM_SOURCES})
endif()
set_target_properties(kvmem PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kvmem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kvmem PUBLIC cxx_std_17)
target_compile_definitions(kvmem PUBLIC KVMEM_ENABLE_NVME=$<BOOL:${KVMEM_ENABLE_NVME}>)

function(kvmem_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE kvmem)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

kvmem_add_test(kvmem_store_test)
kvmem_add_test(pinned_kv_tier_test)
if (KVMEM_ENABLE_NVME)
    kvmem_add_test(nvme_kv_tier_test)
else()
    kvmem_add_test(nvme_disabled_test)
endif()
kvmem_add_test(kvmem_runtime_test)
kvmem_add_test(raw_kv_store_test)
