"""KVMem HIP port: idempotent patch for the ROCm-on-Windows toolchain bug.

Root cause (llama.cpp issue #22570, "sometimes it works, sometimes not"):
clang's HIP compatibility headers forward-declare isgreater/isless/... as
__device__-only, which collides with MSVC 14.5x <cmath>'s _CLANG_BUILTIN2
declarations that the CUDA/HIP implicit attributes feature turns into
__host__ __device__ overloads. The affected math functions are not used by
ggml-cuda / KVMem kernels, so skipping the __device__ side is safe under
MSVC. Guarded by KVMEM_HIP_SKIP_MATH_FWD to keep the patch inert unless the
build passes -DKVMEM_HIP_SKIP_MATH_FWD.

Usage: python patch_rocm_headers.py <path-to-_rocm_sdk_devel>

Newlines: universal-newline read + platform write is a byte-exact round trip
for the CRLF headers shipped by the ROCm-on-Windows wheel (verified
289->301->289 and 850->856->850 CRLF). LF-ending headers on Windows would be
normalized to CRLF; this script targets the Windows wheel only.
"""
import re
import sys
from pathlib import Path


def _read(p):
    # Read with universal newlines: the ROCm-on-Windows wheel ships the clang
    # headers with CRLF endings, so this normalizes \r\n -> \n and the \n-based
    # guard regexes below match. (Using newline="" would keep \r\n and every
    # regex would silently match nothing -> "fresh tree" false negative.)
    return Path(p).read_text(encoding="utf-8")


def _write(p, s):
    # Platform write: on Windows \n -> \r\n, preserving the wheel's CRLF endings
    # so a patch->revert round trip stays byte-identical to the pristine header.
    Path(p).write_text(s, encoding="utf-8")


devel = Path(sys.argv[1])
inc = next(iter(devel.glob("lib/llvm/lib/clang/*/include")), None)
if inc is None:
    sys.exit("clang include dir not found under " + str(devel))

GUARD = "KVMEM_HIP_SKIP_MATH_FWD"
FUNCS = ["isgreater", "isgreaterequal", "isless", "islessequal",
         "islessgreater", "isunordered"]

changed = []

# --- 1. __clang_cuda_math_forward_declares.h : single-line decls ---
f1 = inc / "__clang_cuda_math_forward_declares.h"
t1 = _read(f1)
if "// " + GUARD in t1:
    print("forward_declares already patched")
else:
    def wrap(m):
        return ("// " + GUARD + " begin\n#ifndef " + GUARD + "\n"
                + m.group(0) + "#endif\n// " + GUARD + " end\n")
    pat = "|".join(FUNCS)
    t1_new = re.sub(
        r"(?:^__DEVICE__ bool (?:%s)\(.*\;\n){1,}" % pat, wrap, t1, flags=re.M)
    if t1_new != t1:
        _write(f1, t1_new)
        changed.append(f1.name)

# --- 2. __clang_hip_cmath.h : multi-line definitions ---
f2 = inc / "__clang_hip_cmath.h"
t2 = _read(f2)
if "/* " + GUARD in t2:
    print("hip_cmath already patched")
else:
    def block_repl(m):
        return ("/* " + GUARD + " */\n#ifndef " + GUARD + "\n"
                + m.group(0) + "#endif\n")
    pat = (r"(?:^__DEVICE__ __CONSTEXPR__ bool (?:%s)\([^;]*?\)\s*\{\s*"
           r"return __builtin_[a-z]+\(.*?;\s*\}\n){1,}" % "|".join(FUNCS))
    t2_new = re.sub(pat, block_repl, t2, flags=re.M)
    if t2_new != t2:
        _write(f2, t2_new)
        changed.append(f2.name)

print("patched:", changed or "nothing (check regex vs file layout)")
# Fail loudly if a fresh (unpatched) tree matched nothing: a silent no-op here
# turns into an unrelated clang <cmath> error later in the build.
if not changed and ("// " + GUARD not in _read(f1)
                    and "/* " + GUARD not in _read(f2)):
    sys.exit("ERROR: guard regex matched nothing on a fresh tree - check file layout")
