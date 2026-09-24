"""Revert the KVMEM_HIP_SKIP_MATH_FWD guard insertions made by
patch-rocm-headers.py.

patch-rocm-headers.py guards TWO clang headers, so this revert must undo both:
  1. __clang_cuda_math_forward_declares.h  (wrapped with // GUARD begin/end)
  2. __clang_hip_cmath.h                    (wrapped with a /* GUARD */ marker)

Reverting only the first (the previous behaviour) left a residual #ifndef block
in __clang_hip_cmath.h, so a later rebuild behaved inconsistently with a fresh
tree. This script is idempotent: running it on an already-clean tree is a no-op.

Usage: python revert-guards.py <path-to-_rocm_sdk_devel>
"""
from pathlib import Path
import re
import sys

GUARD = "KVMEM_HIP_SKIP_MATH_FWD"

if len(sys.argv) < 2:
    sys.exit("usage: revert-guards.py <path-to-_rocm_sdk_devel>")

devel = Path(sys.argv[1])
inc = next(iter(devel.glob("lib/llvm/lib/clang/*/include")), None)
if inc is None:
    sys.exit("clang include dir not found under " + str(devel))

# (filename, pattern that captures the original block in group 1). Each pattern
# mirrors exactly how patch-rocm-headers.py wrapped that file.
targets = [
    ("__clang_cuda_math_forward_declares.h", re.compile(
        r"// " + GUARD + r" begin\n"
        r"#ifndef " + GUARD + r"\n"
        r"(.*?)\n#endif\n"
        r"// " + GUARD + r" end\n",
        re.S)),
    ("__clang_hip_cmath.h", re.compile(
        r"/\* " + GUARD + r" \*/\n"
        r"#ifndef " + GUARD + r"\n"
        r"(.*?)\n#endif\n",
        re.S)),
]

total = 0
for name, pat in targets:
    f = inc / name
    if not f.exists():
        print("skip (missing):", name)
        continue
    # Universal-newline read: the ROCm-on-Windows wheel ships CRLF clang headers,
    # so read_text normalizes \r\n -> \n and the \n-based regexes match; the
    # platform write_text turns \n back into CRLF, preserving the original endings.
    t = f.read_text(encoding="utf-8")
    t2, n = pat.subn(lambda m: m.group(1) + "\n", t)
    if n:
        f.write_text(t2, encoding="utf-8")
    total += n
    print(f"{name}: removed {n} guard block(s)")
    for line in t2.splitlines():
        if "KVMEM" in line:
            sys.exit(f"RESIDUE in {name}: " + repr(line))

print("total guard blocks removed:", total)
