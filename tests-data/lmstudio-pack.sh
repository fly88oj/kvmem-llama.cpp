#!/usr/bin/env bash
# Echo the directory of the newest installed KVMem LM Studio backend pack.
# Our packs carry a "(KVMem)" marker in display-data.json; official packs do not,
# so the marker - not the version number - is what identifies a usable tree.
# Override with KVMEM_PACK=<dir> to point at another build.
set -u
if [ -n "${KVMEM_PACK:-}" ]; then
    [ -x "$KVMEM_PACK/llama-server.exe" ] || { echo "KVMEM_PACK has no llama-server.exe: $KVMEM_PACK" >&2; exit 1; }
    echo "$KVMEM_PACK"
    exit 0
fi
root="$HOME/.lmstudio/extensions/backends"
best=""
for dir in "$root"/llama.cpp-win-x86_64-amd-rocm-avx2-*/; do
    [ -f "$dir/display-data.json" ] || continue
    grep -q "KVMem" "$dir/display-data.json" || continue
    [ -x "$dir/llama-server.exe" ] || continue
    best="$dir"
done
if [ -z "$best" ]; then
    echo "no installed KVMem pack under $root - run scripts/windows/make-lms-extension.ps1 -Install" >&2
    exit 1
fi
printf '%s\n' "${best%/}"
