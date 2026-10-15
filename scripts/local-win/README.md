# local-win — this machine's build entry points (fork of kvmem-llama.cpp 0.18.0)

All scripts assume the repo at `E:\Works\KVMem\kvmem-src`, the portable CUDA
13.2 toolkit at `E:\Works\KVMem\cuda-toolkit`, and VS2022. `env.bat` sets that
environment up (call it from other scripts).

## Layout change vs the pre-0.18.0 fork

There is no `llama.cpp/` submodule branch anymore. Upstream 0.18.0 pins
`backends/llamacpp` to a ggml-org base commit and applies versioned patches
at build time; the fork adds a second overlay patch on top:

- `backends/patches/llamacpp-kvmem.patch` — official, sha-pinned in
  `backends/versions.json` (untouched by the fork)
- `backends/patches/llamacpp-kvmem-fork.patch` — fork extras (gemma4 capture
  hooks, batched `seq_rm_logical_ranges`, kv-cells holes fix, getenv caching,
  LM Studio parallel clamp), registered via the `fork_patch` keys that
  `scripts/prepare-backends.py` applies after the official patch

The submodule HEAD always stays at the base commit; never commit inside
`backends/llamacpp`. The old `llama/trunk` build branch survives only on the
GitHub fork for rollback.

## Targets

```
build.bat server     llama-kvmem-server   -> build-backends\llamacpp\bin\
build.bat lmstudio   KVMEM_DEFAULT_ON llama-server -> build-lmstu-v018\bin\
                                          (packaged via E:\Works\KVMem\package-lmstudio-runtime.sh)
build.bat test       host-side CTest (kvmem store/tier/runtime suites)
build.bat all        server, then lmstudio
```

Default CUDA archs `89-real;120a-real` cover the 4090 + 5080. Pass jobs as
the first argument (default 8).

## LM Studio runtime knobs (env, no CLI)

`KVMEM_ENABLE` (0 = stock multi-slot, 1 = on; unset = on in these builds),
`KVMEM_BUDGET` (default 32768), `KVMEM_GEN_RESERVE` (16384),
`KVMEM_CACHE_TYPE` (f16/f32/q8_0/q5_0/q4_0; launcher f16 lowered to q8_0),
`KVMEM_CPU_BYTES` / `KVMEM_NVME_BYTES` / `KVMEM_NVME_DIR`, plus the full
`KVMEM_*` set in `llama_kvmem_apply_env_once`. `--parallel > 1` (LM Studio's
default 4) is clamped to 1 while KVMem is enabled.
