# KVMem disk (NVMe/SSD) spill tier on Windows — implementation plan

> **Goal:** make KVMem's swapped-out KV live on **disk** instead of consuming all
> host RAM. Today the evicted history lands in a *pinned* CPU arena
> (`--kvmem-cpu-gb`, non-pageable RAM). On a 47 GB box with an 11 GB 27B model,
> filling ~256K of context exhausts RAM (measured: peak 39.4 GB used at a 40K
> fill; a 244K fill would OOM/freeze). A disk tier moves the cold KV to an NVMe
> file and keeps only a small pinned working set in RAM.
>
> **Scope of this doc:** design/plan §§1-8, plus §9 recording the completed
> implementation and on-hardware validation (2026-09-24). Code references are
> to pin `b81c99b` in this tree.

---

## 1. Good news: the disk tier already exists — it is only *disabled* on Windows

`kvmem/include/kvmem/nvme_kv_tier.hpp` is a complete, tested `NvmeKvTier`:

- fixed-slot arena file (`total_bytes / slot_bytes` slots), positional `pread`/`pwrite`
  so stage-in reads and stage-out writes run safely on **different worker threads**;
- batch spans **coalesce adjacent full records into one syscall**;
- LRU eviction (`place_block_evicting`, `lru_`) and a free-slot stack;
- `drop_page_cache` (`posix_fadvise(DONTNEED)` + `sync_file_range`) so the kernel
  does not keep a *second* copy of the arena in RAM;
- `direct_read` (`O_DIRECT`) for aligned pinned reads; `preallocate`
  (`posix_fallocate`); ephemeral arena via **open-then-`unlink`** (blocks auto-reclaim
  on exit/SIGKILL); `durable`/`direct_mapped`/`read_only`+CoW `overlay` archive modes.

The runtime already chains **GPU → pinned-CPU → NVMe**:
`KvMemRuntime` holds `cpu_tier_` + `nvme_tier_`, exposes `spill_outgoing()`,
`admit_incoming()` ("host/NVMe → GPU"), `spill_bytes_to_nvme()`, and blocks carry
`nvme_slot` / `tier == KvTier::SSD`. `kvmem_runtime_test.cpp::
test_cpu_full_spills_to_nvme_and_roundtrips` proves the CPU-full → NVMe spill +
round-trip. A second consumer, `raw_kv_store.cpp`, uses `NvmeKvTier` for the
`--kvmem-raw-k-nvme` raw-K path. The CLI plumbs `--kvmem-nvme-gb`, `--kvmem-nvme-dir`,
`--kvmem-raw-k-nvme`; the adapter wires `nvme_bytes`/`nvme_dir`.

**So this is a platform-port task, not a redesign.** The only things standing in the
way are: (a) every POSIX syscall lives in `nvme_kv_tier.hpp`; (b) CMake hard-blocks
it on Windows; (c) a couple of `/tmp/...` path defaults.

## 2. The gate is compile-time, so nothing downstream needs runtime changes

- `kvmem/CMakeLists.txt`: `WIN32 -> KVMEM_NVME_DEFAULT=OFF` and a
  `FATAL_ERROR` if `KVMEM_ENABLE_NVME=ON` on Windows. It publishes
  `-DKVMEM_ENABLE_NVME=$<BOOL:...>` to every consumer.
- `nvme_kv_tier.hpp` has a clean `#else` `NvmeKvTier` stub (all ops throw
  "NVMe offload is disabled") so the memory-only build links unchanged.
- `llama-kvmem-cli.cpp:223` and the adapter guard NVMe with
  `#if !KVMEM_ENABLE_NVME`.

Therefore, once libkvmem is **built with `KVMEM_ENABLE_NVME=ON` on Windows**, the
CLI/adapter gates compile out and the whole chain activates with **no logic changes**.

## 3. Port surface = one file, ~8 platform primitives

`grep` shows all POSIX symbols (`::open/pread/pwrite/unlink/close`, `posix_fallocate`,
`posix_fadvise`, `sync_file_range`, `fdatasync`, `O_DIRECT`, `mkdir`, `stat/S_IFDIR`,
`EINTR`, `<fcntl.h>/<unistd.h>/<sys/stat.h>`) are confined to **`nvme_kv_tier.hpp`**
(41 hits; `raw_kv_store` uses only the class API). Recommended structure: put a thin
`PlatformFileHandle` behind `#if defined(_WIN32)` inside that header (keep the
buffered `fd_` + optional unbuffered `direct_fd_` split), leaving the tier logic intact.

| Need | POSIX today | Win32 mapping |
|---|---|---|
| open/create | `::open(O_CREAT\|O_RDWR\|O_CLOEXEC[+O_DIRECT])` | `CreateFileW(GENERIC_*, CREATE_ALWAYS, FILE_SHARE_*, …, FILE_FLAG_RANDOM_ACCESS[+FILE_FLAG_OVERLAPPED][+FILE_FLAG_NO_BUFFERING])` |
| close | `::close` | `CloseHandle` |
| positional read (no shared cursor) | `::pread` | `ReadFile` with a per-call `OVERLAPPED{Offset,OffsetHigh}` (positional, thread-safe) + `GetOverlappedResult`; simpler stopgap: `SetFilePointerEx`+`ReadFile` under a per-handle `SRWLOCK` |
| positional write | `::pwrite` | `WriteFile` with `OVERLAPPED` offset |
| ephemeral arena | open-then-`::unlink` | `FILE_FLAG_DELETE_ON_CLOSE` (native, cleaner than unlink) |
| preallocate | `posix_fallocate` | `SetFilePointerEx(end,size-1)`+`SetEndOfFile` (sparse extend) |
| flush | `sync_file_range` / `fdatasync` | `FlushFileBuffers` (whole-file) |
| drop page cache | `posix_fadvise(DONTNEED)` | `FILE_FLAG_NO_BUFFERING` on the read path (bypass cache); otherwise rely on the page cache being **reclaimable** (the key win vs pinned RAM) |
| `O_DIRECT` reads | `O_DIRECT` | `FILE_FLAG_NO_BUFFERING` (needs sector-aligned buffer/offset/size) |
| mkdir / stat | `mkdir` / `stat`+`S_IFDIR` | `CreateDirectoryW` / `GetFileAttributesW` |
| `EINTR` retry | needed | no-op on Windows |

Do **not** use the MSVC CRT `_open`/`_read`/`_write` for the positional path —
they share a file cursor (not thread-safe) and lack `pread`/`pwrite`.

## 4. Constraints & gotchas

- **Alignment for `FILE_FLAG_NO_BUFFERING`**: offset, byte-count, and buffer address
  must be sector multiples (512, some drives 4096). The tier's slot layout
  (`slot_offset = slot * slot_bytes`) must be a sector multiple → require
  `slot_bytes % 4096 == 0`, and keep reads through the pinned (aligned) buffer.
  Keep a **buffered** primary descriptor and add the **NO_BUFFERING** descriptor only
  for the aligned fast path (mirrors the existing `fd_`/`direct_fd_` split), so the
  tier still works on any filesystem.
- **Path defaults**: `llama-memory-kvmem.cpp` and the CLI default `nvme_dir` to
  `/tmp/kvmem_nvme`. On Windows use `std::filesystem::temp_directory_path()`
  (already used by the tests) or `%LOCALAPPDATA%\kvmem\nvme`, and build paths with
  `std::filesystem::path` (not string `+ "/"`).
- **Delete-on-close vs unlink**: `FILE_FLAG_DELETE_ON_CLOSE` gives the exact
  ephemeral semantics; the `durable` mode (keeps the file for cross-process archives)
  must NOT set it — same branch as today's `if (!cfg_.durable) unlink(...)`.
- **Page cache**: on Linux `drop_page_cache` bounds RAM. On Windows, `FILE_FLAG_NO_BUFFERING`
  on the hot path plus small pinned CPU tier achieves the goal; a plain buffered port
  still relieves RAM pressure because page-cache pages are reclaimable (unlike pinned).
- **Disk space + wear**: 256K q8_0 KV ≈ 8.7 GB/file; durable/direct-mapped archives
  can be tens of GB. Use a dedicated NVMe (avoid the OS/system drive) to prevent
  I/O contention; ephemeral arenas are sparse so they cost only touched slots.

## 5. Performance model (honest expectations)

- **Reads are selective**: retrieval brings back only the top-k blocks that fit the
  GPU budget (`budget 36864` → ~288×4.25 MB ≈ 1.2 GB staged per turn), and the
  recent + sink + hot working set usually stays in the small pinned CPU tier, so
  cold NVMe reads are the minority. Prefetching the selected spans on the existing
  worker/spill stream hides most latency behind compute.
- A PCIe 4.0 NVMe is ~5-7 GB/s seq / ~0.1 ms random-4K; a 4.25 MB slot read ≈ ~1 ms
  buffered. Overlapped/batched (`write_spans`/`read_spans` coalescing already exist)
  keeps per-turn retrieval overhead in the low hundreds of ms at 256K — a large win
  vs today's RAM ceiling (256K simply does not fit) and vs a freeze.
- This is for the **cold-history** tier; GPU<->CPU pinned traffic (the hot path) is
  unchanged. Do not expect throughput gains on fits-in-RAM contexts; the benefit is
  *capability* (much longer contexts) at modest latency cost.

## 6. Phased implementation

- **P0 — port + unit tests (no perf work).** Add the Win32 primitive layer in
  `nvme_kv_tier.hpp`; `FILE_FLAG_DELETE_ON_CLOSE` for ephemeral; buffered primary +
  `FILE_FLAG_NO_BUFFERING` optional read fd; `SetFilePointerEx`/`ReadFile`+`OVERLAPPED`.
  Remove the `FATAL_ERROR`; add `option(KVMEM_ENABLE_NVME … ON)` on Windows. Run
  `nvme_kv_tier_test` + `kvmem_runtime_test` (spill round-trip) + `raw_kv_store_test`.
- **P1 — wire the CLI/adapter defaults.** Portable `nvme_dir` default; expose
  `--kvmem-nvme-gb/--kvmem-nvme-dir`; rebuild libkvmem `-DKVMEM_ENABLE_NVME=ON` +
  llama-kvmem (`build-hip.ps1`), regenerate `patches/llama-kvmem-current.patch` if
  adapter edits are needed.
- **P2 — real validation.** With `safe-run.ps1` (watchdog), run Qwen3.8-27B @256K with
  `--kvmem-cpu-gb 2 --kvmem-nvme-gb 40` and a deep ~244K fill; confirm RAM stays
  bounded (pinned arena ~2 GB not 10 GB), the file is created + reclaimed on exit,
  needle recall holds, and measure `retrieval_ms` delta. Compare to the 9B.
- **P3 — tuning.** Prefetch-on-the-spill-stream for the selected blocks;
  `direct_mapped`/`durable` archive mode on Windows (prebuilt 256K KV archives);
  double-buffered O_DIRECT-style aligned reads; SSD-drive placement guidance.

## 7. Effort & risk

Small-to-moderate: ~1 file of platform code (a few hundred lines) + CMake + path
defaults. The tier's data model, eviction, batching, and the GPU↔CPU↔SSD wiring are
already written and unit-tested; the adapter/CLI activate it for free. Main risks are
`NO_BUFFERING` alignment and `DELETE_ON_CLOSE`/durable semantics — both handled by
keeping a buffered primary descriptor. Highest-value first step: **P0**, which makes
`--kvmem-nvme-gb` work on Windows and unblocks 256K+ contexts that currently cannot
fit RAM.

## 8. Alternatives considered

- **`mmap` the arena instead of positional I/O.** Attractive (OS manages paging) but
  reintroduces a page-cache copy that is only *reclaimable*, gives less control over
  eviction ordering vs the pinned tier, and needs a second path for the ephemeral
  semantics; more churn than porting the existing positional API. Not chosen.
- **`FILE_FLAG_NO_BUFFERING` everywhere.** Maximizes cache bypass but forces every
  read/write to sector alignment and complicates the short/unaligned paths; better as
  an optional fast-path descriptor (as the code already splits `fd_`/`direct_fd_`).
- **Grow RAM / smaller context.** Avoids the work but defeats KVMem's purpose; the
  disk tier is the scalable answer for multi-hundred-K agent sessions.

---

## 9. Implementation status (2026-09-24): DONE - P0/P1 landed, P2 validated on hardware

**P0 (port, TDD).** New `kvmem/src/host/nvme_platform.cpp` implements the
platform primitives (declared in `nvme_kv_tier.hpp`, `windows.h` confined to the
.cpp so HIP/clang TUs stay clean): Win32 positional I/O via
`FILE_FLAG_OVERLAPPED` + per-call `OVERLAPPED` offsets/events (1 GiB DWORD
clamp; EOF mapped to `pread()==0` semantics), POSIX-semantics ephemeral delete
via `FileDispositionInfoEx{POSIX_SEMANTICS}` with `DeleteFileW` fallback,
full share mode (`READ|WRITE|DELETE`) to mirror POSIX non-mandatory sharing,
`SetFileInformationByHandle(FileEndOfFileInfo)` preallocation, and a
documented no-op cache-drop (Windows standby pages are OS-reclaimable - the
actual goal vs pinned RAM). CMake defaults `KVMEM_ENABLE_NVME=ON` everywhere;
the WIN32 `FATAL_ERROR` gate is gone; the CLI/adapter `#if !KVMEM_ENABLE_NVME`
guards light up automatically. RED was verified first (C1083 `unistd.h`), then
three root-cause bugs were found and fixed during GREEN: (a)
`FileDispositionInfoEx` is enum **18**, and `0x4` is `DO_NOT_DISCARD`, not
"delete"; (b) `std::filesystem::exists` **throws** `ACCESS_DENIED` on
delete-pending names (test harness now uses the `error_code` overload via
`name_visible()`); (c) a delete-pending name cannot be recreated while the old
handle lives (raw-store test uses a distinct arena file - POSIX-equivalent).
Host tests **5/5 green (MSVC)**; full HIP toolchain build **13/13 ctest green**.

**P1 (wiring).** Portable defaults: `/tmp/kvmem_nvme` ->
`std::filesystem::temp_directory_path()/"kvmem_nvme"` (adapter x3 + help
texts); `build.ps1`/`build-hip.ps1` pass `-DKVMEM_ENABLE_NVME=ON`;
`package.ps1` now requires an NVMe-enabled cache and publishes
`nvme_supported=true`; `test-package.ps1` passes. CodeReview findings (errno
propagation for `posix_fadvise`, packaging-test alignment, stale docs) all
fixed before hardware tests, as requested.

**P2 (hardware validation, Qwen3.8-27B UD-IQ3_S 11.2 GB, RX 9070 XT 16 GB,
47 GB RAM, arena on the OS NVMe).**

| Test | Config | Result |
|---|---|---|
| 32K tier probe | `cpu-gb 0.25, nvme-gb 2`, 10K prompt | `nvme_slots=481` active; prefill 788 t/s; temp dir reclaimed **empty** at exit |
| **256K deep fill** | `cpu-gb 2, nvme-gb 16`, **244,455-token** prompt | prefill **375 t/s** (10.9 min), decode 27.6; RAM peak 39.9 GB used / **min-free 8.4 GB** (watchdog never fired; the large share is *reclaimable* page cache, not pinned arena); VRAM peak 16.1 GB (summed counter); retrieval 3.48 s = **0.5% of prefill** (stage_out 1.41 + admit 1.59 + score 0.08); temp dir **empty** after exit |
| Needle recall | marker planted at ~200K depth | **HIT** - model answered `cobalt-heron-771` exactly |

Eviction arithmetic confirms disk usage: 1620 blocks pressured out vs only 481
CPU-tier slots, so ~1139 blocks (~4.8 GB) resided on NVMe while recall stayed
exact. The same deep fill with the RAM-only arena (`cpu-gb 10`) was projected
to exhaust the 47 GB machine and was deliberately not attempted - this was the
capability the disk tier was ported for.

**P3 (tuning, data-driven).** Retrieval is 0.5% of deep prefill, so
spill-stream prefetch and double-buffered aligned reads are **deferred for
lack of a measured bottleneck**; archive modes (durable/direct-mapped/
read-only+overlay) work through the same platform layer and are unit-covered;
SSD placement: any NVMe with >20 GB free is fine (the arena is sparse and
self-deleting; the OS drive worked without issue), a dedicated drive merely
avoids I/O contention.

**Validated Windows recipe (27B @ 256K, RAM-bounded):**

```
llama-kvmem-cli -m Qwen3.8-27B-UD-IQ3_S.gguf -c 262144 -ngl 99 ^
  --kvmem --kvmem-method retrieval --kvmem-budget 36864 --kvmem-gen-reserve 16384 ^
  --kvmem-block-tokens 128 --kvmem-cpu-gb 2 --kvmem-nvme-gb 16 ^
  -ctk q8_0 -ctv q8_0 --spec-type none
```

### Ultra-review pass (2026-09-24, second-round deep review + fixes + retest)

A second full review over 8 deep-dive angles (concurrency, error paths, leaks,
NO_BUFFERING alignment, POSIX semantic drift, security, build/packaging, test
integrity) found **1 Critical + 6 Warnings + 8 Suggestions - all verified and
fixed**:

- **C1 encoding contract**: MSVC `path::string()`/`argv` emit **ACP** bytes but
  `to_wide()` decoded strictly as UTF-8 - non-ASCII usernames/paths would fail
  at startup (invisible on ASCII-only test machines). Fixed: strict UTF-8 probe
  (`MB_ERR_INVALID_CHARS`) with ACP fallback in `to_wide()`, `.u8string()` on
  all producers, plus a non-ASCII regression sentinel test (UTF-8 π + GBK
  bytes, spelled as byte escapes to keep sources ASCII). Residual ambiguity
  (documented, accepted): on an ACP system, an ACP byte sequence that happens
  to be valid UTF-8 decodes as UTF-8 - callers should pass UTF-8
  (`.u8string()`) for non-ASCII paths, which all in-tree producers now do.
- **W1 durability barrier**: the Win32 `drop_cached_range` no-op had lost
  Linux's `sync_file_range(WAIT_AFTER)` barrier - ENOSPC would surface in the
  lazy writer *after* the RAM copy was released (silent KV corruption).
  Fixed: `FlushFileBuffers` on the write path + one-time policy announcement.
- **W2 fallocate**: `FileEndOfFileInfo` alone reserves nothing;
  `FileAllocationInfo` alone does not set EOF. The **new**
  `test_preallocate_sets_size` caught this immediately - fixed by doing both
  (allocation first so ENOSPC fails up front, matching posix_fallocate).
- **W3**: `CloseHandle` could clobber the last-error behind the tier's only
  failure message - saved/restored around the close.
- **W4**: basename guard now also rejects `\` and `.`/`..` (ephemeral mode
  POSIX-deletes the target - traversal would delete unintended files).
- **W5**: `build.ps1 -DisableNvme` switch (define/targets/ctest follow);
  OFF path re-verified 5/5.
- **W6**: the "unit-covered" archive-mode claim was false - closed with 4 new
  tests: durable roundtrip (OPEN_EXISTING + name survives close + read-only
  refuses writes), preallocate size, `FILE_FLAG_NO_BUFFERING` direct read
  (4096-aligned), overlay CoW (base stays byte-identical).
- **S1-S8**: ctor RAII handle guard (open_overlay throw leaked descriptors),
  io_loop catch now resets flushing/inflight bookkeeping, `name_visible()`
  error whitelist (only NOT_FOUND/PATH_NOT_FOUND/ACCESS_DENIED count as gone),
  degraded-delete hint for concurrent instances, global flush-mutex intent
  documented, `nvme_platform.cpp` conditionally compiled (MSVC LNK4221),
  test comment accuracy, docs alignment.

**Test matrix after fixes**: host ON 5/5 (tier test now 11 sub-tests), host
OFF (`-DisableNvme`) 5/5, HIP toolchain **13/13**.

**Hardware retest (post-fix, with a 5.1 GB background process competing for
RAM the whole time)**: 32K probe OK; 40K@256K OK (decode 25.2 vs 25.3-25.9
pre-fix - no regression; tiers + reclaim OK); two 244K deep-fill attempts were
**protectively killed by the watchdog** at 4.85/4.88 GB free (peak 43.4 GB
both times, exactly baseline + the measured 5.1 GB background hog - the fixes
add no RAM-path change); a **163K deep fill completed**: 163,009 tokens, 972
blocks pressured out vs 481 CPU slots (~2.1 GB resident on NVMe), **needle
recall HIT at ~122K depth** (`Answer: cobalt-heron-771`), retrieval 3.35 s,
RAM min-free 5.8 GB, temp dir reclaimed empty. The watchdog kills demonstrate
the safety design working exactly as intended under hostile memory conditions.

### A/B head-to-head: NVMe tier vs RAM-only (2026-09-24)

Same model (Qwen3.8-27B IQ3_S), same 122,286-token prompt @ `-c 262144`,
same q8_0/budget/reserve, needle planted at ~82K depth; only the spill tier
differs. A = RAM-only (`cpu-gb 4`, all 539 evicted blocks fit in the pinned
arena); B = disk (`cpu-gb 1 + nvme-gb 16`, ~299 blocks forced to NVMe).

| Metric | A: RAM-only | B: NVMe tier | Delta |
|---|---:|---:|---|
| RAM peak used | 41,564 MB | 40,600 MB | **-964 MB** |
| RAM min free | 6,748 MB | **7,712 MB** | +964 MB headroom |
| C: disk delta (during run) | 1,546 MB (**pagefile stress** - RAM-only pushed the OS to page) | 3,056 MB (~1.3 GB intentional KV spill + pagefile) | B moves cold KV to disk *by design* instead of forcing the OS to page |
| VRAM peak | 15,438 MB | 15,439 MB | identical (GPU budget unchanged) |
| prefill | 220.1 t/s (555.7 s) | 219.9 t/s (556.0 s) | **identical** (GPU-bound) |
| decode | 25.2 t/s | 25.3 t/s | identical |
| retrieval (end-of-prefill) | 2,708 ms | 3,233 ms | **+525 ms (+19%)** = NVMe cold-block read-back; 0.09% of the total run |
| needle recall @82K | HIT | HIT | equal quality |
| temp dir after exit | n/a | EMPTY (reclaimed) | |

Scaling note: at 122K the RAM-only arena only needed 4 GB, so the RAM saving
is modest (~1 GB); the gap widens with depth - a 244K fill needs an ~8.7 GB
arena for RAM-only (measured to hit the 5 GB watchdog floor and get killed
under background load), while the NVMe configuration keeps its 1-2 GB pinned
arena constant and puts the rest on disk (163K/244K fills completed).
Verification caveat learned here: the `\LogicalDisk` performance counter
**under-reports** free-space changes of POSIX-deleted (delete-pending) arena
files - disk deltas above were measured with direct `Get-PSDrive`/NTFS
polling; spill itself was independently proven via `KVMEM_TRACE`
(`stage_out_nvme` x44 in a forced-spill probe) plus external polling (-132 MB).

### Clean-background retest after local commits (2026-09-24)

Commits: `7dc917a` (AMD HIP port), `0f5ca89` (NVMe Win32 port). Post-commit
matrix: host 5/5 (MSVC), HIP 13/13 (clang). The 5.1 GB background hog was
terminated (user-authorized) before these runs.

| Metric (122K fill, clean) | A: RAM-only (`cpu-gb 4`) | B: NVMe (`cpu-gb 1`+`nvme-gb 16`) |
|---|---:|---:|
| RAM peak used / min free | 39,362 / 8,950 MB | 38,622 / **9,690 MB** |
| VRAM peak | 15,386 MB | 15,386 MB |
| C: disk delta | **7 MB** (zero spill) | **3,061 MB** (KV on disk) |
| prefill / decode | 218.6 / 25.1 t/s | 214.7 / 25.5 t/s (-1.8% prefill) |
| retrieval | 3,864 ms | 4,032 ms (+4.3%) |
| needle @82K | HIT | HIT |
| temp reclaimed | n/a | EMPTY |

**244K deep fill (NVMe, clean)**: completed - prefill 206 t/s (19.7 min),
decode 25.7, retrieval 4.2 s, **C: delta 10.9 GB** (the disk absorbed the
overflow), needle HIT at ~200K depth, RAM min-free 5.2 GB. This is at the
capacity edge of a 47 GB machine with an 11.2 GB model; the RAM-only
configuration cannot run 244K at all (needs an ~8.7 GB pinned arena on top).

**Regression probes (32K, post-commit binary)**: pure-RAM 786.4 t/s,
NVMe-config 785.4 t/s, pre-fix NVMe-config was 788 - the NVMe changes add no
measurable prefill cost. The 244K prefill difference vs the pre-review run
(375 -> 206 t/s) tracks host-memory headroom, not code: deep-fill harvest
D2H is host-bound (`d2h_wait` 630 -> 1160 s) and the idle baseline had grown
~2 GB, pushing min-free near the floor where the OS reclaims page cache
aggressively.

**Hostile-environment datapoint (kept for contrast)**: with the 5.1 GB
background hog resident, RAM-only 122K was watchdog-KILLED (min-free
4,963 MB, pagefile delta 4.6 GB) while NVMe 122K completed (min-free
5,212 MB) with needle HIT - identical workload, identical machine.
