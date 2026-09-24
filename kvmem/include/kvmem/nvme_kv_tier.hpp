#pragma once

// Standalone header users get the tier on every supported platform; CMake
// publishes an explicit value to every consumer of libkvmem.
#ifndef KVMEM_ENABLE_NVME
#define KVMEM_ENABLE_NVME 1
#endif

// NVMe KV tier — fixed-slot metadata plus positional byte I/O.
//
// The legacy API remains synchronous. Positional pread/pwrite removes the
// shared FILE* cursor so stage-out writes and stage-in reads may safely run on
// different host workers. Batch spans coalesce adjacent full records into one
// syscall when both their file offsets and buffer ranges are contiguous.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if KVMEM_ENABLE_NVME

// Platform file primitives, implemented in src/host/nvme_platform.cpp.
// Positional and thread-safe (pread/pwrite semantics); windows.h and the
// POSIX headers stay confined to that TU so this header remains includable
// from HIP/clang translation units.
namespace kvmem {
namespace platform {

#ifdef _WIN32
using FileHandle = void *;  // HANDLE; INVALID_HANDLE_VALUE when closed
#else
using FileHandle = int;     // fd; -1 when closed
#endif

FileHandle invalid_handle();
bool valid(FileHandle h);
std::string last_error();
// create/truncate mirror O_CREAT/O_TRUNC. delete_on_close reproduces
// open-then-unlink with POSIX semantics (the directory entry disappears
// immediately; data is reclaimed at last close / process death).
// no_buffering requests O_DIRECT / FILE_FLAG_NO_BUFFERING; unsupported
// platforms return an invalid handle so callers degrade to buffered I/O.
FileHandle open_file(const std::string & path, bool read, bool write,
                     bool create, bool truncate, bool delete_on_close,
                     bool no_buffering);
void close_file(FileHandle h);
// One positional transfer. May be partial (caller loops); returns true with
// *done == 0 at EOF, matching pread() == 0. False on error (last_error()).
bool transfer_at(FileHandle h, void * buf, uint64_t bytes, uint64_t offset,
                 uint64_t * done, bool write);
// 0 = ok, 1 = unsupported (degrade to sparse growth), -1 = error.
int fallocate(FileHandle h, uint64_t bytes);
// Relinquish page-cache residency of [offset, offset+bytes); writeback first
// when `write`. False (with last_error()) when the platform cannot.
bool drop_cached_range(FileHandle h, uint64_t offset, uint64_t bytes,
                       bool write);
// 0 = ok, -1 = exists but is not a directory, -2 = creation failed.
int ensure_dir(const std::string & dir);

}  // namespace platform
}  // namespace kvmem

#endif

namespace kvmem {

struct NvmeKvTierConfig {
    std::string dir;
    // Separate logical tiers may share one directory while retaining
    // independent anonymous backing files. The directory entry is unlinked
    // immediately after open, so this is a collision-avoidance/debug label,
    // not a persistent cache name.
    std::string file_name = "kvmem_nvme.bin";
    uint64_t total_bytes = 0;
    uint64_t slot_bytes = 0;
    // Context-archive modes. `durable` keeps the directory entry instead of
    // unlinking it, so the arena survives the process; it also stops truncating
    // an existing file so a build can be resumed. `direct_mapped` assigns
    // slot == block_id and disables the free list and LRU eviction, which
    // removes the need to persist any slot table. `read_only` opens the arena
    // O_RDONLY so several processes can attach the same archive at once.
    bool durable = false;
    bool direct_mapped = false;
    bool read_only = false;
    // Reserve the complete arena before the first write. Archive builders know
    // their maximum direct-mapped capacity up front; preallocation avoids
    // interleaving raw-K and V extents into highly fragmented files and makes
    // ENOSPC fail at startup rather than hours into a build. Unsupported file
    // systems degrade to sparse growth with a warning.
    bool preallocate = false;
    // Copy-on-write overlay for a read-only arena. A session attached to a
    // sealed archive still produces KV of its own (the residual re-prefill, the
    // question, the decode), and evicting any of it needs somewhere to spill.
    // Writes land in an ephemeral sparse file at the same slot offsets and
    // subsequent reads of those slots come from the overlay, so the archive
    // stays byte-identical and several sessions can diverge from it at once.
    std::string overlay_dir;
    std::string overlay_file_name = "kvmem_overlay.bin";
    // Buffered I/O otherwise keeps a second copy of the SSD arena in the
    // kernel page cache.  For long contexts that copy can be tens of GiB and
    // defeats the explicit CPU-tier memory budget.  When enabled, completed
    // writes are range-written back and both reads and writes are advised
    // DONTNEED after the user buffer owns the data.
    bool drop_page_cache = false;
    // Read sealed/direct-mapped archive payloads through a second O_DIRECT
    // descriptor whenever the caller supplies naturally aligned pinned
    // staging memory.  Index rebuilds and short/unaligned compatibility reads
    // continue through the buffered descriptor.  Opening O_DIRECT is best
    // effort so the same archive remains portable across filesystems.
    bool direct_read = false;
};

struct NvmeSlotPlacement {
    int32_t slot = -1;
    int32_t evicted_block = -1;
};

struct NvmeIoSpan {
    int32_t slot = -1;
    uint64_t buffer_offset = 0;
    uint64_t bytes = 0;
};

struct NvmeBatchIoStats {
    uint64_t bytes = 0;
    uint64_t syscalls = 0;
    uint64_t duration_ns = 0;
    uint64_t cache_drop_bytes = 0;
    uint64_t cache_drop_failures = 0;
    uint64_t cpu_copy_bytes = 0;
    uint64_t cpu_copy_ns = 0;
    uint32_t cpu_copy_blocks = 0;
};

#if KVMEM_ENABLE_NVME
class NvmeKvTier {
public:
    explicit NvmeKvTier(NvmeKvTierConfig cfg) : cfg_(std::move(cfg)) {
        if (cfg_.slot_bytes > 0 && cfg_.total_bytes >= cfg_.slot_bytes) {
            slot_count_ = static_cast<uint32_t>(
                cfg_.total_bytes / cfg_.slot_bytes);
        }
        if (slot_count_ == 0 || cfg_.dir.empty()) return;

        ensure_dir(cfg_.dir);
        if (!is_plain_basename(cfg_.file_name)) {
            throw std::runtime_error(
                "NVMe KV tier file_name must be a non-empty basename");
        }
        path_ = cfg_.dir + "/" + cfg_.file_name;
        // The ephemeral (non-durable) arena is opened then deleted: both
        // platforms remove the directory entry immediately while the data
        // survives until the last handle closes, including abnormal exits.
        // This also prevents stale kvmem_nvme.bin files from accumulating
        // across evaluations.
        fd_ = platform::open_file(
            path_, /*read=*/true, /*write=*/!cfg_.read_only,
            /*create=*/!cfg_.read_only,
            /*truncate=*/!cfg_.read_only && !cfg_.durable,
            /*delete_on_close=*/!cfg_.durable, /*no_buffering=*/false);
        if (!platform::valid(fd_)) {
            throw std::runtime_error(
                "failed to open NVMe KV tier file: " + path_ + ": " +
                platform::last_error());
        }
        // Ctor-throw safety: a throwing constructor never runs ~NvmeKvTier,
        // so close any already-opened descriptors if a later step throws
        // (including bad_alloc from the free-slot vector below).
        struct HandleGuard {
            platform::FileHandle * primary;
            platform::FileHandle * direct;
            platform::FileHandle * overlay;
            bool armed = true;
            ~HandleGuard() {
                if (armed) {
                    platform::close_file(*overlay);
                    platform::close_file(*direct);
                    platform::close_file(*primary);
                }
            }
        } guard { &fd_, &direct_fd_, &overlay_fd_ };
        if (cfg_.direct_read && cfg_.durable && cfg_.read_only) {
            direct_fd_ = platform::open_file(
                path_, /*read=*/true, /*write=*/false, /*create=*/false,
                /*truncate=*/false, /*delete_on_close=*/false,
                /*no_buffering=*/true);
            if (!platform::valid(direct_fd_)) {
                std::fprintf(
                    stderr,
                    "[kvmem-io] direct_read_degraded=1 path=%s "
                    "message=%s action=buffered-fallback\n",
                    path_.c_str(), platform::last_error().c_str());
            } else {
                std::fprintf(
                    stderr,
                    "[kvmem-io] direct_read=1 path=%s alignment=%llu\n",
                    path_.c_str(),
                    static_cast<unsigned long long>(kDirectAlignment));
            }
        }
        if (cfg_.preallocate && !cfg_.read_only && cfg_.total_bytes > 0) {
            const int rc = platform::fallocate(fd_, cfg_.total_bytes);
            if (rc < 0) {
                const std::string message = platform::last_error();
                throw std::runtime_error(
                    "failed to preallocate NVMe KV tier file: " + path_ +
                    ": " + message);
            }
            if (rc != 0) {
                std::fprintf(stderr,
                             "[kvmem-io] preallocate_degraded=1 path=%s "
                             "bytes=%llu action=sparse-growth-fallback\n",
                             path_.c_str(),
                             static_cast<unsigned long long>(cfg_.total_bytes));
            } else {
                std::fprintf(stderr,
                             "[kvmem-io] preallocated path=%s bytes=%llu\n",
                             path_.c_str(),
                             static_cast<unsigned long long>(cfg_.total_bytes));
            }
        }
        if (cfg_.read_only && !cfg_.overlay_dir.empty()) open_overlay();
        if (cfg_.direct_mapped) {
            guard.armed = false;
            return;
        }
        free_slots_.reserve(slot_count_);
        for (uint32_t i = 0; i < slot_count_; ++i) {
            free_slots_.push_back(
                static_cast<int32_t>(slot_count_ - 1U - i));
        }
        guard.armed = false;  // every throwing step has completed
    }

    NvmeKvTier(const NvmeKvTier &) = delete;
    NvmeKvTier &operator=(const NvmeKvTier &) = delete;

    ~NvmeKvTier() {
        platform::close_file(direct_fd_);
        platform::close_file(fd_);
        platform::close_file(overlay_fd_);
    }

    bool enabled() const { return platform::valid(fd_) && slot_count_ > 0; }
    uint32_t slot_count() const { return slot_count_; }
    uint64_t slot_bytes() const { return cfg_.slot_bytes; }
    const std::string &path() const { return path_; }
    bool drops_page_cache() const { return cfg_.drop_page_cache; }
    bool direct_mapped() const { return cfg_.direct_mapped; }
    bool read_only() const {
        return cfg_.read_only && !platform::valid(overlay_fd_);
    }
    bool has_overlay() const { return platform::valid(overlay_fd_); }
    bool direct_reads() const { return platform::valid(direct_fd_); }

    // Declare block ids [begin,end) resident at their identity slots. Only
    // meaningful for a direct-mapped arena, where an attached archive knows
    // every block is present without replaying any placement history.
    void mark_present_range(uint32_t begin, uint32_t end) {
        if (!cfg_.direct_mapped) {
            throw std::runtime_error(
                "NVMe tier mark_present_range requires a direct-mapped arena");
        }
        std::lock_guard<std::mutex> lock(meta_mu_);
        for (uint32_t id = begin; id < end && id < slot_count_; ++id) {
            block_to_slot_[id] = static_cast<int32_t>(id);
        }
    }

    uint32_t free_slots() const {
        std::lock_guard<std::mutex> lock(meta_mu_);
        if (cfg_.direct_mapped) {
            return slot_count_ -
                   static_cast<uint32_t>(block_to_slot_.size());
        }
        return static_cast<uint32_t>(free_slots_.size());
    }

    uint32_t used_slots() const {
        return slot_count_ - free_slots();
    }

    uint64_t slot_offset(int32_t slot) const {
        return static_cast<uint64_t>(slot) * cfg_.slot_bytes;
    }

    int32_t block_slot(uint32_t block_id) const {
        std::lock_guard<std::mutex> lock(meta_mu_);
        auto it = block_to_slot_.find(block_id);
        return it == block_to_slot_.end() ? -1 : it->second;
    }

    NvmeSlotPlacement place_block(uint32_t block_id) {
        std::lock_guard<std::mutex> lock(meta_mu_);
        return place_block_locked(block_id);
    }

    NvmeSlotPlacement place_block_evicting(uint32_t block_id) {
        std::lock_guard<std::mutex> lock(meta_mu_);
        NvmeSlotPlacement out = place_block_locked(block_id);
        if (out.slot >= 0 || !free_slots_.empty()) return out;
        if (lru_.empty()) return out;
        const uint32_t victim = lru_.front();
        auto vit = block_to_slot_.find(victim);
        if (vit == block_to_slot_.end()) {
            erase_lru_locked(victim);
            return out;
        }
        const int32_t slot = vit->second;
        block_to_slot_.erase(vit);
        erase_lru_locked(victim);
        block_to_slot_[block_id] = slot;
        touch_locked(block_id);
        out.slot = slot;
        out.evicted_block = static_cast<int32_t>(victim);
        return out;
    }

    void release_block(uint32_t block_id) {
        std::lock_guard<std::mutex> lock(meta_mu_);
        auto it = block_to_slot_.find(block_id);
        if (it == block_to_slot_.end()) return;
        if (!cfg_.direct_mapped) free_slots_.push_back(it->second);
        block_to_slot_.erase(it);
        erase_lru_locked(block_id);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(meta_mu_);
        free_slots_.clear();
        if (!cfg_.direct_mapped) {
            for (uint32_t i = 0; i < slot_count_; ++i) {
                free_slots_.push_back(
                    static_cast<int32_t>(slot_count_ - 1U - i));
            }
        }
        block_to_slot_.clear();
        lru_.clear();
    }

    void touch(uint32_t block_id) {
        std::lock_guard<std::mutex> lock(meta_mu_);
        touch_locked(block_id);
    }

    int32_t lru_victim() const {
        std::lock_guard<std::mutex> lock(meta_mu_);
        return lru_.empty() ? -1 : static_cast<int32_t>(lru_.front());
    }

    void write_block(uint32_t block_id, const void *data, uint64_t bytes) {
        validate_io(data, bytes, "write");
        auto p = place_block(block_id);
        if (p.slot < 0) {
            throw std::runtime_error("NVMe KV tier is full");
        }
        write_slot(p.slot, data, bytes);
        touch(block_id);
    }

    void read_block(uint32_t block_id, void *data, uint64_t bytes) {
        validate_io(data, bytes, "read");
        const int32_t slot = block_slot(block_id);
        if (slot < 0) {
            throw std::runtime_error("NVMe block is not resident");
        }
        read_slot(slot, data, bytes);
        touch(block_id);
    }

    // The slot must already have been reserved with place_block(). These raw
    // positional methods do not touch metadata and are safe on worker threads.
    void write_slot(int32_t slot, const void *data, uint64_t bytes) const {
        write_slot_range(slot, 0, data, bytes);
    }

    void write_slot_range(int32_t slot, uint64_t slot_byte_offset,
                          const void *data, uint64_t bytes) const {
        if (read_only()) {
            throw std::runtime_error(
                "NVMe KV tier is attached read-only: " + path_);
        }
        validate_slot_range_io(
            slot, slot_byte_offset, data, bytes, "write");
        const uint64_t offset = slot_offset(slot) + slot_byte_offset;
        if (platform::valid(overlay_fd_)) {
            write_overlay_slot_range(slot, slot_byte_offset, data, bytes);
            if (cfg_.drop_page_cache) {
                (void) drop_cached_range(
                    overlay_fd_, offset, bytes, /*write=*/true);
            }
            return;
        }
        const platform::FileHandle fd = write_fd();
        pwrite_all(fd, data, bytes, offset);
        if (cfg_.drop_page_cache) {
            (void) drop_cached_range(fd, offset, bytes, /*write=*/true);
        }
    }

    void read_slot(int32_t slot, void *data, uint64_t bytes) const {
        read_slot_range(slot, 0, data, bytes);
    }

    void read_slot_range(int32_t slot, uint64_t slot_byte_offset,
                         void *data, uint64_t bytes) const {
        validate_slot_range_io(
            slot, slot_byte_offset, data, bytes, "read");
        const uint64_t offset = slot_offset(slot) + slot_byte_offset;
        const platform::FileHandle fd =
            read_fd_for_range(slot, data, bytes, offset);
        pread_all(fd, data, bytes, offset);
        if (cfg_.drop_page_cache && fd != direct_fd_) {
            (void) drop_cached_range(fd, offset, bytes, /*write=*/false);
        }
    }

    void write_spans(const std::vector<NvmeIoSpan> &spans,
                     const void *buffer, uint64_t buffer_bytes,
                     NvmeBatchIoStats *stats = nullptr) const {
        if (read_only()) {
            throw std::runtime_error(
                "NVMe KV tier is attached read-only: " + path_);
        }
        run_spans(spans, const_cast<void *>(buffer), buffer_bytes,
                  /*write=*/true, stats);
    }

    void read_spans(const std::vector<NvmeIoSpan> &spans,
                    void *buffer, uint64_t buffer_bytes,
                    NvmeBatchIoStats *stats = nullptr) const {
        run_spans(spans, buffer, buffer_bytes, /*write=*/false, stats);
    }

private:
    // Reject both platforms' path separators ('/' and Windows '\\') and the
    // "."/".." entries: file_name is appended to dir and, in ephemeral mode,
    // POSIX-deleted - a traversal escape would write to (or delete!) an
    // unintended path. Rejecting '\\' on POSIX is a deliberate cross-platform
    // tightening: such names are legal there but refused so arena file names
    // stay portable between platforms.
    static bool is_plain_basename(const std::string &n) {
        if (n.empty() || n == "." || n == "..") return false;
        return n.find_first_of("/\\") == std::string::npos;
    }

    static void ensure_dir(const std::string &dir) {
        const int rc = platform::ensure_dir(dir);
        if (rc == -1) {
            throw std::runtime_error(
                "NVMe KV tier path is not a directory: " + dir);
        }
        if (rc == -2) {
            throw std::runtime_error(
                "failed to create NVMe KV tier directory: " + dir + ": " +
                platform::last_error());
        }
    }

    void validate_io(const void *data, uint64_t bytes,
                     const char *op) const {
        if (!enabled()) {
            throw std::runtime_error("NVMe KV tier is disabled");
        }
        if (!data && bytes > 0) {
            throw std::runtime_error(
                std::string("NVMe ") + op + " null data");
        }
        if (bytes > cfg_.slot_bytes) {
            throw std::runtime_error(
                std::string("NVMe ") + op + " exceeds slot size");
        }
    }

    void validate_slot_io(int32_t slot, const void *data, uint64_t bytes,
                          const char *op) const {
        validate_io(data, bytes, op);
        if (slot < 0 || static_cast<uint32_t>(slot) >= slot_count_) {
            throw std::runtime_error(
                std::string("NVMe ") + op + " has invalid slot");
        }
    }

    void validate_slot_range_io(int32_t slot, uint64_t slot_byte_offset,
                                const void *data, uint64_t bytes,
                                const char *op) const {
        validate_slot_io(slot, data, bytes, op);
        if (slot_byte_offset > cfg_.slot_bytes ||
            bytes > cfg_.slot_bytes - slot_byte_offset) {
            throw std::runtime_error(
                std::string("NVMe ") + op + " range exceeds slot size");
        }
    }

    NvmeSlotPlacement place_block_locked(uint32_t block_id) {
        NvmeSlotPlacement out;
        if (cfg_.direct_mapped) {
            // Identity placement. No free list, no LRU, and therefore no slot
            // table to persist: a reader recovers the mapping from block_id.
            if (block_id >= slot_count_) return out;
            block_to_slot_[block_id] = static_cast<int32_t>(block_id);
            out.slot = static_cast<int32_t>(block_id);
            return out;
        }
        auto it = block_to_slot_.find(block_id);
        if (it != block_to_slot_.end()) {
            out.slot = it->second;
            touch_locked(block_id);
            return out;
        }
        if (free_slots_.empty()) return out;
        const int32_t slot = free_slots_.back();
        free_slots_.pop_back();
        block_to_slot_[block_id] = slot;
        touch_locked(block_id);
        out.slot = slot;
        return out;
    }

    void open_overlay() {
        ensure_dir(cfg_.overlay_dir);
        if (!is_plain_basename(cfg_.overlay_file_name)) {
            throw std::runtime_error(
                "NVMe KV tier overlay_file_name must be a non-empty basename");
        }
        const std::string overlay_path =
            cfg_.overlay_dir + "/" + cfg_.overlay_file_name;
        // Same rationale as the ephemeral arena: the overlay is scratch that
        // must not outlive the process, and the file stays sparse so it costs
        // only the slots the session actually diverges on. delete_on_close
        // reproduces open-then-unlink on both platforms.
        overlay_fd_ = platform::open_file(
            overlay_path, /*read=*/true, /*write=*/true, /*create=*/true,
            /*truncate=*/true, /*delete_on_close=*/true,
            /*no_buffering=*/false);
        if (!platform::valid(overlay_fd_)) {
            throw std::runtime_error(
                "failed to open NVMe KV tier overlay file: " + overlay_path +
                ": " + platform::last_error());
        }
        overlay_valid_ = std::unique_ptr<std::atomic<uint8_t>[]>(
            new std::atomic<uint8_t>[slot_count_]);
        for (uint32_t i = 0; i < slot_count_; ++i) {
            overlay_valid_[i].store(0, std::memory_order_relaxed);
        }
    }

    platform::FileHandle write_fd() const {
        return platform::valid(overlay_fd_) ? overlay_fd_ : fd_;
    }

    platform::FileHandle read_fd(int32_t slot) const {
        if (!platform::valid(overlay_fd_)) return fd_;
        uint8_t state = overlay_valid_[slot].load(std::memory_order_acquire);
        while (state == kOverlayInitializing) {
            std::this_thread::yield();
            state = overlay_valid_[slot].load(std::memory_order_acquire);
        }
        return state == kOverlayValid ? overlay_fd_ : fd_;
    }

    void write_overlay_slot_range(int32_t slot, uint64_t slot_byte_offset,
                                  const void *data, uint64_t bytes) const {
        // The slot is the coherence unit. A short final block or an isolated
        // main/MTP segment write must copy the untouched bytes from the archive
        // before the overlay can become visible; otherwise sparse-file zeros
        // would silently replace the other part of the record.
        uint8_t state = overlay_valid_[slot].load(std::memory_order_acquire);
        while (state != kOverlayValid) {
            if (state == kOverlayBase) {
                uint8_t expected = kOverlayBase;
                if (overlay_valid_[slot].compare_exchange_strong(
                        expected, kOverlayInitializing,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    try {
                        const bool full_slot =
                            slot_byte_offset == 0 && bytes == cfg_.slot_bytes;
                        if (!full_slot) {
                            constexpr uint64_t kCopyChunk = 1ull << 20;
                            std::vector<uint8_t> scratch(static_cast<size_t>(
                                std::min<uint64_t>(kCopyChunk,
                                                   cfg_.slot_bytes)));
                            for (uint64_t copied = 0;
                                 copied < cfg_.slot_bytes;) {
                                const uint64_t n = std::min<uint64_t>(
                                    scratch.size(), cfg_.slot_bytes - copied);
                                pread_all(fd_, scratch.data(), n,
                                          slot_offset(slot) + copied);
                                pwrite_all(overlay_fd_, scratch.data(), n,
                                           slot_offset(slot) + copied);
                                copied += n;
                            }
                        }
                        pwrite_all(overlay_fd_, data, bytes,
                                   slot_offset(slot) + slot_byte_offset);
                        overlay_valid_[slot].store(
                            kOverlayValid, std::memory_order_release);
                    } catch (...) {
                        overlay_valid_[slot].store(
                            kOverlayBase, std::memory_order_release);
                        throw;
                    }
                    return;
                }
                state = expected;
                continue;
            }
            std::this_thread::yield();
            state = overlay_valid_[slot].load(std::memory_order_acquire);
        }
        pwrite_all(overlay_fd_, data, bytes,
                   slot_offset(slot) + slot_byte_offset);
    }

    void pwrite_all(platform::FileHandle fd, const void *data, uint64_t bytes,
                    uint64_t offset) const {
        const uint8_t *src = static_cast<const uint8_t *>(data);
        uint64_t done = 0;
        while (done < bytes) {
            uint64_t n = 0;
            if (!platform::transfer_at(
                    fd, const_cast<uint8_t *>(src + done), bytes - done,
                    offset + done, &n, /*write=*/true) || n == 0) {
                throw std::runtime_error(
                    "NVMe positional write failed: " + platform::last_error());
            }
            done += n;
        }
    }

    void pread_all(platform::FileHandle fd, void *data, uint64_t bytes,
                   uint64_t offset) const {
        uint8_t *dst = static_cast<uint8_t *>(data);
        uint64_t done = 0;
        while (done < bytes) {
            uint64_t n = 0;
            if (!platform::transfer_at(fd, dst + done, bytes - done,
                                       offset + done, &n, /*write=*/false)) {
                throw std::runtime_error(
                    "NVMe positional read failed: " + platform::last_error());
            }
            if (n == 0) {
                throw std::runtime_error(
                    "NVMe positional read failed or reached unwritten data");
            }
            done += n;
        }
    }

    void run_spans(const std::vector<NvmeIoSpan> &spans, void *buffer,
                   uint64_t buffer_bytes, bool write,
                   NvmeBatchIoStats *stats) const {
        if (stats) *stats = NvmeBatchIoStats{};
        if (spans.empty()) return;
        if (!buffer) throw std::runtime_error("NVMe batch I/O null buffer");
        uint8_t *base = static_cast<uint8_t *>(buffer);
        // Writes to a CoW arena need per-slot copy-up semantics when a span is
        // shorter than the physical record. Overlay traffic is only the live
        // suffix of an attached archive; retain coalescing for the much larger
        // immutable-base read path below.
        if (write && platform::valid(overlay_fd_)) {
            for (const NvmeIoSpan &span : spans) {
                if (span.buffer_offset + span.bytes > buffer_bytes) {
                    throw std::runtime_error(
                        "NVMe batch span exceeds buffer");
                }
                write_slot_range(span.slot, /*slot_byte_offset=*/0,
                                 base + span.buffer_offset, span.bytes);
                if (stats) {
                    stats->bytes += span.bytes;
                    ++stats->syscalls;
                }
            }
            return;
        }
        size_t i = 0;
        while (i < spans.size()) {
            const NvmeIoSpan &first = spans[i];
            validate_slot_io(first.slot, base + first.buffer_offset,
                             first.bytes, write ? "write" : "read");
            if (first.buffer_offset + first.bytes > buffer_bytes) {
                throw std::runtime_error("NVMe batch span exceeds buffer");
            }
            const uint64_t first_file_offset = slot_offset(first.slot);
            const platform::FileHandle fd = write
                ? write_fd()
                : read_fd_for_range(
                      first.slot, base + first.buffer_offset,
                      first.bytes, first_file_offset);
            uint64_t merged_bytes = first.bytes;
            size_t j = i + 1;
            while (j < spans.size()) {
                const NvmeIoSpan &prev = spans[j - 1];
                const NvmeIoSpan &next = spans[j];
                const bool contiguous_file =
                    next.slot == prev.slot + 1 &&
                    prev.bytes == cfg_.slot_bytes;
                const bool contiguous_buffer =
                    next.buffer_offset ==
                    spans[i].buffer_offset + merged_bytes;
                // Adjacent slots can sit in different files once an overlay is
                // in play, and merging across that boundary would read the
                // wrong bytes for one of them.
                const uint64_t next_file_offset = slot_offset(next.slot);
                const bool same_file =
                    write ||
                    read_fd_for_range(
                        next.slot, base + next.buffer_offset,
                        next.bytes, next_file_offset) == fd;
                if (!contiguous_file || !contiguous_buffer || !same_file) break;
                validate_slot_io(next.slot, base + next.buffer_offset,
                                 next.bytes, write ? "write" : "read");
                if (next.buffer_offset + next.bytes > buffer_bytes) {
                    throw std::runtime_error(
                        "NVMe batch span exceeds buffer");
                }
                merged_bytes += next.bytes;
                ++j;
            }
            if (write) {
                pwrite_all(fd, base + first.buffer_offset, merged_bytes,
                           slot_offset(first.slot));
            } else {
                pread_all(fd, base + first.buffer_offset, merged_bytes,
                          slot_offset(first.slot));
            }
            if (cfg_.drop_page_cache && fd != direct_fd_) {
                const uint64_t file_offset = slot_offset(first.slot);
                const bool dropped =
                    drop_cached_range(fd, file_offset, merged_bytes, write);
                if (stats) {
                    if (dropped) {
                        stats->cache_drop_bytes += merged_bytes;
                    } else {
                        ++stats->cache_drop_failures;
                    }
                }
            }
            if (stats) {
                stats->bytes += merged_bytes;
                ++stats->syscalls;
            }
            i = j;
        }
    }

    bool direct_range_eligible(const void *data, uint64_t bytes,
                               uint64_t offset) const {
        if (!platform::valid(direct_fd_) || !data || bytes == 0) return false;
        const uintptr_t address = reinterpret_cast<uintptr_t>(data);
        return address % kDirectAlignment == 0 &&
               bytes % kDirectAlignment == 0 &&
               offset % kDirectAlignment == 0;
    }

    platform::FileHandle read_fd_for_range(int32_t slot, const void *data,
                                           uint64_t bytes,
                                           uint64_t offset) const {
        const platform::FileHandle ordinary = read_fd(slot);
        // Overlay records must always use their own buffered descriptor. The
        // immutable base can use O_DIRECT only for aligned transfer slabs.
        if (ordinary == fd_ && direct_range_eligible(data, bytes, offset)) {
            return direct_fd_;
        }
        return ordinary;
    }

    bool drop_cached_range(platform::FileHandle fd, uint64_t offset,
                           uint64_t bytes, bool write) const {
        if (bytes == 0) return true;
        if (!platform::drop_cached_range(fd, offset, bytes, write)) {
            warn_cache_drop_failure("page-cache-drop", platform::last_error());
            return false;
        }
        return true;
    }

    void warn_cache_drop_failure(const char *phase,
                                 const std::string &error) const {
        bool expected = false;
        if (!cache_drop_warned_.compare_exchange_strong(expected, true)) {
            return;
        }
        std::fprintf(
            stderr,
            "[kvmem-io] page_cache_drop_degraded=1 phase=%s message=%s "
            "action=continue-with-kernel-page-cache\n",
            phase, error.c_str());
    }

    void touch_locked(uint32_t block_id) {
        // A direct-mapped arena never evicts, so maintaining recency would only
        // pay the linear erase below once per access.
        if (cfg_.direct_mapped) return;
        erase_lru_locked(block_id);
        lru_.push_back(block_id);
    }

    void erase_lru_locked(uint32_t block_id) {
        if (cfg_.direct_mapped) return;
        for (auto it = lru_.begin(); it != lru_.end(); ++it) {
            if (*it == block_id) {
                lru_.erase(it);
                return;
            }
        }
    }

    NvmeKvTierConfig cfg_;
    static constexpr uint64_t kDirectAlignment = 4096;
    uint32_t slot_count_ = 0;
    std::string path_;
    platform::FileHandle fd_ = platform::invalid_handle();
    platform::FileHandle direct_fd_ = platform::invalid_handle();
    platform::FileHandle overlay_fd_ = platform::invalid_handle();
    static constexpr uint8_t kOverlayBase = 0;
    static constexpr uint8_t kOverlayInitializing = 1;
    static constexpr uint8_t kOverlayValid = 2;
    std::unique_ptr<std::atomic<uint8_t>[]> overlay_valid_;
    mutable std::mutex meta_mu_;
    mutable std::atomic<bool> cache_drop_warned_{false};
    std::vector<int32_t> free_slots_;
    std::unordered_map<uint32_t, int32_t> block_to_slot_;
    std::vector<uint32_t> lru_;
};

#else
// Keep the memory-only runtime's interface intact without importing POSIX
// headers. Storage operations fail rather than silently discarding KV data.
class NvmeKvTier {
public:
    explicit NvmeKvTier(const NvmeKvTierConfig & cfg) {
        if (cfg.total_bytes) unavailable();
    }
    bool enabled() const { return false; }
    uint32_t slot_count() const { return 0; }
    uint64_t slot_bytes() const { return 0; }
    uint32_t free_slots() const { return 0; }
    uint32_t used_slots() const { return 0; }
    int32_t block_slot(uint32_t) const { return -1; }
    int32_t lru_victim() const { return -1; }
    void release_block(uint32_t) {}
    void clear() {}
    void touch(uint32_t) {}
    NvmeSlotPlacement place_block(uint32_t) { unavailable(); }
    NvmeSlotPlacement place_block_evicting(uint32_t) { unavailable(); }
    void write_block(uint32_t, const void *, uint64_t) { unavailable(); }
    void read_block(uint32_t, void *, uint64_t) { unavailable(); }
    void write_slot(int32_t, const void *, uint64_t) const { unavailable(); }
    void read_slot(int32_t, void *, uint64_t) const { unavailable(); }
    void write_spans(const std::vector<NvmeIoSpan> &, const void *, uint64_t,
                     NvmeBatchIoStats * = nullptr) const { unavailable(); }
    void read_spans(const std::vector<NvmeIoSpan> &, void *, uint64_t,
                    NvmeBatchIoStats * = nullptr) const { unavailable(); }
private:
    [[noreturn]] static void unavailable() {
        throw std::runtime_error("NVMe offload is disabled in this build (KVMEM_ENABLE_NVME=OFF)");
    }
};
#endif

} // namespace kvmem
