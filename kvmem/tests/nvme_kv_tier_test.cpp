// NvmeKvTier host-logic test. Covers slot sizing, block residency, read/write,
// explicit LRU eviction, release/reuse, and crash-safe backing-file cleanup.

#include "kvmem/nvme_kv_tier.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

using namespace kvmem;

static int g_fail = 0;
#define CHECK(cond) do {                                                   \
    if (!(cond)) {                                                         \
        std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

static std::string temp_dir() {
    // std::filesystem honors TMPDIR/TEMP on both platforms.
    return (std::filesystem::temp_directory_path() /
            "qw3_nvme_kv_tier_test").u8string();
}

// "Directory entry is gone" check that is correct on both platforms.
// POSIX: after unlink, stat reports ENOENT -> exists() == false.
// Windows: after a POSIX-semantics delete (FileDispositionInfoEx), name
// lookups report ACCESS_DENIED while the last handle is still open, and
// NOT_FOUND after close; both mean the entry cannot survive the process.
// Only those expected status errors count as "gone"; anything else must
// fail loudly instead of masquerading as a successful delete.
static bool name_visible(const std::string & p) {
    std::error_code ec;
    const bool e = std::filesystem::exists(p, ec);
    if (!ec) {
        return e;
    }
#ifdef _WIN32
    // ERROR_FILE_NOT_FOUND(2) / ERROR_PATH_NOT_FOUND(3): gone.
    // ERROR_ACCESS_DENIED(5): delete-pending, name being reclaimed.
    const int v = ec.value();
    return !(v == 2 || v == 3 || v == 5);
#else
    // POSIX: a deleted name yields ENOENT with NO error_code; any error here
    // is a real failure (permissions, ENOTDIR) and must fail the CHECK
    // instead of masquerading as "deleted".
    return true;
#endif
}

static void test_disabled() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 0;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);
    CHECK(!t.enabled());
    CHECK(t.slot_count() == 0);
}

static void test_write_read_release() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 256;
    cfg.slot_bytes = 64;
    cfg.drop_page_cache = true;
    NvmeKvTier t(cfg);
    CHECK(t.enabled());
    CHECK(t.drops_page_cache());
    CHECK(t.slot_count() == 4);
    // The open descriptor remains usable, but the cache has no directory
    // entry and therefore cannot survive process exit as a stale large file.
    CHECK(!name_visible(t.path()));

    std::vector<uint8_t> a(64), b(64), out(64);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<uint8_t>(i);
        b[i] = static_cast<uint8_t>(255 - i);
    }

    t.write_block(10, a.data(), a.size());
    CHECK(t.block_slot(10) == 0);
    t.read_block(10, out.data(), out.size());
    CHECK(out == a);

    t.write_block(10, b.data(), b.size());
    t.read_block(10, out.data(), out.size());
    CHECK(out == b);

    t.release_block(10);
    CHECK(t.block_slot(10) == -1);
    CHECK(t.free_slots() == 4);
}

static void test_slot_ranges_and_file_names() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.file_name = "qw3_raw_k_range_test.bin";
    cfg.total_bytes = 256;
    cfg.slot_bytes = 128;
    NvmeKvTier t(cfg);
    CHECK(t.enabled());
    CHECK(t.path().find(cfg.file_name) != std::string::npos);
    CHECK(!name_visible(t.path()));

    const auto p = t.place_block(7);
    CHECK(p.slot == 0);
    std::vector<uint8_t> main(80, 0x31);
    std::vector<uint8_t> mtp(24, 0x92);
    std::vector<uint8_t> main_out(main.size(), 0);
    std::vector<uint8_t> mtp_out(mtp.size(), 0);
    t.write_slot_range(p.slot, 0, main.data(), main.size());
    t.write_slot_range(p.slot, 96, mtp.data(), mtp.size());
    t.read_slot_range(p.slot, 0, main_out.data(), main_out.size());
    t.read_slot_range(p.slot, 96, mtp_out.data(), mtp_out.size());
    CHECK(main_out == main);
    CHECK(mtp_out == mtp);

    bool rejected = false;
    try {
        t.read_slot_range(p.slot, 120, mtp_out.data(), mtp_out.size());
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    CHECK(rejected);
}

static void test_evicting_place() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 128;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);
    std::vector<uint8_t> a(64, 1), b(64, 2), c(64, 3), out(64);
    t.write_block(1, a.data(), a.size());
    t.write_block(2, b.data(), b.size());
    t.touch(1);  // block 2 becomes LRU.

    auto p = t.place_block_evicting(3);
    CHECK(p.slot == 1);
    CHECK(p.evicted_block == 2);
    CHECK(t.block_slot(2) == -1);
    CHECK(t.block_slot(3) == 1);

    t.write_block(3, c.data(), c.size());
    t.read_block(3, out.data(), out.size());
    CHECK(out == c);
}

static void test_coalesced_batch_io() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 64 * 8;
    cfg.slot_bytes = 64;
    cfg.drop_page_cache = true;
    NvmeKvTier t(cfg);
    CHECK(t.drops_page_cache());

    std::vector<uint8_t> input(64 * 3), output(64 * 3, 0);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint8_t>((i * 17) & 0xff);
    }
    std::vector<NvmeIoSpan> spans;
    for (uint32_t block = 0; block < 3; ++block) {
        const auto p = t.place_block(100 + block);
        CHECK(p.slot == static_cast<int32_t>(block));
        spans.push_back(NvmeIoSpan{
            p.slot, static_cast<uint64_t>(block) * 64, 64});
    }

    NvmeBatchIoStats writes;
    t.write_spans(spans, input.data(), input.size(), &writes);
    CHECK(writes.bytes == input.size());
    CHECK(writes.syscalls == 1);
    CHECK(writes.cache_drop_bytes == input.size());
    CHECK(writes.cache_drop_failures == 0);

    NvmeBatchIoStats reads;
    t.read_spans(spans, output.data(), output.size(), &reads);
    CHECK(reads.bytes == output.size());
    CHECK(reads.syscalls == 1);
    CHECK(reads.cache_drop_bytes == output.size());
    CHECK(reads.cache_drop_failures == 0);
    CHECK(output == input);
}

static void test_concurrent_positional_batches() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 64 * 8;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);

    std::vector<uint8_t> a(64 * 3, 0x35);
    std::vector<uint8_t> b(64 * 3, 0xca);
    std::vector<uint8_t> output(64 * 6, 0);
    std::vector<NvmeIoSpan> a_spans;
    std::vector<NvmeIoSpan> b_spans;
    for (uint32_t block = 0; block < 6; ++block) {
        const auto p = t.place_block(200 + block);
        CHECK(p.slot == static_cast<int32_t>(block));
        auto &spans = block < 3 ? a_spans : b_spans;
        spans.push_back(NvmeIoSpan{
            p.slot, static_cast<uint64_t>(block % 3) * 64, 64});
    }

    auto aw = std::async(std::launch::async, [&]() {
        t.write_spans(a_spans, a.data(), a.size());
    });
    auto bw = std::async(std::launch::async, [&]() {
        t.write_spans(b_spans, b.data(), b.size());
    });
    aw.get();
    bw.get();

    std::vector<NvmeIoSpan> all;
    for (uint32_t slot = 0; slot < 6; ++slot) {
        all.push_back(NvmeIoSpan{
            static_cast<int32_t>(slot),
            static_cast<uint64_t>(slot) * 64, 64});
    }
    NvmeBatchIoStats reads;
    t.read_spans(all, output.data(), output.size(), &reads);
    CHECK(reads.syscalls == 1);
    CHECK(std::equal(a.begin(), a.end(), output.begin()));
    CHECK(std::equal(
        b.begin(), b.end(), output.begin() +
            static_cast<std::ptrdiff_t>(a.size())));
}

static void test_durable_archive_roundtrip() {
    // Covers the Win32 branches no ephemeral test reaches: OPEN_EXISTING
    // (read-only reopen), OPEN_ALWAYS-free durable create (no delete-on-close
    // so the NAME must survive), and direct-mapped identity slots.
    const std::string dir = temp_dir() + "_durable";
    const std::string name = "qw3_archive.bin";
    std::vector<uint8_t> a(64, 0x77), out(64, 0);
    {
        NvmeKvTierConfig cfg;
        cfg.dir = dir;
        cfg.file_name = name;
        cfg.total_bytes = 256;
        cfg.slot_bytes = 64;
        cfg.durable = true;
        cfg.direct_mapped = true;
        NvmeKvTier t(cfg);
        CHECK(t.enabled());
        t.write_block(3, a.data(), a.size());  // identity slot 3
    }
    const std::string path = dir + "/" + name;
    CHECK(name_visible(path));  // durable: the archive survives close
    {
        NvmeKvTierConfig ro;
        ro.dir = dir;
        ro.file_name = name;
        ro.total_bytes = 256;
        ro.slot_bytes = 64;
        ro.durable = true;
        ro.read_only = true;
        ro.direct_mapped = true;
        NvmeKvTier t(ro);
        CHECK(t.enabled());
        CHECK(t.read_only());
        t.mark_present_range(0, 4);
        t.read_block(3, out.data(), out.size());
        CHECK(out == a);
        bool rejected = false;
        try {
            t.write_block(1, a.data(), a.size());
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        CHECK(rejected);  // read-only attach refuses writes
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

static void test_preallocate_sets_size() {
    // Covers platform::fallocate (FileAllocationInfo on Win32): the file must
    // really carry total_bytes after a preallocated durable create.
    const std::string dir = temp_dir() + "_prealloc";
    {
        NvmeKvTierConfig cfg;
        cfg.dir = dir;
        cfg.file_name = "qw3_prealloc.bin";
        cfg.total_bytes = 1024 * 1024;
        cfg.slot_bytes = 64;
        cfg.durable = true;
        cfg.preallocate = true;
        NvmeKvTier t(cfg);
        CHECK(t.enabled());
    }
    std::error_code ec;
    const auto sz = std::filesystem::file_size(
        std::filesystem::path(dir) / "qw3_prealloc.bin", ec);
    CHECK(!ec);
    CHECK(sz == 1024 * 1024);
    std::filesystem::remove_all(dir, ec);
}

static void test_direct_read_aligned() {
    // Covers the FILE_FLAG_NO_BUFFERING descriptor: durable+read-only+
    // direct_read with a 4096-aligned buffer/offset/size must succeed and
    // return identical bytes.
    const std::string dir = temp_dir() + "_direct";
    const std::string name = "qw3_direct.bin";
    constexpr uint64_t kSlot = 4096;
    std::vector<uint8_t> a(static_cast<size_t>(kSlot), 0),
        out(static_cast<size_t>(kSlot), 0);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<uint8_t>(i * 31);
    }
    {
        NvmeKvTierConfig cfg;
        cfg.dir = dir;
        cfg.file_name = name;
        cfg.total_bytes = kSlot * 4;
        cfg.slot_bytes = kSlot;
        cfg.durable = true;
        cfg.direct_mapped = true;
        NvmeKvTier t(cfg);
        t.write_block(1, a.data(), a.size());
    }
    std::vector<uint8_t> backing(static_cast<size_t>(kSlot) + 4096, 0);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(backing.data());
    uint8_t * aligned =
        backing.data() + ((4096 - addr % 4096) % 4096);
    {
        NvmeKvTierConfig ro;
        ro.dir = dir;
        ro.file_name = name;
        ro.total_bytes = kSlot * 4;
        ro.slot_bytes = kSlot;
        ro.durable = true;
        ro.read_only = true;
        ro.direct_read = true;
        ro.direct_mapped = true;
        NvmeKvTier t(ro);
        if (!t.direct_reads()) {
            // tmpfs/overlayfs (a common /tmp on Linux) and similar refuse
            // O_DIRECT; the tier degrades to buffered reads. The roundtrip
            // below still validates the data path on both.
            std::printf("note: O_DIRECT unavailable on this filesystem; "
                        "direct-read assertion skipped\n");
        }
        t.mark_present_range(0, 4);
        t.read_block(1, aligned, kSlot);
        CHECK(std::memcmp(aligned, a.data(), static_cast<size_t>(kSlot)) == 0);
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

static void test_readonly_overlay_cow() {
    // Covers open_overlay (second ephemeral POSIX-deleted handle) and the
    // copy-up write path: writes divert to the overlay, the base archive
    // stays byte-identical.
    const std::string dir = temp_dir() + "_ovlbase";
    const std::string odir = temp_dir() + "_ovldelta";
    const std::string name = "qw3_ovl.bin";
    std::vector<uint8_t> a(64, 0x5a), b(64, 0xa5), out(64, 0);
    {
        NvmeKvTierConfig cfg;
        cfg.dir = dir;
        cfg.file_name = name;
        cfg.total_bytes = 256;
        cfg.slot_bytes = 64;
        cfg.durable = true;
        cfg.direct_mapped = true;
        NvmeKvTier t(cfg);
        t.write_block(2, a.data(), a.size());
    }
    {
        NvmeKvTierConfig ro;
        ro.dir = dir;
        ro.file_name = name;
        ro.total_bytes = 256;
        ro.slot_bytes = 64;
        ro.durable = true;
        ro.read_only = true;
        ro.direct_mapped = true;
        ro.overlay_dir = odir;
        NvmeKvTier t(ro);
        CHECK(t.enabled());
        CHECK(t.has_overlay());
        CHECK(!t.read_only());  // writable through the overlay
        t.mark_present_range(0, 4);
        t.read_block(2, out.data(), out.size());
        CHECK(out == a);  // base read-through
        t.write_block(2, b.data(), b.size());  // copy-up into overlay
        std::fill(out.begin(), out.end(), 0);
        t.read_block(2, out.data(), out.size());
        CHECK(out == b);
    }
    {
        // Base archive untouched: reopen without the overlay.
        NvmeKvTierConfig ro;
        ro.dir = dir;
        ro.file_name = name;
        ro.total_bytes = 256;
        ro.slot_bytes = 64;
        ro.durable = true;
        ro.read_only = true;
        ro.direct_mapped = true;
        NvmeKvTier t(ro);
        t.mark_present_range(0, 4);
        t.read_block(2, out.data(), out.size());
        CHECK(out == a);
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::remove_all(odir, ec);
}

static void test_non_ascii_paths() {
    // Regression sentinel for the to_wide encoding contract: accept both
    // UTF-8 bytes (u8string producers) and ACP bytes (path::string()/argv
    // defaults on Windows). Non-ASCII is spelled as explicit byte escapes so
    // this source file stays pure ASCII (no /utf-8 flag dependency).
    //   "_\xCF\x80"       = UTF-8 for GREEK SMALL LETTER PI
    //   "_\xD6\xD0\xCE\xC4" = GBK/ACP bytes (invalid UTF-8 -> ACP fallback)
    for (const char * suffix : {"_\xCF\x80", "_\xD6\xD0\xCE\xC4"}) {
        const std::string dir = temp_dir() + suffix;
        NvmeKvTierConfig cfg;
        cfg.dir = dir;
        cfg.file_name = std::string("arena") + suffix + ".bin";
        cfg.total_bytes = 256;
        cfg.slot_bytes = 64;
        NvmeKvTier t(cfg);
        CHECK(t.enabled());
        std::vector<uint8_t> a(64, 0x3c), out(64, 0);
        t.write_block(5, a.data(), a.size());
        t.read_block(5, out.data(), out.size());
        CHECK(out == a);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }
}

int main() {
    // Unbuffered so CHECK failures survive a later crash, and exceptions are
    // reported instead of terminating with an opaque 0xC0000409.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        test_disabled();
        test_write_read_release();
        test_slot_ranges_and_file_names();
        test_evicting_place();
        test_coalesced_batch_io();
        test_concurrent_positional_batches();
        test_durable_archive_roundtrip();
        test_preallocate_sets_size();
        test_direct_read_aligned();
        test_readonly_overlay_cow();
        test_non_ascii_paths();
    } catch (const std::exception & e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
