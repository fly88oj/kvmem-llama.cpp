// Platform file I/O primitives for NvmeKvTier (declared in nvme_kv_tier.hpp).
//
// Positional and thread-safe on both platforms: stage-in reads and stage-out
// writes may run concurrently on different workers, exactly like pread/pwrite.
//   - POSIX: pread/pwrite with EINTR retry.
//   - Win32: FILE_FLAG_OVERLAPPED handles with the position carried in each
//     call's OVERLAPPED structure. The shared file pointer is never touched,
//     so concurrent positional I/O needs no locking.
//
// Ephemeral arenas (open-then-unlink):
//   - POSIX: ::unlink right after ::open.
//   - Win32: FILE_SHARE_DELETE at open, then FileDispositionInfoEx with
//     POSIX semantics so the directory entry disappears immediately while the
//     data survives until the last handle closes (including abnormal exit).
//     Pre-Win10-1607 falls back to DeleteFileW (name lingers until close).
//
// Keeping windows.h confined to this TU matters: nvme_kv_tier.hpp is included
// by HIP/clang translation units through kvmem_runtime.hpp, and windows.h
// macros would leak into them.

#include "kvmem/nvme_kv_tier.hpp"

#if KVMEM_ENABLE_NVME

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace kvmem {
namespace platform {

#ifdef _WIN32
// ------------------------------------------------------------------- Win32 --

FileHandle invalid_handle() { return INVALID_HANDLE_VALUE; }

bool valid(FileHandle h) { return h != INVALID_HANDLE_VALUE; }

static std::wstring to_wide(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    const int len = static_cast<int>(s.size());
    // Strict UTF-8 probe first: MB_ERR_INVALID_CHARS makes the call fail on
    // any non-UTF-8 byte. Producers on Windows hand us ACP bytes by default
    // (std::filesystem::path::string() uses the native narrow encoding and
    // argv is ACP too), so fall back to CP_ACP for those.
    UINT cp = CP_UTF8;
    int n = ::MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, s.c_str(), len, nullptr, 0);
    if (n <= 0) {
        cp = CP_ACP;
        n = ::MultiByteToWideChar(cp, 0, s.c_str(), len, nullptr, 0);
        if (n <= 0) {
            return std::wstring();
        }
    }
    std::wstring w(static_cast<size_t>(n), L'\0');
    ::MultiByteToWideChar(cp, 0, s.c_str(), len, &w[0], n);
    return w;
}

std::string last_error() {
    const unsigned long e = static_cast<unsigned long>(::GetLastError());
    char * msg = nullptr;
    const DWORD n = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, e, 0, reinterpret_cast<char *>(&msg), 0, nullptr);
    std::string text = (n != 0 && msg) ? std::string(msg, n) : "unknown";
    if (msg) {
        ::LocalFree(msg);
    }
    while (!text.empty() &&
           (text.back() == '\r' || text.back() == '\n' || text.back() == ' ')) {
        text.pop_back();
    }
    return "win32 error " + std::to_string(e) + ": " + text;
}

FileHandle open_file(const std::string & path, bool read, bool write,
                     bool create, bool truncate, bool delete_on_close,
                     bool no_buffering) {
    const std::wstring wpath = to_wide(path);
    DWORD access = 0;
    if (read) access |= GENERIC_READ;
    if (write) access |= GENERIC_WRITE;
    // DELETE access is required for SetFileInformationByHandle(FileDispositionInfoEx).
    if (delete_on_close) access |= DELETE;
    DWORD creation;
    if (!create) {
        creation = OPEN_EXISTING;
    } else if (truncate) {
        creation = CREATE_ALWAYS;
    } else {
        creation = OPEN_ALWAYS;
    }
    DWORD flags = FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED;
    if (no_buffering) flags |= FILE_FLAG_NO_BUFFERING;
    // POSIX open() has no mandatory sharing: any number of descriptors may
    // read/write the same file concurrently. Mirror that with the full share
    // mode so a second NvmeKvTier/RawKvStore can reopen a path whose earlier
    // handle is still alive (the tier serializes via slot metadata and
    // positional I/O, not via the OS). FILE_SHARE_DELETE also lets the
    // ephemeral delete below succeed while this handle stays open.
    HANDLE h = ::CreateFileW(
        wpath.c_str(), access,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        creation, flags, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return invalid_handle();
    }
    if (delete_on_close) {
        // POSIX unlink semantics: the name disappears now, the data is
        // reclaimed when the last handle closes (or the process dies).
        // FILE_DISPOSITION_INFO_EX: deletion is the DEFAULT action;
        // FILE_DISPOSITION_FLAG_DO_NOT_DISCARD (0x4) opts out, and
        // FILE_DISPOSITION_FLAG_POSIX_SEMANTICS (0x1) removes the directory
        // entry immediately instead of at last-handle close.
        struct { DWORD Flags; } disposition;
        static constexpr DWORD kPosixSemantics = 0x00000001;
        disposition.Flags = kPosixSemantics;
        BOOL removed = ::SetFileInformationByHandle(
            h, FileDispositionInfoEx, &disposition, sizeof(disposition));
        if (!removed) {
            // Pre-Win10-1607 fallback: classic delete. The name lingers
            // (delete-pending, ACCESS_DENIED to staters) until the last handle
            // closes, so log the degradation instead of failing silently.
            const unsigned long e =
                static_cast<unsigned long>(::GetLastError());
            removed = ::DeleteFileW(wpath.c_str());
            if (removed) {
                std::fprintf(
                    stderr,
                    "[kvmem-io] win32_posix_delete_degraded=1 error=%lu "
                    "action=classic-delete-on-close "
                    "hint=name-lingers-until-last-close;"
                    "concurrent-instances-must-use-distinct-files\n", e);
            }
        }
        if (!removed) {
            ::CloseHandle(h);
            return invalid_handle();
        }
    }
    return static_cast<FileHandle>(h);
}

void close_file(FileHandle h) {
    if (valid(h)) {
        ::CloseHandle(static_cast<HANDLE>(h));
    }
}

bool transfer_at(FileHandle h, void * buf, uint64_t bytes, uint64_t offset,
                 uint64_t * done, bool write) {
    *done = 0;
    // ReadFile/WriteFile take a DWORD count; the callers loop over partial
    // transfers, so clamping here is transparent.
    static constexpr uint64_t kMaxChunk = 0x40000000ull;  // 1 GiB
    if (bytes > kMaxChunk) {
        bytes = kMaxChunk;
    }
    OVERLAPPED ov {};
    ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFull);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    ov.hEvent = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!ov.hEvent) {
        return false;
    }
    DWORD n = 0;
    BOOL ok = write
        ? ::WriteFile(static_cast<HANDLE>(h), buf, static_cast<DWORD>(bytes),
                      &n, &ov)
        : ::ReadFile(static_cast<HANDLE>(h), buf, static_cast<DWORD>(bytes),
                     &n, &ov);
    DWORD failure = ERROR_SUCCESS;
    if (!ok) {
        failure = ::GetLastError();
        if (failure == ERROR_IO_PENDING) {
            ok = ::GetOverlappedResult(static_cast<HANDLE>(h), &ov, &n, TRUE);
            if (!ok) {
                failure = ::GetLastError();
            }
        } else if (!write && failure == ERROR_HANDLE_EOF) {
            ::CloseHandle(ov.hEvent);
            return true;  // EOF: *done stays 0, matching pread() == 0.
        }
    }
    ::CloseHandle(ov.hEvent);
    if (!ok) {
        // CloseHandle may clobber the last-error; restore the real I/O
        // failure so the caller's platform::last_error() message is accurate.
        ::SetLastError(failure == ERROR_SUCCESS ? ERROR_IO_INCOMPLETE
                                                : failure);
        return false;
    }
    *done = n;
    return true;
}

int fallocate(FileHandle h, uint64_t bytes) {
    // posix_fallocate does two things: reserves the disk blocks (up-front
    // ENOSPC) AND extends the file size. Windows needs both calls:
    // FileAllocationInfo reserves clusters without moving EOF, and
    // FileEndOfFileInfo sets the logical size.
    FILE_ALLOCATION_INFO alloc;
    alloc.AllocationSize.QuadPart = static_cast<LONGLONG>(bytes);
    if (!::SetFileInformationByHandle(static_cast<HANDLE>(h),
                                      FileAllocationInfo, &alloc,
                                      sizeof(alloc))) {
        if (::GetLastError() == ERROR_DISK_FULL) {
            return -1;  // hard failure, same as posix_fallocate ENOSPC
        }
        return 1;  // unsupported (non-NTFS etc): degrade to sparse growth
    }
    FILE_END_OF_FILE_INFO eof;
    eof.EndOfFile.QuadPart = static_cast<LONGLONG>(bytes);
    // Clusters are reserved at this point; failing to move EOF would leave
    // the file reserved-but-empty, so treat any second-step failure as hard
    // (the caller closes the descriptor and the space is reclaimed) rather
    // than reporting a misleading sparse-growth degrade.
    if (!::SetFileInformationByHandle(static_cast<HANDLE>(h),
                                      FileEndOfFileInfo, &eof,
                                      sizeof(eof))) {
        return -1;
    }
    return 0;
}

bool drop_cached_range(FileHandle h, uint64_t, uint64_t, bool write) {
    static std::once_flag announced;
    std::call_once(announced, [] {
        std::fprintf(stderr,
                     "[kvmem-io] page_cache_policy=win32-standby-reclaimable "
                     "range_eviction=unsupported\n");
    });
    if (write) {
        // Durability barrier equivalent to Linux sync_file_range(WAIT_AFTER):
        // surface ENOSPC synchronously to the caller instead of inside the
        // lazy writer after the RAM copy has already been released. Whole-
        // file flush; only paid when the tier runs with drop_page_cache.
        return ::FlushFileBuffers(static_cast<HANDLE>(h)) != 0;
    }
    // Reads leave clean pages on the standby list: OS-reclaimable under
    // memory pressure - unlike the pinned host arena this tier replaces.
    return true;
}

int ensure_dir(const std::string & dir) {
    const std::wstring wdir = to_wide(dir);
    const DWORD attrs = ::GetFileAttributesW(wdir.c_str());
    if (attrs != INVALID_FILE_ATTRIBUTES) {
        return (attrs & FILE_ATTRIBUTE_DIRECTORY) ? 0 : -1;
    }
    if (::CreateDirectoryW(wdir.c_str(), nullptr)) {
        return 0;
    }
    return ::GetLastError() == ERROR_ALREADY_EXISTS ? 0 : -2;
}

#else
// ------------------------------------------------------------------- POSIX --

FileHandle invalid_handle() { return -1; }

bool valid(FileHandle h) { return h >= 0; }

std::string last_error() { return std::strerror(errno); }

FileHandle open_file(const std::string & path, bool read, bool write,
                     bool create, bool truncate, bool delete_on_close,
                     bool no_buffering) {
    int flags = O_CLOEXEC;
    if (read && write) {
        flags |= O_RDWR;
    } else if (write) {
        flags |= O_WRONLY;
    } else {
        flags |= O_RDONLY;
    }
    if (create) flags |= O_CREAT;
    if (truncate) flags |= O_TRUNC;
#if defined(__linux__) && defined(O_DIRECT)
    if (no_buffering) flags |= O_DIRECT;
#else
    // Mirrors the historical linux-only O_DIRECT gate: unsupported platforms
    // return an invalid handle so callers take the buffered-fallback path.
    if (no_buffering) {
        errno = ENOTSUP;
        return invalid_handle();
    }
#endif
    int fd;
    do {
        fd = ::open(path.c_str(), flags, 0644);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        return invalid_handle();
    }
    if (delete_on_close && ::unlink(path.c_str()) != 0) {
        const int unlink_error = errno;
        ::close(fd);
        errno = unlink_error;
        return invalid_handle();
    }
    return fd;
}

void close_file(FileHandle h) {
    if (valid(h)) {
        ::close(h);
    }
}

bool transfer_at(FileHandle h, void * buf, uint64_t bytes, uint64_t offset,
                 uint64_t * done, bool write) {
    *done = 0;
    ssize_t n;
    do {
        n = write ? ::pwrite(h, buf, static_cast<size_t>(bytes),
                             static_cast<off_t>(offset))
                  : ::pread(h, buf, static_cast<size_t>(bytes),
                            static_cast<off_t>(offset));
    } while (n < 0 && errno == EINTR);
    if (n < 0) {
        return false;
    }
    *done = static_cast<uint64_t>(n);
    return true;
}

int fallocate(FileHandle h, uint64_t bytes) {
    if (bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        errno = EFBIG;
        return -1;
    }
    int rc;
    do {
        rc = ::posix_fallocate(h, 0, static_cast<off_t>(bytes));
    } while (rc == EINTR);
    if (rc == 0) {
        return 0;
    }
    if (rc == EOPNOTSUPP || rc == ENOSYS || rc == EINVAL) {
        return 1;  // unsupported: degrade to sparse growth
    }
    errno = rc;
    return -1;
}

bool drop_cached_range(FileHandle h, uint64_t offset, uint64_t bytes,
                       bool write) {
    if (write) {
#if defined(__linux__) && defined(SYNC_FILE_RANGE_WRITE) && \
    defined(SYNC_FILE_RANGE_WAIT_BEFORE) && defined(SYNC_FILE_RANGE_WAIT_AFTER)
        int rc;
        do {
            rc = ::sync_file_range(
                h, static_cast<off64_t>(offset), static_cast<off64_t>(bytes),
                SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE |
                    SYNC_FILE_RANGE_WAIT_AFTER);
        } while (rc != 0 && errno == EINTR);
        if (rc != 0) {
            return false;
        }
#else
        // Portable fallback flushes the whole file rather than one range, so
        // serialize concurrent batches to avoid redundant fdatasync storms.
        // The mutex is intentionally global (not per-instance): fdatasync is
        // a whole-file operation, and cross-instance serialization here only
        // affects non-Linux POSIX platforms.
        static std::mutex flush_mu;
        std::lock_guard<std::mutex> lock(flush_mu);
        int rc;
        do {
            rc = ::fdatasync(h);
        } while (rc != 0 && errno == EINTR);
        if (rc != 0) {
            return false;
        }
#endif
    }
#if defined(POSIX_FADV_DONTNEED)
    const int advise = ::posix_fadvise(h, static_cast<off_t>(offset),
                                       static_cast<off_t>(bytes),
                                       POSIX_FADV_DONTNEED);
    if (advise != 0) {
        // posix_fadvise RETURNS the error code instead of setting errno;
        // propagate it so platform::last_error() reports the real failure.
        errno = advise;
    }
    return advise == 0;
#else
    errno = ENOTSUP;
    return false;
#endif
}

int ensure_dir(const std::string & dir) {
    struct stat st {};
    if (::stat(dir.c_str(), &st) == 0) {
        return (st.st_mode & S_IFDIR) ? 0 : -1;
    }
    if (::mkdir(dir.c_str(), 0755) == 0) {
        return 0;
    }
    return errno == EEXIST ? 0 : -2;
}

#endif

}  // namespace platform
}  // namespace kvmem

#endif  // KVMEM_ENABLE_NVME
