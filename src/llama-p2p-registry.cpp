#include "llama-p2p-registry.h"
#include "llama-impl.h"

#ifdef LLAMA_USE_XNVME

#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <libxnvme.h>

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct pending_registration {
    // vector of futures - one per dev - each carrying an error code.
    std::vector<std::future<int>> per_dev;
    size_t size = 0;
};

struct p2p_registry {
    std::mutex                            mu;
    std::string                           uris_csv;
    std::vector<std::string>              uris;
    std::vector<xnvme_dev *>              devs;
    std::unordered_map<void *,
                       pending_registration> pending;   // vaddr -> in-flight maps
    std::unordered_map<void *,
                       std::vector<xnvme_dev *>> live;  // vaddr -> devs mapped
    bool                                  initialised = false;
    bool                                  shutting_down = false;
};

static p2p_registry & registry() {
    static p2p_registry r;
    return r;
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t p = s.find(',', start);
        size_t len = (p == std::string::npos ? s.size() : p) - start;
        std::string tok = s.substr(start, len);
        size_t a = 0, b = tok.size();
        while (a < b && std::isspace(static_cast<unsigned char>(tok[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(tok[b - 1]))) --b;
        if (b > a) out.push_back(tok.substr(a, b - a));
        if (p == std::string::npos) break;
        start = p + 1;
    }
    return out;
}

// The alloc hook must not block: it just kicks off async mem_map on each
// dev and stashes the futures so a later barrier can wait for them. The
// futures are joined either at llama_p2p_registry_barrier() (before load)
// or at buffer-free time (before unmap).
static void alloc_hook_impl(void * /*user_data*/, int device, void * base, size_t size) {
    p2p_registry & r = registry();
    std::lock_guard<std::mutex> g(r.mu);
    if (!r.initialised || r.shutting_down) return;

    // Skip if we've already seen this buffer (idempotent alloc hook is polite).
    if (r.pending.find(base) != r.pending.end() || r.live.find(base) != r.live.end()) {
        return;
    }

    // Register the buffer exactly as ggml allocated it. Rounding the size up
    // to a page, which older xNVMe needed because it required page-aligned
    // vaddr and nbytes, now asks to register past the end of the allocation:
    // CUDA reports an allocation's size as exactly what was requested, ggml
    // aligns buffer sizes to 128 bytes, and xNVMe refuses a range that leaves
    // the allocation it recovered. Current xNVMe recovers the containing
    // allocation itself, so any size is fine.
    const size_t aligned_size = size;

    pending_registration pr;
    pr.size = aligned_size;
    pr.per_dev.reserve(r.devs.size());

    // Snapshot the devs vector for the launched threads. Threads run to
    // completion even if the registry is later torn down; shutdown blocks
    // for their completion before unmapping.
    std::vector<xnvme_dev *> devs_snapshot = r.devs;
    // Fire ONE background thread that iterates all devs sequentially.
    //
    // The heavy cost of xnvme_mem_map used to be cuMemGetHandleForAddressRange
    // + dmabuf_attach per 2 MiB chunk. xNVMe now exports once per allocation
    // instead, because ROCm returns the whole buffer object regardless of the
    // range asked for and a per-chunk export therefore resolves sub-ranges
    // wrongly, so a buffer of any size costs one export rather than one per
    // 2 MiB. Subsequent devs still amortise to ~zero: the registry is process
    // global, so registering the same base for another dev only bumps a
    // refcount. That also means the per-dev loop is now redundant rather than
    // cheap, and could be dropped once older xNVMe no longer needs supporting.
    //
    // These calls land in the same process-global registry, so several of
    // these threads registering at once race unless xNVMe serialises it. It
    // does, from the release that introduced the per-allocation export.
    pr.per_dev.emplace_back(std::async(std::launch::async,
        [devs_snapshot, base, aligned_size, device]() -> int {
            cudaError_t cerr = cudaSetDevice(device);
            if (cerr != cudaSuccess) {
                LLAMA_LOG_WARN("llama_p2p_registry alloc hook: cudaSetDevice(%d) failed: %s\n",
                               device, cudaGetErrorString(cerr));
                return -EIO;
            }
            int worst = 0;
            for (xnvme_dev * dev : devs_snapshot) {
                int rc = xnvme_mem_map(dev, base, aligned_size);
                if (rc) {
                    LLAMA_LOG_WARN("llama_p2p_registry alloc hook: xnvme_mem_map(base=%p, size=%zu) failed: rc=%d\n",
                                   base, aligned_size, rc);
                    worst = rc;
                }
            }
            return worst;
        }));
    r.pending.emplace(base, std::move(pr));
    r.live.emplace(base, std::move(devs_snapshot));
}

static void free_hook_impl(void * /*user_data*/, int /*device*/, void * base) {
    p2p_registry & r = registry();

    // We need to be careful with locking: xnvme_mem_unmap can be slow, so we
    // don't want to hold the mutex the whole time. Extract state under lock,
    // then unmap outside.
    std::vector<std::future<int>> pending_futures;
    std::vector<xnvme_dev *>      devs_for_this_buf;
    {
        std::lock_guard<std::mutex> g(r.mu);
        if (!r.initialised) return;
        auto pit = r.pending.find(base);
        if (pit != r.pending.end()) {
            pending_futures = std::move(pit->second.per_dev);
            r.pending.erase(pit);
        }
        auto lit = r.live.find(base);
        if (lit != r.live.end()) {
            devs_for_this_buf = std::move(lit->second);
            r.live.erase(lit);
        }
    }

    // Wait for any in-flight registrations to finish before unmapping.
    for (auto & f : pending_futures) {
        (void) f.get();
    }
    for (xnvme_dev * dev : devs_for_this_buf) {
        xnvme_mem_unmap(dev, base);
    }
}

}  // namespace

int llama_p2p_registry_init(const std::string & uris_csv) {
    p2p_registry & r = registry();
    std::lock_guard<std::mutex> g(r.mu);

    if (r.initialised) {
        if (r.uris_csv == uris_csv) return 0;
        LLAMA_LOG_WARN("%s: registry already initialised with a different URI list\n", __func__);
        return -EINVAL;
    }

    std::vector<std::string> uris = split_csv(uris_csv);
    if (uris.empty()) {
        LLAMA_LOG_WARN("%s: empty URI list; nothing to register\n", __func__);
        return -EINVAL;
    }

    xnvme_opts opts{};
    // Name the backend only. async/sync/admin are implementation ids, a
    // different namespace: the upcie-cuda config shares the plain upcie
    // command path, whose id is "upcie". Pinning them to the backend name
    // used to match by coincidence, when each flavour carried its own, and
    // now filters every config out so dev_open returns -ENXIO.
    opts.be     = "upcie-cuda";
    opts.nsid   = 1;
    opts.rdonly = 1;

    std::vector<xnvme_dev *> devs;
    devs.reserve(uris.size());
    for (const auto & uri : uris) {
        xnvme_dev * dev = xnvme_dev_open(uri.c_str(), &opts);
        if (!dev) {
            LLAMA_LOG_WARN("%s: xnvme_dev_open('%s', be=upcie-cuda) failed (errno=%d); "
                           "leaving registry uninitialised\n",
                           __func__, uri.c_str(), errno);
            for (xnvme_dev * d : devs) xnvme_dev_close(d);
            return -EIO;
        }
        int drc = xnvme_dev_derive_geo(dev);
        if (drc) {
            LLAMA_LOG_WARN("%s: xnvme_dev_derive_geo('%s') failed: %d\n", __func__, uri.c_str(), drc);
            xnvme_dev_close(dev);
            for (xnvme_dev * d : devs) xnvme_dev_close(d);
            return -EIO;
        }
        devs.push_back(dev);
    }

    r.uris_csv = uris_csv;
    r.uris     = std::move(uris);
    r.devs     = std::move(devs);
    r.initialised = true;

    ggml_backend_cuda_set_buffer_hooks(alloc_hook_impl, free_hook_impl, nullptr);

    // Guarantee mem_unmap + dev_close run on normal process exit even if
    // nobody calls llama_p2p_registry_shutdown() explicitly. Without this,
    // an abnormal termination (SIGABRT from a downstream CUDA error, panic,
    // or unhandled exception) leaves cuMemGetHandleForAddressRange-backed
    // dma-buf attachments pinned in the kernel driver, holding VRAM without
    // any user-space owner - a state the driver only clears on reboot or
    // module reload. Registered once via a first-time-only flag so
    // repeated re-inits don't stack callbacks.
    static bool atexit_installed = false;
    if (!atexit_installed) {
        std::atexit([]() { llama_p2p_registry_shutdown(); });
        atexit_installed = true;
    }

    LLAMA_LOG_WARN("%s: opened %zu upcie-cuda dev(s); ggml-cuda buffer hook installed\n",
                   __func__, r.devs.size());
    return 0;
}

const std::vector<xnvme_dev *> * llama_p2p_registry_devs() {
    p2p_registry & r = registry();
    std::lock_guard<std::mutex> g(r.mu);
    if (!r.initialised) return nullptr;
    return &r.devs;
}

void llama_p2p_registry_barrier() {
    p2p_registry & r = registry();
    // Move pending futures out of the map under the lock, then wait outside.
    std::vector<std::pair<void *, pending_registration>> drain;
    {
        std::lock_guard<std::mutex> g(r.mu);
        drain.reserve(r.pending.size());
        for (auto & kv : r.pending) {
            drain.emplace_back(kv.first, std::move(kv.second));
        }
        r.pending.clear();
    }
    for (auto & entry : drain) {
        for (auto & f : entry.second.per_dev) {
            (void) f.get();
        }
    }
}

void llama_p2p_registry_shutdown() {
    p2p_registry & r = registry();
    std::vector<std::pair<void *, pending_registration>>       drain;
    std::vector<std::pair<void *, std::vector<xnvme_dev *>>>   live;
    std::vector<xnvme_dev *>                                   devs;
    {
        std::lock_guard<std::mutex> g(r.mu);
        if (!r.initialised) return;
        r.shutting_down = true;
        drain.reserve(r.pending.size());
        for (auto & kv : r.pending) drain.emplace_back(kv.first, std::move(kv.second));
        r.pending.clear();
        live.reserve(r.live.size());
        for (auto & kv : r.live) live.emplace_back(kv.first, std::move(kv.second));
        r.live.clear();
        devs = std::move(r.devs);
        r.initialised = false;
    }

    ggml_backend_cuda_set_buffer_hooks(nullptr, nullptr, nullptr);

    // Wait for anything still registering. We MUST NOT release pending
    // futures until they've resolved, otherwise the mem_map thread would
    // touch freed state during unwind.
    for (auto & entry : drain) {
        for (auto & f : entry.second.per_dev) (void) f.get();
    }

    // Fast-shutdown path: the polite per-buffer per-dev xnvme_mem_unmap +
    // xnvme_dev_close walk hits the same driver-lock the mem_map path did,
    // so tearing down ~4k dma-buf attachments serially adds several seconds
    // of wall-clock we pay AFTER the model has already loaded (visible as
    // the delta between load_time and total wallclock). On process exit
    // the OS + driver clean up the CUDA context on their own, so we can
    // skip the polite tearoff entirely without leaking anything past
    // process death. Set LLAMA_XNVME_FAST_SHUTDOWN=0 to opt into the polite
    // path for debugging.
    const char * fast_env = std::getenv("LLAMA_XNVME_FAST_SHUTDOWN");
    const bool fast_shutdown = !fast_env || std::atoi(fast_env) != 0;
    if (fast_shutdown) {
        LLAMA_LOG_WARN("%s: fast-shutdown: skipping %zu mem_unmap + %zu dev_close calls "
                       "(driver reclaims on process exit)\n",
                       __func__, live.size(), devs.size());
        return;
    }

    for (auto & entry : live) {
        for (xnvme_dev * dev : entry.second) {
            xnvme_mem_unmap(dev, entry.first);
        }
    }
    for (xnvme_dev * dev : devs) {
        xnvme_dev_close(dev);
    }
}

#else  // LLAMA_USE_XNVME undefined - compile-out stubs

int  llama_p2p_registry_init(const std::string &) { return 0; }
void llama_p2p_registry_barrier()                 { }
void llama_p2p_registry_shutdown()                { }

#endif
