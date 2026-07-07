#include "llama-loader-xnvme.h"
#include "llama-impl.h"

#ifdef LLAMA_USE_XNVME

#include <libxnvme.h>
#include <libxnvme_file.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <unistd.h>

namespace {

// Shared bookkeeping across all in-flight submissions.
struct load_state {
    ggml_backend_t        upload_backend = nullptr;
    std::atomic<uint64_t> outstanding{0};
    std::atomic<int>      err{0};
};

// Round-robin ring slot. host_buf carries a fixed io_size-byte slice of
// the shared pinned buffer; event is recorded on the upload_backend
// after each async H->D copy so the next reuse of this slot can wait.
// The chunk-metadata fields (tensor, tensor_off, skip, len, is_host)
// belong to the currently in-flight chunk on this slot and are safe to
// touch again only after needs_wait's event fires. Embedding the
// metadata in the slot means one heap object per slot for the whole
// load instead of one malloc per submit.
struct slot {
    uint8_t *            host_buf   = nullptr;
    ggml_backend_event_t event      = nullptr;
    bool                 needs_wait = false;
    // per-chunk metadata (set at submit time, read in the callback)
    struct ggml_tensor * tensor     = nullptr;
    size_t               tensor_off = 0;
    size_t               skip       = 0;
    size_t               len        = 0;
    bool                 is_host    = false;
    struct load_state *  state      = nullptr;
};

static void on_completion(struct xnvme_cmd_ctx * ctx, void * cb_arg) {
    auto * s  = static_cast<slot *>(cb_arg);
    auto * st = s->state;

    if (xnvme_cmd_ctx_cpl_status(ctx)) {
        st->err.store(EIO, std::memory_order_relaxed);
        s->needs_wait = false;
    } else if (s->is_host) {
        std::memcpy(static_cast<uint8_t *>(s->tensor->data) + s->tensor_off,
                    s->host_buf + s->skip, s->len);
        s->needs_wait = false;
    } else {
        ggml_backend_tensor_set_async(st->upload_backend, s->tensor,
                                      s->host_buf + s->skip,
                                      s->tensor_off, s->len);
        ggml_backend_event_record(s->event, st->upload_backend);
        s->needs_wait = true;
    }

    xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
    st->outstanding.fetch_sub(1, std::memory_order_release);
}

static uint32_t env_u32(const char * name, uint32_t def) {
    const char * v = std::getenv(name);
    if (!v || !*v) return def;
    char * end = nullptr;
    unsigned long r = std::strtoul(v, &end, 0);
    if (end == v) return def;
    return static_cast<uint32_t>(r);
}

static inline uint64_t align_down(uint64_t x, uint64_t a) { return x & ~(a - 1); }
static inline uint64_t align_up  (uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }

// Map opts.be to opts.async. For file-backed xnvme configs the two are
// bound together, so callers only need to name the backend.
static const char * async_for_be(const std::string & be) {
    if (be == "libaio_file")  return "libaio";
    if (be == "thrpool_file") return "thrpool";
    if (be == "posix_file")   return "posix";
    if (be == "emu_file")     return "emu";
    // io_uring_file, io_uring_bdev, and anything else default to io_uring.
    return "io_uring";
}

}  // namespace

bool llama_loader_xnvme_run(
    const std::vector<llama_loader_xnvme_job> & jobs,
    const llama_files & files,
    ggml_backend_t upload_backend,
    ggml_backend_buffer_type_t host_buft,
    bool use_direct_io,
    const std::string & xnvme_be,
    size_t & size_done,
    size_t   size_data,
    llama_progress_callback progress_cb,
    void   * progress_ud)
{
    if (!upload_backend || !host_buft) {
        // No usable device backend / pinned host buft; caller should fall back.
        return false;
    }

    const uint32_t qd      = std::max<uint32_t>(env_u32("LLAMA_XNVME_QD",      64), 2);
    const uint32_t io_size = std::max<uint32_t>(env_u32("LLAMA_XNVME_IO_SIZE", 128 * 1024), 4096);
    const uint32_t n_slots = qd;

    // Optionally pin the submitter thread to a dedicated CPU core so the
    // xnvme queue poker + cudaMemcpyAsync submitter isn't preempted by
    // llama.cpp's other threads. xnvmeperf shows a single-queue single-CPU
    // io_uring setup hits the drive line rate when it owns its core.
    // Restored at exit so we don't leak affinity into the caller.
    cpu_set_t prev_affinity;
    bool have_prev_affinity = false;
    const char * cpu_env = std::getenv("LLAMA_XNVME_CPU");
    if (cpu_env && cpu_env[0]) {
        int cpu = std::atoi(cpu_env);
        if (cpu >= 0) {
            CPU_ZERO(&prev_affinity);
            if (sched_getaffinity(0, sizeof(prev_affinity), &prev_affinity) == 0) {
                have_prev_affinity = true;
                cpu_set_t mask;
                CPU_ZERO(&mask);
                CPU_SET(cpu, &mask);
                if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
                    LLAMA_LOG_WARN("%s: sched_setaffinity(cpu=%d) failed: %s\n",
                                   __func__, cpu, std::strerror(errno));
                    have_prev_affinity = false;
                }
            }
        }
    }

    const std::string be = xnvme_be.empty() ? std::string("io_uring_file") : xnvme_be;

    if (be == "io_uring_cmd") {
        // Passthrough backend needs opening /dev/ngXnY plus FIEMAP extent
        // translation; not implemented in this integration. Fall back so
        // the caller can retry via STREAM (or pick a file-backed backend).
        LLAMA_LOG_WARN("%s: xnvme backend '%s' requires the passthrough integration; falling back to STREAM\n",
                       __func__, be.c_str());
        return false;
    }

    // Open one xnvme_dev per source file, all using the same backend.
    std::vector<struct xnvme_dev *>   devs(files.size(), nullptr);
    std::vector<struct xnvme_queue *> queues(files.size(), nullptr);

    auto teardown = [&](bool warn_setup) {
        for (auto * q : queues) if (q) xnvme_queue_term(q);
        for (auto * d : devs)   if (d) xnvme_dev_close(d);
        if (have_prev_affinity) {
            sched_setaffinity(0, sizeof(prev_affinity), &prev_affinity);
            have_prev_affinity = false;
        }
        if (warn_setup) {
            LLAMA_LOG_WARN("%s: xnvme setup failed (be='%s'); falling back to STREAM\n",
                           __func__, be.c_str());
        }
    };

    for (size_t i = 0; i < files.size(); ++i) {
        const std::string & fname = files[i]->path();
        if (fname.empty() || fname == "(file*)") {
            teardown(true);
            return false;
        }
        xnvme_opts opts{};
        opts.be     = be.c_str();
        opts.async  = async_for_be(be);
        opts.rdonly = 1;
        opts.direct = use_direct_io ? 1 : 0;

        devs[i] = xnvme_file_open(fname.c_str(), &opts);
        if (!devs[i]) {
            teardown(true);
            return false;
        }
        int rc = xnvme_queue_init(devs[i], static_cast<uint16_t>(qd), 0, &queues[i]);
        if (rc) {
            teardown(true);
            return false;
        }
    }

    // One pinned host buffer for the whole ring, sliced n_slots ways.
    ggml_backend_buffer_t slot_buffer =
        ggml_backend_buft_alloc_buffer(host_buft, static_cast<size_t>(n_slots) * io_size);
    if (!slot_buffer) {
        teardown(true);
        return false;
    }

    uint8_t * slot_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(slot_buffer));

    ggml_backend_dev_t dev = ggml_backend_get_device(upload_backend);
    std::vector<slot> slots(n_slots);
    for (uint32_t i = 0; i < n_slots; ++i) {
        slots[i].host_buf = slot_base + static_cast<size_t>(i) * io_size;
        slots[i].event    = ggml_backend_event_new(dev);
    }

    load_state st;
    st.upload_backend = upload_backend;

    bool cancelled = false;
    uint32_t next_slot = 0;

    for (const auto & job : jobs) {
        if (cancelled || st.err.load(std::memory_order_relaxed) != 0) break;

        struct xnvme_queue * q = queues[job.file_idx];

        size_t align = 1;
        if (use_direct_io && files[job.file_idx]->has_direct_io()) {
            align = files[job.file_idx]->read_alignment();
            if (align == 0) align = 1;
        }
        if (align > io_size) {
            teardown(false);
            LLAMA_LOG_WARN("%s: io_size (%u) is smaller than the file's alignment (%zu); falling back to STREAM\n",
                           __func__, io_size, align);
            for (auto & s : slots) if (s.event) ggml_backend_event_free(s.event);
            ggml_backend_buffer_free(slot_buffer);
            return false;
        }

        size_t remaining = job.n_size;
        size_t src_off   = job.offs;
        size_t dst_off   = 0;

        while (remaining) {
            uint64_t file_read_start = align_down(src_off, align);
            size_t   head_skip       = src_off - file_read_start;

            size_t want_ceiling = io_size - head_skip;
            if (align > 1 && want_ceiling > (align - 1)) {
                want_ceiling -= (align - 1);
            }
            size_t want = std::min<size_t>(remaining, want_ceiling);
            if (want == 0) {
                st.err.store(EINVAL, std::memory_order_relaxed);
                break;
            }

            uint64_t file_read_end = align_up(src_off + want, align);
            uint64_t read_len      = file_read_end - file_read_start;
            if (read_len > io_size) {
                read_len = io_size;
                file_read_end = file_read_start + read_len;
                want = (file_read_end > src_off) ? (file_read_end - src_off) : 0;
                if (want > remaining) want = remaining;
            }

            slot & s = slots[next_slot];
            next_slot = (next_slot + 1) % n_slots;

            if (s.needs_wait) {
                ggml_backend_event_synchronize(s.event);
                s.needs_wait = false;
            }
            // Only poke when we actually need capacity. Submits between
            // pokes accumulate in the io_uring SQ so the first flush to
            // the kernel lands as a batch of ~qd requests, giving the
            // drive real queue pressure to fan out.
            while (st.outstanding.load(std::memory_order_acquire) >= qd) {
                xnvme_queue_poke(q, 0);
            }

            struct xnvme_cmd_ctx * ctx = xnvme_cmd_ctx_from_queue(q);
            while (!ctx) {
                xnvme_queue_poke(q, 0);
                ctx = xnvme_cmd_ctx_from_queue(q);
            }

            s.tensor     = job.tensor;
            s.tensor_off = dst_off;
            s.skip       = head_skip;
            s.len        = want;
            s.is_host    = job.is_host;
            s.state      = &st;

            ctx->async.cb     = on_completion;
            ctx->async.cb_arg = &s;

            int rc;
            do {
                rc = xnvme_file_pread(ctx, s.host_buf, static_cast<size_t>(read_len),
                                      static_cast<off_t>(file_read_start));
                if (rc == -EBUSY) xnvme_queue_poke(q, 0);
            } while (rc == -EBUSY);

            if (rc) {
                xnvme_queue_put_cmd_ctx(q, ctx);
                st.err.store(rc < 0 ? -rc : rc, std::memory_order_relaxed);
                break;
            }

            st.outstanding.fetch_add(1, std::memory_order_release);

            src_off   += want;
            dst_off   += want;
            remaining -= want;
        }

        size_done += job.n_size;
        if (progress_cb && !progress_cb(
                static_cast<float>(size_done) / static_cast<float>(size_data),
                progress_ud)) {
            cancelled = true;
        }
    }

    for (auto * q : queues) {
        if (!q) continue;
        int rc = xnvme_queue_drain(q);
        if (rc < 0) {
            LLAMA_LOG_WARN("%s: xnvme_queue_drain returned %d\n", __func__, rc);
        }
    }
    for (auto & s : slots) {
        if (s.needs_wait) {
            ggml_backend_event_synchronize(s.event);
            s.needs_wait = false;
        }
    }

    bool ok = st.err.load(std::memory_order_relaxed) == 0;
    if (!ok) {
        LLAMA_LOG_WARN("%s: one or more submissions failed (errno %d); loader returning error\n",
                       __func__, st.err.load(std::memory_order_relaxed));
    }

    for (auto & s : slots) {
        if (s.event) ggml_backend_event_free(s.event);
    }
    ggml_backend_buffer_free(slot_buffer);
    for (auto * q : queues) if (q) xnvme_queue_term(q);
    for (auto * d : devs)   if (d) xnvme_dev_close(d);

    if (have_prev_affinity) {
        sched_setaffinity(0, sizeof(prev_affinity), &prev_affinity);
    }

    return ok;
}

#else  // LLAMA_USE_XNVME undefined -> compiled-out fallback

bool llama_loader_xnvme_run(
    const std::vector<llama_loader_xnvme_job> &,
    const llama_files &,
    ggml_backend_t,
    ggml_backend_buffer_type_t,
    bool,
    const std::string &,
    size_t &,
    size_t,
    llama_progress_callback,
    void *)
{
    LLAMA_LOG_WARN("%s: llama.cpp was built without LLAMA_XNVME; falling back to STREAM\n", __func__);
    return false;
}

#endif  // LLAMA_USE_XNVME
