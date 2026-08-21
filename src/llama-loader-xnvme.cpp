#include "llama-loader-xnvme.h"
#include "llama-impl.h"
#include "llama-p2p-registry.h"

#ifdef LLAMA_USE_XNVME

#include <libxnvme.h>
#include <libxnvme_file.h>
#include <libxnvme_nvm.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sched.h>
#include <string>
#include <unistd.h>
#include <vector>

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

// Forward decl for the upcie-cuda P2P path (see below).
static bool llama_loader_xnvme_run_upcie_cuda(
    const std::vector<llama_loader_xnvme_job> & jobs,
    const llama_files & files,
    ggml_backend_t upload_backend,
    ggml_backend_buffer_type_t host_buft,
    const std::string & xnvme_be,
    size_t & size_done,
    size_t   size_data,
    llama_progress_callback progress_cb,
    void   * progress_ud);

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

    // NVMe passthrough P2P path: xnvme opens the raw NVMe device via upcie,
    // xnvme_buf_alloc returns VRAM, xnvme_nvm_read DMAs directly into VRAM.
    // No host bounce; per byte crosses PCIe once instead of twice. Requires
    // the target drive to be pre-bound to VFIO with the model raw-dd'd to
    // LBA 0 (see cijoe/scripts/setup_dma_target.py).
    if (xnvme_be == "upcie-cuda") {
        return llama_loader_xnvme_run_upcie_cuda(
            jobs, files, upload_backend, host_buft,
            xnvme_be, size_done, size_data,
            progress_cb, progress_ud);
    }

    const uint32_t qd      = std::max<uint32_t>(env_u32("LLAMA_XNVME_QD",      64), 2);
    const uint32_t io_size = std::max<uint32_t>(env_u32("LLAMA_XNVME_IO_SIZE", 512 * 1024), 4096);
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

    // Compute per-file alignment now so we can size the pinned host pool
    // to the largest requirement and reject setups where io_size is too
    // small for any file.
    std::vector<size_t> file_align(files.size(), 1);
    for (size_t i = 0; i < files.size(); ++i) {
        size_t align = 1;
        if (use_direct_io && files[i]->has_direct_io()) {
            align = files[i]->read_alignment();
            if (align == 0) align = 1;
        }
        if (align > io_size) {
            teardown(false);
            LLAMA_LOG_WARN("%s: io_size (%u) is smaller than file %zu's alignment (%zu); falling back to STREAM\n",
                           __func__, io_size, i, align);
            return false;
        }
        file_align[i] = align;
    }

    // Per-file slot ring: each file's queue gets its own qd slots so
    // rotating across files doesn't clobber an in-flight read's buffer
    // just because we visited a different file's queue. Total pinned
    // host buffer grows to n_files * n_slots * io_size.
    const size_t total_slots = files.size() * n_slots;
    ggml_backend_buffer_t slot_buffer =
        ggml_backend_buft_alloc_buffer(host_buft, total_slots * io_size);
    if (!slot_buffer) {
        teardown(true);
        return false;
    }

    uint8_t * slot_base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(slot_buffer));
    ggml_backend_dev_t dev = ggml_backend_get_device(upload_backend);

    load_state st;
    st.upload_backend = upload_backend;

    // Per-file cursor holds its queue, slot ring, outstanding gate, and the
    // subset of jobs that read from this file. The rotate loop below walks
    // all cursors in order, poking each queue and refilling from that
    // file's cursor until either the queue is full or the file is done.
    // For n_files==1 the topology degenerates to the pre-refactor
    // single-cursor behaviour.
    struct file_cursor {
        struct xnvme_queue *      queue = nullptr;
        std::vector<slot>         slots;
        uint32_t                  next_slot = 0;
        size_t                    align = 1;
        std::atomic<uint64_t>     outstanding{0};
        std::vector<size_t>       job_indices;
        size_t                    job_cursor    = 0;
        size_t                    src_off_cur   = 0;
        size_t                    dst_off_cur   = 0;
        size_t                    remaining_cur = 0;
        file_cursor() = default;
        file_cursor(const file_cursor &) = delete;
        file_cursor & operator=(const file_cursor &) = delete;
    };
    std::vector<std::unique_ptr<file_cursor>> cursors;
    cursors.reserve(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        auto c = std::unique_ptr<file_cursor>(new file_cursor);
        c->queue = queues[i];
        c->align = file_align[i];
        c->slots.resize(n_slots);
        for (uint32_t k = 0; k < n_slots; ++k) {
            const size_t global_slot = i * n_slots + k;
            c->slots[k].host_buf = slot_base + global_slot * io_size;
            c->slots[k].event    = ggml_backend_event_new(dev);
        }
        cursors.push_back(std::move(c));
    }

    // Partition jobs by file_idx onto per-file cursors. Preserves each
    // file's disk-order (jobs upstream are sorted (file_idx, offs)).
    for (size_t j = 0; j < jobs.size(); ++j) {
        const uint16_t fi = jobs[j].file_idx;
        if (fi >= cursors.size()) {
            LLAMA_LOG_WARN("%s: job %zu has file_idx=%u beyond files.size()=%zu\n",
                           __func__, j, (unsigned) fi, cursors.size());
            for (auto & c : cursors) {
                for (auto & s : c->slots) if (s.event) ggml_backend_event_free(s.event);
            }
            ggml_backend_buffer_free(slot_buffer);
            teardown(true);
            return false;
        }
        cursors[fi]->job_indices.push_back(j);
    }

    // Free the per-file dev/queue arrays -- ownership has moved into
    // cursors' non-owning pointers, and the cursors' unique_ptr lifetime
    // spans the rest of this function.
    (void) devs;  // devs still owned locally, released at the end.

    // Submit one chunk for cursor `c`. Returns true iff a chunk was
    // submitted (or the cursor is done and there is nothing to do);
    // false when the queue has no free ctx or a free slot right now.
    auto submit_one = [&](file_cursor & c) -> bool {
        // Advance to the next job in this file if the current is done.
        while (c.remaining_cur == 0) {
            if (c.job_cursor >= c.job_indices.size()) return false;
            const auto & job = jobs[c.job_indices[c.job_cursor]];
            c.src_off_cur   = job.offs;
            c.dst_off_cur   = 0;
            c.remaining_cur = job.n_size;
        }
        const auto & job = jobs[c.job_indices[c.job_cursor]];

        uint64_t file_read_start = align_down(c.src_off_cur, c.align);
        size_t   head_skip       = c.src_off_cur - file_read_start;
        size_t   want_ceiling    = io_size - head_skip;
        if (c.align > 1 && want_ceiling > (c.align - 1)) {
            want_ceiling -= (c.align - 1);
        }
        size_t want = std::min<size_t>(c.remaining_cur, want_ceiling);
        if (want == 0) {
            st.err.store(EINVAL, std::memory_order_relaxed);
            return false;
        }
        uint64_t file_read_end = align_up(c.src_off_cur + want, c.align);
        uint64_t read_len      = file_read_end - file_read_start;
        if (read_len > io_size) {
            read_len = io_size;
            file_read_end = file_read_start + read_len;
            want = (file_read_end > c.src_off_cur) ? (file_read_end - c.src_off_cur) : 0;
            if (want > c.remaining_cur) want = c.remaining_cur;
        }

        slot & s = c.slots[c.next_slot];
        if (s.needs_wait) {
            ggml_backend_event_synchronize(s.event);
            s.needs_wait = false;
        }
        c.next_slot = (c.next_slot + 1) % c.slots.size();

        struct xnvme_cmd_ctx * ctx = xnvme_cmd_ctx_from_queue(c.queue);
        if (!ctx) return false;

        s.tensor     = job.tensor;
        s.tensor_off = c.dst_off_cur;
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
            if (rc == -EBUSY) xnvme_queue_poke(c.queue, 0);
        } while (rc == -EBUSY);
        if (rc) {
            xnvme_queue_put_cmd_ctx(c.queue, ctx);
            st.err.store(rc < 0 ? -rc : rc, std::memory_order_relaxed);
            return false;
        }

        c.outstanding.fetch_add(1, std::memory_order_release);
        st.outstanding.fetch_add(1, std::memory_order_release);
        c.src_off_cur   += want;
        c.dst_off_cur   += want;
        c.remaining_cur -= want;

        if (c.remaining_cur == 0) {
            size_done += job.n_size;
            c.job_cursor++;
        }
        return true;
    };

    // Initial fill: for each file cursor, submit until its queue is full
    // or its assigned partition is exhausted.
    bool cancelled = false;
    for (auto & c : cursors) {
        while (submit_one(*c)) { /* keep filling */ }
        if (st.err.load(std::memory_order_relaxed) != 0) { cancelled = true; break; }
    }
    if (progress_cb && !progress_cb(
            static_cast<float>(size_done) / static_cast<float>(size_data),
            progress_ud)) {
        cancelled = true;
    }

    // Rotate loop: poke each queue, refill from tail; exit when every
    // cursor's partition is exhausted and no I/O remains outstanding.
    // The completion callback fires from xnvme_queue_poke and decrements
    // st.outstanding (the shared gate) but not the per-cursor counter -
    // that's fine because we exit on (all-cursors-done && shared-zero),
    // and the per-cursor counter only informs the picker on retry.
    while (!cancelled && st.err.load(std::memory_order_relaxed) == 0) {
        bool any_pending = false;
        for (auto & c : cursors) {
            xnvme_queue_poke(c->queue, 0);
            while (submit_one(*c)) { /* keep filling */ }
            if (c->job_cursor < c->job_indices.size()) any_pending = true;
        }
        const bool any_outstanding =
            st.outstanding.load(std::memory_order_acquire) > 0;
        if (!any_pending && !any_outstanding) break;
        if (progress_cb && !progress_cb(
                static_cast<float>(size_done) / static_cast<float>(size_data),
                progress_ud)) {
            cancelled = true;
        }
    }

    // Drain all queues, then retire any in-flight H->D copies.
    for (auto & c : cursors) {
        int rc = xnvme_queue_drain(c->queue);
        if (rc < 0) {
            LLAMA_LOG_WARN("%s: xnvme_queue_drain returned %d\n", __func__, rc);
        }
    }
    for (auto & c : cursors) {
        for (auto & s : c->slots) {
            if (s.needs_wait) {
                ggml_backend_event_synchronize(s.event);
                s.needs_wait = false;
            }
        }
    }

    bool ok = st.err.load(std::memory_order_relaxed) == 0;
    if (!ok) {
        LLAMA_LOG_WARN("%s: one or more submissions failed (errno %d); loader returning error\n",
                       __func__, st.err.load(std::memory_order_relaxed));
    }

    for (auto & c : cursors) {
        for (auto & s : c->slots) {
            if (s.event) ggml_backend_event_free(s.event);
        }
    }
    ggml_backend_buffer_free(slot_buffer);
    for (auto * q : queues) if (q) xnvme_queue_term(q);
    for (auto * d : devs)   if (d) xnvme_dev_close(d);

    if (have_prev_affinity) {
        sched_setaffinity(0, sizeof(prev_affinity), &prev_affinity);
    }

    return ok;
}

// ============================================================================
// upcie-cuda P2P path
// ============================================================================
//
// Opens a raw NVMe device via xNVMe's userspace upcie backend, allocates the
// ring's host-buffer equivalents as VRAM (xnvme_buf_alloc with opts.be=
// "upcie-cuda" returns cudaMalloc'd pointers that are DMA-mappable), submits
// xnvme_nvm_read with LBA-based addressing, and moves data from slot VRAM to
// the tensor's device buffer via cudaMemcpyAsync(DeviceToDevice). Each byte
// crosses PCIe once (SSD -> GPU) instead of twice (SSD -> host -> GPU).
//
// Requirements the caller must have satisfied before invocation:
//   - The DMA target drive has been raw-dd'd with the .gguf so its LBAs
//     starting at 0 match the file's byte offsets (or an extent map is
//     provided; MVP assumes the raw layout).
//   - The DMA target drive is bound to VFIO (xnvme-driver / spdk-driver).
//   - LLAMA_XNVME_DMA_URI is set to the PCIe URI (e.g. "0000:01:00.0").
//
// The 'files' argument is only used to open the source file for metadata;
// llama.cpp has already done that by the time load_all_data runs.

namespace {

// One FIEMAP-derived extent. lba is the starting LBA (in units of the
// drive's lba_nbytes), file_offset is the byte offset within the .gguf
// where the extent begins, length is the extent size in bytes. Sorted
// by file_offset so a binary search finds the extent covering a target
// offset.
struct dma_extent {
    uint64_t file_offset;
    uint64_t lba;
    uint64_t length;
};

struct upcie_drive;
struct upcie_state;

// Ring slot for the P2P path. dma_buf is VRAM allocated via xnvme_buf_alloc
// (which routes to cudaMalloc under upcie-cuda). The completion callback
// fires cudaMemcpyAsync(DeviceToDevice) from dma_buf into the tensor buffer.
// `drive` back-points to the owner so the callback can decrement its own
// outstanding counter.
//
// Slot lifecycle:
//   FREE         : in_flight_read=false, needs_wait=false. Never used or fully
//                  drained. Safe to submit into.
//   READ_ACTIVE  : in_flight_read=true, needs_wait=false. NVMe is writing to
//                  dma_buf; not safe to reuse until the callback fires.
//   D2D_QUEUED   : in_flight_read=false (cleared by callback), needs_wait=true.
//                  NVMe read is done, D2D from dma_buf into tensor is enqueued.
//                  Safe to reuse after cudaEventSynchronize on the event.
// The picker in submit_one scans for FREE or D2D_QUEUED slots (syncing the
// event in the latter case) and refuses to reuse a READ_ACTIVE slot — that
// would clobber an in-flight DMA.
struct upcie_slot {
    void *               dma_buf         = nullptr; // scratch VRAM buffer (io_size bytes)
    void *               direct_target   = nullptr; // when non-null, NVMe writes directly here and no D2D is needed
    cudaEvent_t          event           = nullptr;
    bool                 in_flight_read  = false;
    bool                 needs_wait      = false;
    struct ggml_tensor * tensor          = nullptr;
    size_t               tensor_off      = 0;
    size_t               skip            = 0;
    size_t               len             = 0;
    struct upcie_drive * drive           = nullptr;
    struct upcie_state * state           = nullptr;
};

struct upcie_state {
    cudaStream_t          stream = nullptr;
    std::atomic<int>      err{0};
    std::atomic<uint64_t> direct_reads{0};
    std::atomic<uint64_t> scratch_reads{0};
    std::atomic<uint64_t> direct_bytes{0};
    std::atomic<uint64_t> scratch_bytes{0};
};

// Per-drive state. Each drive owns one xnvme_dev + one xnvme_queue + its
// own slot ring + its own extents map + its own tensor-job partition.
// The main thread rotates through drives, poking each and refilling its
// queue - mirrors xnvmeperf.c:453-478. Cursor fields track where we are
// inside the drive's assigned job partition.
struct upcie_drive {
    std::string           uri;
    struct xnvme_dev *    dev        = nullptr;
    struct xnvme_queue *  queue      = nullptr;
    std::vector<upcie_slot> slots;
    std::vector<dma_extent> extents;
    uint32_t              nsid       = 0;
    uint32_t              lba_nbytes = 0;
    uint32_t              mdts_nlb   = 0;
    uint32_t              next_slot  = 0;
    std::atomic<uint64_t> outstanding{0};

    // Job partition assigned to this drive (indices into the main jobs vector).
    std::vector<size_t>   job_indices;
    size_t                job_cursor    = 0;   // next un-submitted job in job_indices
    size_t                src_off_cur   = 0;   // in-progress cursor within current job
    size_t                dst_off_cur   = 0;
    size_t                remaining_cur = 0;

    // ggml backend buffers registered via xnvme_mem_map so NVMe can DMA
    // directly into tensor VRAM instead of bouncing through a scratch slot.
    // Owned pointers; each must be xnvme_mem_unmap'd on teardown.
    std::vector<void *>   mapped_buffers;

    // Per-drive submission counters, for balancing the fan-out.
    std::atomic<uint64_t> subs_count{0};
    std::atomic<uint64_t> subs_bytes{0};

    // Constructor and assignment operators must be deleted because
    // std::atomic is not copyable / movable.
    upcie_drive() = default;
    upcie_drive(const upcie_drive &) = delete;
    upcie_drive & operator=(const upcie_drive &) = delete;
};

// Minimal JSON reader for the schema emitted by
// cijoe/scripts/setup_dma_target.py. Only understands the exact shape
// used here (a top-level "file_size" integer plus an "extents" array of
// objects with "file_offset", "lba", "length" integers). Robust enough
// that reordering keys or adding unrelated top-level fields is fine;
// intentionally not a general JSON parser.
static bool parse_extents_json(const char * path,
                               std::vector<dma_extent> & out,
                               uint64_t & file_size)
{
    FILE * f = std::fopen(path, "r");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    if (n <= 0) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_SET);
    std::string buf(static_cast<size_t>(n), '\0');
    size_t got = std::fread(&buf[0], 1, static_cast<size_t>(n), f);
    std::fclose(f);
    if (got != static_cast<size_t>(n)) return false;

    // Return the offset just past the numeric literal that follows the
    // colon after a "<key>" occurrence at or after `from`. Fills `v`.
    auto find_u64_from = [&](const std::string & key, size_t from, uint64_t & v) -> size_t {
        std::string needle = "\"" + key + "\"";
        size_t p = buf.find(needle, from);
        if (p == std::string::npos) return std::string::npos;
        p = buf.find(':', p);
        if (p == std::string::npos) return std::string::npos;
        p++;
        char * end = nullptr;
        unsigned long long parsed = std::strtoull(buf.c_str() + p, &end, 10);
        if (end == buf.c_str() + p) return std::string::npos;
        v = static_cast<uint64_t>(parsed);
        return static_cast<size_t>(end - buf.c_str());
    };

    file_size = 0;
    find_u64_from("file_size", 0, file_size);

    size_t ext_p = buf.find("\"extents\"");
    if (ext_p == std::string::npos) return false;
    size_t arr_start = buf.find('[', ext_p);
    if (arr_start == std::string::npos) return false;
    size_t arr_end   = std::string::npos;
    int    depth    = 0;
    for (size_t i = arr_start; i < buf.size(); ++i) {
        if (buf[i] == '[') depth++;
        else if (buf[i] == ']') { if (--depth == 0) { arr_end = i; break; } }
    }
    if (arr_end == std::string::npos) return false;

    size_t p = arr_start + 1;
    while (p < arr_end) {
        size_t obj_start = buf.find('{', p);
        if (obj_start == std::string::npos || obj_start >= arr_end) break;
        size_t obj_end = buf.find('}', obj_start);
        if (obj_end == std::string::npos || obj_end > arr_end) break;
        // Scope key search to the current object body.
        auto find_in_obj = [&](const std::string & key, uint64_t & v) -> bool {
            std::string needle = "\"" + key + "\"";
            size_t k = buf.find(needle, obj_start);
            if (k == std::string::npos || k > obj_end) return false;
            k = buf.find(':', k);
            if (k == std::string::npos || k > obj_end) return false;
            k++;
            char * end = nullptr;
            unsigned long long parsed = std::strtoull(buf.c_str() + k, &end, 10);
            if (end == buf.c_str() + k) return false;
            v = static_cast<uint64_t>(parsed);
            return true;
        };
        dma_extent e{};
        if (find_in_obj("file_offset", e.file_offset) &&
            find_in_obj("lba",         e.lba) &&
            find_in_obj("length",      e.length)) {
            out.push_back(e);
        }
        p = obj_end + 1;
    }

    std::sort(out.begin(), out.end(),
              [](const dma_extent & a, const dma_extent & b) {
                  return a.file_offset < b.file_offset;
              });
    return !out.empty();
}

// Locate the extent that contains `file_off`. Returns nullptr if none.
// Extents are sorted by file_offset and treated as non-overlapping.
static const dma_extent * find_extent(const std::vector<dma_extent> & exts,
                                      uint64_t file_off)
{
    auto it = std::upper_bound(exts.begin(), exts.end(), file_off,
                               [](uint64_t v, const dma_extent & e) {
                                   return v < e.file_offset;
                               });
    if (it == exts.begin()) return nullptr;
    --it;
    if (file_off >= it->file_offset && file_off < it->file_offset + it->length) {
        return &(*it);
    }
    return nullptr;
}

static void on_completion_upcie(struct xnvme_cmd_ctx * ctx, void * cb_arg) {
    auto * s  = static_cast<upcie_slot *>(cb_arg);
    auto * d  = s->drive;
    auto * st = s->state;

    // NVMe read has landed - regardless of success. Clear in_flight_read
    // so the picker knows this slot's buffer is no longer being written by
    // the SSD.
    s->in_flight_read = false;

    if (xnvme_cmd_ctx_cpl_status(ctx)) {
        st->err.store(EIO, std::memory_order_relaxed);
    } else if (s->direct_target) {
        // Direct path: NVMe wrote straight into the tensor's VRAM (registered
        // via xnvme_mem_map). No D2D bounce, no event needed - slot is
        // immediately reusable.
        s->needs_wait = false;
    } else {
        // Scratch path: NVMe wrote into the slot's scratch VRAM. Copy just
        // the payload bytes into the tensor's VRAM. cudaMemcpyAsync
        // DeviceToDevice runs at HBM bandwidth (~800 GB/s), so this is only
        // used for edge reads (unaligned head, non-LBA-multiple tail).
        cudaMemcpyAsync(static_cast<uint8_t *>(s->tensor->data) + s->tensor_off,
                        static_cast<uint8_t *>(s->dma_buf) + s->skip,
                        s->len, cudaMemcpyDeviceToDevice, st->stream);
        cudaEventRecord(s->event, st->stream);
        s->needs_wait = true;
    }

    xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
    d->outstanding.fetch_sub(1, std::memory_order_release);
}

// Split a comma-separated env value into pieces. Empty entries are dropped.
static std::vector<std::string> parse_csv_env(const char * name) {
    std::vector<std::string> out;
    const char * v = std::getenv(name);
    if (!v || !*v) return out;
    std::string s(v);
    size_t start = 0;
    while (start <= s.size()) {
        size_t p = s.find(',', start);
        size_t len = (p == std::string::npos ? s.size() : p) - start;
        std::string tok = s.substr(start, len);
        // trim whitespace
        size_t a = 0, b = tok.size();
        while (a < b && std::isspace(static_cast<unsigned char>(tok[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(tok[b - 1]))) --b;
        if (b > a) out.push_back(tok.substr(a, b - a));
        if (p == std::string::npos) break;
        start = p + 1;
    }
    return out;
}

}  // namespace

static bool llama_loader_xnvme_run_upcie_cuda(
    const std::vector<llama_loader_xnvme_job> & jobs,
    const llama_files & /*files*/,
    ggml_backend_t upload_backend,
    ggml_backend_buffer_type_t /*host_buft*/,
    const std::string & xnvme_be,
    size_t & size_done,
    size_t   size_data,
    llama_progress_callback progress_cb,
    void   * progress_ud)
{
    // Resolve the drive fleet. LLAMA_XNVME_DMA_URIS takes precedence and can
    // list multiple URIs (comma-separated); LLAMA_XNVME_DMA_URI is the
    // backward-compat singular form. Same applies to LLAMA_XNVME_DMA_EXTENTS.
    // When N > 1 the model file must be present (byte-identical) on every
    // drive; the loader partitions the tensor-job list across drives and
    // rotates its submit loop through their queues, xnvmeperf-style.
    std::vector<std::string> uris    = parse_csv_env("LLAMA_XNVME_DMA_URIS");
    std::vector<std::string> exts    = parse_csv_env("LLAMA_XNVME_DMA_EXTENTS");
    if (uris.empty()) {
        const char * single_uri = std::getenv("LLAMA_XNVME_DMA_URI");
        if (single_uri && *single_uri) uris.emplace_back(single_uri);
    }
    if (exts.empty()) {
        const char * single_ext = std::getenv("LLAMA_XNVME_DMA_EXTENTS");
        if (single_ext && *single_ext) exts.emplace_back(single_ext);
    }
    if (uris.empty()) {
        LLAMA_LOG_WARN("%s: no DMA URI(s) set (LLAMA_XNVME_DMA_URIS or LLAMA_XNVME_DMA_URI); falling back to STREAM\n",
                       __func__);
        return false;
    }
    if (!exts.empty() && exts.size() != uris.size()) {
        LLAMA_LOG_WARN("%s: LLAMA_XNVME_DMA_EXTENTS has %zu entries but %zu URI(s); falling back to STREAM\n",
                       __func__, exts.size(), uris.size());
        return false;
    }

    const size_t n_drives = uris.size();

    const uint32_t qd      = std::max<uint32_t>(env_u32("LLAMA_XNVME_QD",      64), 2);
    const uint32_t io_size = std::max<uint32_t>(env_u32("LLAMA_XNVME_IO_SIZE", 512 * 1024), 4096);
    const uint32_t n_slots = qd;

    // Optional CPU pin. Accept a single CPU (backwards compat) or a comma
    // separated list; only the first entry is used because a single core
    // easily drives N queues in the rotate loop.
    cpu_set_t prev_affinity;
    bool have_prev_affinity = false;
    std::vector<std::string> cpus = parse_csv_env("LLAMA_XNVME_CPU");
    if (!cpus.empty()) {
        int cpu = std::atoi(cpus.front().c_str());
        if (cpu >= 0) {
            CPU_ZERO(&prev_affinity);
            if (sched_getaffinity(0, sizeof(prev_affinity), &prev_affinity) == 0) {
                have_prev_affinity = true;
                cpu_set_t mask;
                CPU_ZERO(&mask);
                CPU_SET(cpu, &mask);
                sched_setaffinity(0, sizeof(mask), &mask);
            }
        }
    }

    // If the process-global P2P registry was set up (via env var at library
    // load time, or explicit init), we reuse the devs it opened AND skip the
    // per-load xnvme_mem_map cost - the registry already fired async maps at
    // ggml-cuda buffer alloc time, so by now they're finished (or almost).
    const auto * registry_devs = llama_p2p_registry_devs();
    const bool   use_registry  = registry_devs && registry_devs->size() == n_drives;
    if (registry_devs && registry_devs->size() != n_drives) {
        LLAMA_LOG_WARN("%s: P2P registry has %zu dev(s) but %zu URI(s) requested; "
                       "falling back to per-load open+mem_map\n",
                       __func__, registry_devs->size(), n_drives);
    }

    // Owning storage for the fleet. Vector of unique_ptrs because upcie_drive
    // is non-movable (atomic member).
    std::vector<std::unique_ptr<upcie_drive>> drives;
    drives.reserve(n_drives);

    upcie_state st;

    // Helper: tear down everything allocated so far and return false.
    // When the P2P registry owns the devs (and mappings) we do NOT close
    // them here - registry cleanup happens at process exit.
    auto teardown_and_fail = [&](const char * why) -> bool {
        LLAMA_LOG_WARN("%s: %s; falling back to STREAM\n", __func__, why);
        for (auto & d : drives) {
            for (auto & s : d->slots) {
                if (s.dma_buf) xnvme_buf_free(d->dev, s.dma_buf);
                if (s.event)   cudaEventDestroy(s.event);
            }
            if (!use_registry) {
                for (void * base : d->mapped_buffers) {
                    xnvme_mem_unmap(d->dev, base);
                }
            }
            if (d->queue) xnvme_queue_term(d->queue);
            if (d->dev && !use_registry) xnvme_dev_close(d->dev);
        }
        if (st.stream) cudaStreamDestroy(st.stream);
        if (have_prev_affinity) sched_setaffinity(0, sizeof(prev_affinity), &prev_affinity);
        return false;
    };

    // One CUDA stream shared across all drives. D2D memcpys run at HBM
    // bandwidth (~800 GB/s) so serialising them on one stream is not a
    // bottleneck; per-stream would just add overhead.
    if (cudaStreamCreate(&st.stream) != cudaSuccess) {
        return teardown_and_fail("cudaStreamCreate failed");
    }

    xnvme_opts opts{};
    // Backend name only; see the note in llama-p2p-registry.cpp on why
    // async/sync/admin must not be pinned to it.
    opts.be     = xnvme_be.c_str();
    opts.nsid   = 1;
    opts.rdonly = 1;

    uint32_t io_size_eff = io_size;

    for (size_t i = 0; i < n_drives; ++i) {
        auto d = std::unique_ptr<upcie_drive>(new upcie_drive);
        d->uri = uris[i];
        if (use_registry) {
            d->dev = (*registry_devs)[i];
        } else {
            d->dev = xnvme_dev_open(d->uri.c_str(), &opts);
        }
        if (!d->dev) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "xnvme_dev_open('%s', be='%s') failed (errno=%d)",
                          d->uri.c_str(), xnvme_be.c_str(), errno);
            drives.push_back(std::move(d));
            return teardown_and_fail(msg);
        }
        const struct xnvme_geo * geo = xnvme_dev_get_geo(d->dev);
        d->nsid       = xnvme_dev_get_nsid(d->dev);
        d->lba_nbytes = geo ? geo->lba_nbytes : 0;
        const uint32_t mdts_nbytes = geo ? geo->mdts_nbytes : 0;
        if (d->lba_nbytes == 0) {
            drives.push_back(std::move(d));
            return teardown_and_fail("opened dev has lba_nbytes=0 (opts.nsid wrong?)");
        }
        if (i == 0) {
            io_size_eff = mdts_nbytes && io_size > mdts_nbytes ? mdts_nbytes : io_size;
        }
        d->mdts_nlb = io_size_eff / d->lba_nbytes;

        // Extents map for this drive: parsed from the corresponding
        // LLAMA_XNVME_DMA_EXTENTS entry if any, else the trivial raw-dd
        // {file_offset=0, lba=0, length=HUGE} fallback.
        if (!exts.empty()) {
            uint64_t file_size_ignored = 0;
            if (!parse_extents_json(exts[i].c_str(), d->extents, file_size_ignored)) {
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                              "parse_extents_json('%s') failed for drive '%s'",
                              exts[i].c_str(), d->uri.c_str());
                drives.push_back(std::move(d));
                return teardown_and_fail(msg);
            }
        } else {
            d->extents.push_back({0, 0, UINT64_MAX});
        }

        int rc = xnvme_queue_init(d->dev, static_cast<uint16_t>(qd), 0, &d->queue);
        if (rc) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "xnvme_queue_init(qd=%u) failed for drive '%s': rc=%d",
                          qd, d->uri.c_str(), rc);
            drives.push_back(std::move(d));
            return teardown_and_fail(msg);
        }

        d->slots.resize(n_slots);
        for (uint32_t k = 0; k < n_slots; ++k) {
            d->slots[k].dma_buf = xnvme_buf_alloc(d->dev, io_size_eff);
            if (!d->slots[k].dma_buf) {
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                              "xnvme_buf_alloc(%u) failed for drive '%s' slot %u",
                              io_size_eff, d->uri.c_str(), k);
                drives.push_back(std::move(d));
                return teardown_and_fail(msg);
            }
            cudaEventCreateWithFlags(&d->slots[k].event, cudaEventDisableTiming);
        }

        drives.push_back(std::move(d));
    }

    // Partition tensor jobs across drives. Round-robin by cumulative bytes:
    // walk jobs in disk order, keep assigning to the current drive until it
    // holds ~target_bytes, then move on. Any host tensor short-circuits back
    // to STREAM.
    size_t total_bytes = 0;
    for (const auto & job : jobs) {
        if (job.is_host) {
            LLAMA_LOG_WARN("%s: host tensor '%s' encountered; upcie-cuda path is device-only, falling back\n",
                           __func__, ggml_get_name(job.tensor));
            return teardown_and_fail("host tensor encountered");
        }
        total_bytes += job.n_size;
    }
    // Two partition modes:
    //   PER-FILE (sharded): if the URI list length matches the number of
    //     distinct file_idx values in `jobs`, each drive holds one shard
    //     of a split GGUF (drive-i owns file_idx == i). Route every job to
    //     its owning drive.
    //   BY-BYTES (mirrored): each URI's drive holds a byte-identical copy
    //     of the file, so any drive can serve any tensor. Spread jobs
    //     round-robin by cumulative bytes.
    // Sharded is the 1x-storage production shape; mirrored is the
    //     multi-copy demo shape.
    uint16_t max_file_idx = 0;
    for (const auto & job : jobs) {
        if (job.file_idx > max_file_idx) max_file_idx = job.file_idx;
    }
    const size_t n_files = static_cast<size_t>(max_file_idx) + 1;
    const bool   per_file_partition = (n_files == n_drives) && (n_drives > 1);
    if (per_file_partition) {
        LLAMA_LOG_WARN("%s: per-file partition: %zu shard(s) mapped 1:1 to %zu drive(s)\n",
                       __func__, n_files, n_drives);
        for (size_t j = 0; j < jobs.size(); ++j) {
            drives[jobs[j].file_idx]->job_indices.push_back(j);
        }
    } else {
        LLAMA_LOG_WARN("%s: by-bytes partition: %zu file(s) shared across %zu drive(s) (mirrored layout)\n",
                       __func__, n_files, n_drives);
        const size_t target_bytes = (total_bytes + n_drives - 1) / n_drives;
        size_t assigned_bytes = 0;
        size_t drive_i = 0;
        for (size_t j = 0; j < jobs.size(); ++j) {
            drives[drive_i]->job_indices.push_back(j);
            assigned_bytes += jobs[j].n_size;
            if (drive_i + 1 < n_drives && assigned_bytes >= (drive_i + 1) * target_bytes) {
                drive_i++;
            }
        }
    }

    // Ensure every ggml backend buffer we're about to write into has been
    // mem_map'd on every drive. When the process-global P2P registry is
    // active it has been firing async xnvme_mem_map calls at ggml-cuda
    // buffer alloc time; we just wait for the last stragglers here (usually
    // already done by the time load_all_data is called). Otherwise (no
    // registry, e.g. someone forgot to set LLAMA_XNVME_DMA_URIS at library
    // load time) we do the mem_map ourselves per drive.
    if (use_registry) {
        const auto t0 = std::chrono::steady_clock::now();
        llama_p2p_registry_barrier();
        const auto t1 = std::chrono::steady_clock::now();
        LLAMA_LOG_WARN("%s: P2P registry barrier: %.1f ms (mem_maps already in flight)\n",
                       __func__,
                       std::chrono::duration<double, std::milli>(t1 - t0).count());
    } else {
        std::vector<std::pair<void *, size_t>> unique_bufs;
        std::vector<void *> seen_bases;
        for (const auto & job : jobs) {
            if (!job.tensor || !job.tensor->buffer) continue;
            void * base = ggml_backend_buffer_get_base(job.tensor->buffer);
            const size_t sz = ggml_backend_buffer_get_size(job.tensor->buffer);
            if (std::find(seen_bases.begin(), seen_bases.end(), base) != seen_bases.end()) {
                continue;
            }
            seen_bases.push_back(base);
            unique_bufs.emplace_back(base, sz);
        }
        LLAMA_LOG_WARN("%s: registering %zu unique tensor buffer(s) with %zu drive(s) via xnvme_mem_map (no P2P registry)\n",
                       __func__, unique_bufs.size(), n_drives);
        for (auto & d : drives) {
            for (auto & bs : unique_bufs) {
                const auto t0 = std::chrono::steady_clock::now();
                int rc = xnvme_mem_map(d->dev, bs.first, bs.second);
                const auto t1 = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                LLAMA_LOG_WARN("%s: xnvme_mem_map(dev='%s', base=%p, size=%zu MiB) took %.1f ms\n",
                               __func__, d->uri.c_str(), bs.first,
                               bs.second / (1024 * 1024), ms);
                if (rc) {
                    char msg[256];
                    std::snprintf(msg, sizeof(msg),
                                  "xnvme_mem_map(dev='%s', base=%p, size=%zu) failed: rc=%d",
                                  d->uri.c_str(), bs.first, bs.second, rc);
                    return teardown_and_fail(msg);
                }
                d->mapped_buffers.push_back(bs.first);
            }
        }
    }

    // Helper that submits ONE chunk (up to io_size_eff or the current
    // extent's tail, whichever is smaller) for drive `d`. Returns true if
    // a chunk was submitted (or the drive is done), false if the queue
    // has no free ctx right now (retry on next rotate). Advances the
    // drive's cursor and job pointer on success. Updates size_done and
    // calls progress_cb the moment a job's last chunk is submitted.
    auto submit_one = [&](upcie_drive & d) -> bool {
        // Advance to next job if current is done.
        while (d.remaining_cur == 0) {
            if (d.job_cursor >= d.job_indices.size()) {
                return false;
            }
            const auto & job = jobs[d.job_indices[d.job_cursor]];
            d.src_off_cur   = job.offs;
            d.dst_off_cur   = 0;
            d.remaining_cur = job.n_size;
        }

        const auto & job = jobs[d.job_indices[d.job_cursor]];

        const dma_extent * ext = find_extent(d.extents, d.src_off_cur);
        if (!ext) {
            LLAMA_LOG_WARN("%s: no extent covers file_offset=%zu on drive '%s' (tensor '%s')\n",
                           __func__, d.src_off_cur, d.uri.c_str(), ggml_get_name(job.tensor));
            st.err.store(EINVAL, std::memory_order_relaxed);
            return false;
        }
        const uint64_t off_in_ext  = d.src_off_cur - ext->file_offset;
        const uint64_t rem_in_ext  = ext->length - off_in_ext;
        const uint64_t drive_boff  = ext->lba * static_cast<uint64_t>(d.lba_nbytes) + off_in_ext;
        const uint64_t slba        = drive_boff / d.lba_nbytes;
        const uint64_t drive_align = slba * static_cast<uint64_t>(d.lba_nbytes);
        const size_t   head_skip   = static_cast<size_t>(drive_boff - drive_align);
        size_t         want        = std::min<size_t>(d.remaining_cur, io_size_eff - head_skip);
        if (want > rem_in_ext) want = static_cast<size_t>(rem_in_ext);
        const uint64_t end_off = drive_align + head_skip + want;
        uint32_t nlb = static_cast<uint32_t>(
            ((end_off - drive_align) + d.lba_nbytes - 1) / d.lba_nbytes);
        if (nlb > d.mdts_nlb) {
            nlb  = d.mdts_nlb;
            want = static_cast<size_t>(nlb) * d.lba_nbytes - head_skip;
            if (want > d.remaining_cur) want = d.remaining_cur;
            if (want > rem_in_ext)      want = static_cast<size_t>(rem_in_ext);
        }

        // Pick a slot whose buffer isn't currently being written by the SSD.
        // Explicit scan (not round-robin) so out-of-order completions across
        // multiple queues never let us clobber an in-flight read. A slot with
        // in_flight_read=false is either FREE or D2D_QUEUED; for the latter
        // we sync its event before reuse. If every slot has in_flight_read=
        // true then the queue must also be full (n_slots == qd), so the
        // caller pokes and tries again.
        size_t chosen = SIZE_MAX;
        for (size_t k = 0; k < d.slots.size(); ++k) {
            const size_t idx = (d.next_slot + k) % d.slots.size();
            if (!d.slots[idx].in_flight_read) {
                chosen = idx;
                break;
            }
        }
        if (chosen == SIZE_MAX) return false;
        upcie_slot & s = d.slots[chosen];
        if (s.needs_wait) {
            cudaEventSynchronize(s.event);
            s.needs_wait = false;
        }
        d.next_slot = (chosen + 1) % d.slots.size();

        struct xnvme_cmd_ctx * ctx = xnvme_cmd_ctx_from_queue(d.queue);
        if (!ctx) {
            // Queue full - caller will poke and try again on the next rotate.
            return false;
        }

        // Direct-into-tensor when the read boundaries line up with LBA
        // granularity: head_skip == 0 AND read length is an LBA-multiple.
        // The NVMe controller writes `nlb * lba_nbytes` bytes starting at
        // the given buffer address; when want == nlb * lba_nbytes there is
        // no overshoot beyond the tensor's target region. Scratch stays as
        // the fallback for edge reads (unaligned head, non-LBA-multiple
        // tail).
        // LLAMA_XNVME_FORCE_SCRATCH=1 disables the direct path entirely
        // for A/B testing - we still register buffers via xnvme_mem_map
        // and take on that cost, so comparing against the direct-enabled
        // run isolates the per-submit direct-vs-scratch delta.
        static const bool force_scratch = std::getenv("LLAMA_XNVME_FORCE_SCRATCH") != nullptr
                                       && std::atoi(std::getenv("LLAMA_XNVME_FORCE_SCRATCH")) != 0;
        const bool direct_eligible = !force_scratch &&
                                     (head_skip == 0) &&
                                     (want % d.lba_nbytes == 0) &&
                                     (job.tensor && job.tensor->data);

        s.tensor         = job.tensor;
        s.tensor_off     = d.dst_off_cur;
        s.skip           = head_skip;
        s.len            = want;
        s.drive          = &d;
        s.state          = &st;
        s.in_flight_read = true;
        s.direct_target  = direct_eligible
            ? static_cast<uint8_t *>(job.tensor->data) + d.dst_off_cur
            : nullptr;
        if (direct_eligible) {
            st.direct_reads.fetch_add(1, std::memory_order_relaxed);
            st.direct_bytes.fetch_add(want, std::memory_order_relaxed);
        } else {
            st.scratch_reads.fetch_add(1, std::memory_order_relaxed);
            st.scratch_bytes.fetch_add(want, std::memory_order_relaxed);
        }

        ctx->async.cb     = on_completion_upcie;
        ctx->async.cb_arg = &s;

        void * dma_target = s.direct_target ? s.direct_target : s.dma_buf;
        int r;
        do {
            r = xnvme_nvm_read(ctx, d.nsid, slba, nlb - 1, dma_target, nullptr);
            if (r == -EBUSY) xnvme_queue_poke(d.queue, 0);
        } while (r == -EBUSY);
        if (r) {
            s.in_flight_read = false;
            xnvme_queue_put_cmd_ctx(d.queue, ctx);
            st.err.store(r < 0 ? -r : r, std::memory_order_relaxed);
            return false;
        }

        d.outstanding.fetch_add(1, std::memory_order_release);
        d.subs_count.fetch_add(1, std::memory_order_relaxed);
        d.subs_bytes.fetch_add(want, std::memory_order_relaxed);
        d.src_off_cur   += want;
        d.dst_off_cur   += want;
        d.remaining_cur -= want;

        if (d.remaining_cur == 0) {
            size_done += job.n_size;
            d.job_cursor++;
        }
        return true;
    };

    const auto t_submit_start = std::chrono::steady_clock::now();

    // Initial fill: for every drive, submit until the queue is full or
    // its assigned partition is exhausted.
    bool cancelled = false;
    for (auto & d : drives) {
        while (submit_one(*d)) {
            /* keep filling */
        }
        if (st.err.load(std::memory_order_relaxed) != 0) {
            cancelled = true;
            break;
        }
    }

    // Rotate loop: poke each drive, refill from tail; exit when every
    // drive has exhausted its partition and no I/O remains outstanding.
    // Progress callback fires as jobs finish submitting (matches the
    // former single-drive cadence). Cancellation drains cleanly.
    if (progress_cb && !progress_cb(
            static_cast<float>(size_done) / static_cast<float>(size_data),
            progress_ud)) {
        cancelled = true;
    }
    while (!cancelled && st.err.load(std::memory_order_relaxed) == 0) {
        bool any_pending    = false;
        bool any_outstanding = false;
        for (auto & d : drives) {
            xnvme_queue_poke(d->queue, 0);
            while (submit_one(*d)) {
                /* keep filling */
            }
            if (d->job_cursor < d->job_indices.size()) any_pending = true;
            if (d->outstanding.load(std::memory_order_acquire) > 0) any_outstanding = true;
        }
        if (!any_pending && !any_outstanding) break;
        if (progress_cb && !progress_cb(
                static_cast<float>(size_done) / static_cast<float>(size_data),
                progress_ud)) {
            cancelled = true;
        }
    }

    const auto t_submit_end = std::chrono::steady_clock::now();
    {
        const double ms = std::chrono::duration<double, std::milli>(t_submit_end - t_submit_start).count();
        const double gib_s = ms > 0.0
            ? (static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0)
            : 0.0;
        LLAMA_LOG_WARN("%s: submit+drain phase %.1f ms for %.1f MiB = %.1f GiB/s aggregate\n",
                       __func__, ms,
                       static_cast<double>(total_bytes) / (1024.0 * 1024.0),
                       gib_s);
    }

    // Drain outstanding I/O on every queue.
    for (auto & d : drives) {
        xnvme_queue_drain(d->queue);
    }

    // Retire any in-flight D2D copies.
    for (auto & d : drives) {
        for (auto & s : d->slots) {
            if (s.needs_wait) {
                cudaEventSynchronize(s.event);
                s.needs_wait = false;
            }
        }
    }
    cudaStreamSynchronize(st.stream);

    const bool ok = st.err.load(std::memory_order_relaxed) == 0;
    if (!ok) {
        LLAMA_LOG_WARN("%s: upcie-cuda loader failed (errno=%d)\n",
                       __func__, st.err.load(std::memory_order_relaxed));
    }

    {
        const uint64_t dr = st.direct_reads.load(std::memory_order_relaxed);
        const uint64_t sr = st.scratch_reads.load(std::memory_order_relaxed);
        const uint64_t db = st.direct_bytes.load(std::memory_order_relaxed);
        const uint64_t sb = st.scratch_bytes.load(std::memory_order_relaxed);
        const uint64_t total_reads = dr + sr;
        const uint64_t total_bytes = db + sb;
        const double pct_reads = total_reads ? 100.0 * dr / total_reads : 0.0;
        const double pct_bytes = total_bytes ? 100.0 * db / total_bytes : 0.0;
        LLAMA_LOG_WARN("%s: direct=%lu reads (%lu MiB, %.1f%%), scratch=%lu reads (%lu MiB, %.1f%%)\n",
                       __func__,
                       (unsigned long) dr, (unsigned long) (db / (1024 * 1024)), pct_bytes,
                       (unsigned long) sr, (unsigned long) (sb / (1024 * 1024)), 100.0 - pct_bytes);
        (void) pct_reads;

        for (size_t i = 0; i < drives.size(); ++i) {
            const uint64_t rc = drives[i]->subs_count.load(std::memory_order_relaxed);
            const uint64_t rb = drives[i]->subs_bytes.load(std::memory_order_relaxed);
            const double pct = total_bytes ? 100.0 * (double) rb / (double) total_bytes : 0.0;
            LLAMA_LOG_WARN("%s: drive[%zu] '%s': %lu reads, %lu MiB (%.1f%% of total)\n",
                           __func__, i, drives[i]->uri.c_str(),
                           (unsigned long) rc, (unsigned long) (rb / (1024 * 1024)), pct);
        }
    }

    // Cleanup: buffers, events, mem_map registrations, queues, devs.
    // When the P2P registry owns the devs+mappings, we leave both alone so
    // subsequent loads in the same process can reuse the pre-mapped state.
    for (auto & d : drives) {
        for (auto & s : d->slots) {
            if (s.dma_buf) xnvme_buf_free(d->dev, s.dma_buf);
            if (s.event)   cudaEventDestroy(s.event);
        }
        if (!use_registry) {
            for (void * base : d->mapped_buffers) {
                xnvme_mem_unmap(d->dev, base);
            }
        }
        if (d->queue) xnvme_queue_term(d->queue);
        if (d->dev && !use_registry) xnvme_dev_close(d->dev);
    }
    cudaStreamDestroy(st.stream);
    if (have_prev_affinity) sched_setaffinity(0, sizeof(prev_affinity), &prev_affinity);

    // Consume any lingering CUDA error state so llama.cpp's own kernel
    // launches don't inherit it. Anything showing up here is worth
    // logging (probably a warning-only issue in xnvme_dev_close or a
    // stale event) but shouldn't stop the load from succeeding at the
    // xnvme layer.
    {
        cudaError_t leftover = cudaGetLastError();
        if (leftover != cudaSuccess) {
            LLAMA_LOG_WARN("%s: cleared leftover CUDA error before return: %s\n",
                           __func__, cudaGetErrorString(leftover));
        }
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
