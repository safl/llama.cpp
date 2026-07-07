#pragma once

#include "llama.h"
#include "llama-mmap.h"

#include "ggml-backend.h"

#include <string>
#include <vector>

// Per-tensor work item consumed by the xNVMe pipelined ring loader.
// Populated by llama_model_loader::load_all_data before calling
// llama_loader_xnvme_run.
struct llama_loader_xnvme_job {
    struct ggml_tensor * tensor;   // destination tensor
    uint16_t             file_idx; // index into `files` for the source file
    uint64_t             offs;     // byte offset within the source file
    size_t               n_size;   // bytes to read
    bool                 is_host;  // true = memcpy into tensor->data (CPU tensor);
                                   // false = ggml_backend_tensor_set_async into device buffer
};

// Run the xNVMe pipelined ring loader. Returns true on success. Returns
// false when llama.cpp was built without LLAMA_USE_XNVME (falls through
// to caller which should retry via the STREAM path) or when a runtime
// setup step fails (open, queue init, or unsupported opts.be). On
// setup failure a one-line warning is logged.
//
// The jobs vector is expected to be sorted by (file_idx, offs) for
// sequential disk order.
//
// size_data is the total bytes across all jobs; size_done accumulates
// bytes completed and progress_cb is invoked with size_done / size_data
// as the callback fires per-chunk. Returning false from progress_cb
// aborts the load; the loader drains outstanding submissions and
// returns true with size_done < size_data so the caller can propagate
// the cancellation just like the STREAM path does.
//
// upload_backend and host_buft come from the existing upload_backend
// lambda in load_all_data: the loader reuses them for pinned host
// staging and per-slot ggml_backend_event_t.
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
    void   * progress_ud);
