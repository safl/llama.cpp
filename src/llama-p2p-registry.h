#pragma once

#include <cstddef>
#include <string>
#include <vector>

#ifdef LLAMA_USE_XNVME
struct xnvme_dev;
#endif

// Process-global P2P registry
//
// Opens one or more xNVMe upcie-cuda devices and installs a ggml-cuda buffer
// alloc/free hook so every CUDA backend allocation gets xnvme_mem_map'd on
// each open dev in a background thread. The multi-drive rotate loader
// consumes the already-mapped state instead of doing the (blocking) mem_map
// itself, which hides the ~650 ms mapping cost behind llama.cpp's own
// tensor-allocation / GGUF-parse work.
//
// Idempotent when re-invoked with the same URI list. Initialised
// automatically from the LLAMA_XNVME_DMA_URIS env var at library load time
// if it is set; can also be initialised explicitly.

// Init the registry from a comma-separated URI list. On success returns 0
// and installs the ggml-cuda hooks. Returns negative errno on failure.
// A second call with the identical URI list is a no-op; with a different
// list returns -EINVAL.
int llama_p2p_registry_init(const std::string & uris_csv);

#ifdef LLAMA_USE_XNVME
// Ordered list of open dev handles that the loader should use directly.
// Nullptr if the registry has not been initialised.
const std::vector<xnvme_dev *> * llama_p2p_registry_devs();
#endif

// Wait for any in-flight mem_map registrations to complete. Loader calls
// this once just before its submit loop starts. Cheap when nothing is in
// flight (the common steady-state case).
void llama_p2p_registry_barrier();

// Tear down: unmap every mapping, close every dev, remove the ggml-cuda
// hooks. Safe to call multiple times; safe to call if the registry was
// never initialised.
void llama_p2p_registry_shutdown();
