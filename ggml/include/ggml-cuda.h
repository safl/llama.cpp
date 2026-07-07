#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// Optional observer hooks fired when the CUDA backend's device buffer type
// allocates or frees a buffer via cudaMalloc. Intended for out-of-tree P2P
// integrations (e.g. registering the freshly-allocated VRAM range with a
// user-space NVMe stack) that need to react to allocations without owning
// the ggml backend. Set to NULL to disable. The hooks fire from whichever
// thread calls the buffer type's alloc/free path; the callee is responsible
// for its own synchronization.
typedef void (*ggml_cuda_buffer_alloc_hook_t)(void * user_data, int device, void * base, size_t size);
typedef void (*ggml_cuda_buffer_free_hook_t) (void * user_data, int device, void * base);

GGML_BACKEND_API void ggml_backend_cuda_set_buffer_hooks(
    ggml_cuda_buffer_alloc_hook_t alloc_hook,
    ggml_cuda_buffer_free_hook_t  free_hook,
    void * user_data);

#ifdef  __cplusplus
}
#endif
