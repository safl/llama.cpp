"""Probe the CUDA driver's alloc-granularity minimum and recommended values.

Writes a tiny standalone C driver-API program to a scratch dir on the
target, compiles it against the CUDA toolkit already present for the
llama.cpp build, runs it, and reports the two granularity values.
Answers whether switching cudamem_config from CU_MEM_ALLOC_GRANULARITY_MINIMUM
to CU_MEM_ALLOC_GRANULARITY_RECOMMENDED would reduce our dma-buf chunk
count.
"""


def add_args(parser):
    pass


PROBE_C = r"""
#include <cuda.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    CUresult cr;
    cr = cuInit(0);
    if (cr != CUDA_SUCCESS) { fprintf(stderr, "cuInit: %d\n", cr); return 1; }

    CUdevice dev;
    cr = cuDeviceGet(&dev, 0);
    if (cr != CUDA_SUCCESS) { fprintf(stderr, "cuDeviceGet: %d\n", cr); return 1; }

    char name[128];
    cuDeviceGetName(name, sizeof(name), dev);
    printf("device: %s\n", name);

    size_t total_mem = 0;
    cuDeviceTotalMem(&total_mem, dev);
    printf("device_memsize: %zu (%.2f GiB)\n", total_mem, total_mem / 1073741824.0);

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)dev;

    size_t gran_min = 0, gran_rec = 0;
    cr = cuMemGetAllocationGranularity(&gran_min, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (cr != CUDA_SUCCESS) { fprintf(stderr, "gran min: %d\n", cr); return 1; }
    cr = cuMemGetAllocationGranularity(&gran_rec, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    if (cr != CUDA_SUCCESS) { fprintf(stderr, "gran rec: %d\n", cr); return 1; }

    printf("granularity MINIMUM:     %zu bytes (%.2f MiB)\n",
           gran_min, gran_min / 1048576.0);
    printf("granularity RECOMMENDED: %zu bytes (%.2f MiB)\n",
           gran_rec, gran_rec / 1048576.0);
    printf("ratio recommended/minimum: %.1fx\n",
           gran_min ? (double)gran_rec / (double)gran_min : 0.0);

    size_t chunks_min = (total_mem + gran_min - 1) / gran_min;
    size_t chunks_rec = (total_mem + gran_rec - 1) / gran_rec;
    printf("full-VRAM chunk count MINIMUM:     %zu\n", chunks_min);
    printf("full-VRAM chunk count RECOMMENDED: %zu\n", chunks_rec);

    // For a ~40 GB workload equivalent (like the 70B Q4 model):
    size_t workload = 40ULL << 30;
    printf("40 GiB workload chunks MINIMUM:     %zu\n", (workload + gran_min - 1) / gran_min);
    printf("40 GiB workload chunks RECOMMENDED: %zu\n", (workload + gran_rec - 1) / gran_rec);

    // Feasibility of collapse-at-insertion: does cuMemGetHandleForAddressRange
    // accept a size larger than alloc_granularity? Try increasing sizes on a
    // real cudaMalloc'd buffer.
    printf("\n== cuMemGetHandleForAddressRange size test ==\n");
    CUcontext ctx = NULL;
    // CUDA 13 renamed cuCtxCreate to cuCtxCreate_v4 with an extra params
    // struct; NULL for the params gets the pre-v4 default behaviour.
    cr = cuCtxCreate(&ctx, NULL, 0, dev);
    if (cr != CUDA_SUCCESS) { fprintf(stderr, "cuCtxCreate: %d\n", cr); return 1; }

    // Allocate 42 GiB (matching the 70B Q4_K_M model size) to test if a single
    // cuMemGetHandleForAddressRange call over the whole registration works.
    size_t test_size = 42ULL << 30;
    CUdeviceptr d_ptr = 0;
    cr = cuMemAlloc(&d_ptr, test_size);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "cuMemAlloc(42 GiB) failed: cr=%d, falling back to 8 GiB\n", cr);
        test_size = 8ULL << 30;
        cr = cuMemAlloc(&d_ptr, test_size);
        if (cr != CUDA_SUCCESS) {
            fprintf(stderr, "cuMemAlloc(8 GiB) failed: cr=%d\n", cr);
            return 1;
        }
    }
    printf("test buffer: %zu bytes (%.2f GiB) @ 0x%llx\n",
           test_size, test_size / 1073741824.0, (unsigned long long)d_ptr);

    // Round vaddr up to gran boundary for cuMemGetHandleForAddressRange.
    unsigned long long va = (unsigned long long)d_ptr;
    va = (va + gran_min - 1) & ~(gran_min - 1);
    size_t usable = test_size - ((size_t)(va - (unsigned long long)d_ptr));
    usable = usable & ~(gran_min - 1);

    const size_t sizes[] = {
        1 * gran_min,
        128 * gran_min,     // 256 MiB
        1024 * gran_min,    // 2 GiB
        4096 * gran_min,    // 8 GiB
        16384 * gran_min,   // 32 GiB
        20480 * gran_min,   // 40 GiB
        usable,             // full test buffer (42 GiB if we got it, else 8 GiB)
    };
    for (size_t k = 0; k < sizeof(sizes) / sizeof(sizes[0]); ++k) {
        size_t sz = sizes[k];
        if (sz > usable) { printf("  skip size %zu (bigger than test buffer)\n", sz); continue; }
        int fd = -1;
        cr = cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)va, sz,
                                           CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        if (cr != CUDA_SUCCESS) {
            printf("  size %zu (%zu chunks): FAILED cr=%d\n", sz, sz / gran_min, cr);
            continue;
        }
        printf("  size %zu (%zu chunks): OK, dmabuf_fd=%d\n", sz, sz / gran_min, fd);
        if (fd >= 0) close(fd);
    }

    cuMemFree(d_ptr);
    cuCtxDestroy(ctx);
    return 0;
}
"""


def main(args, cijoe):
    err, _ = cijoe.run("mkdir -p /root/gran-probe")
    if err:
        return err

    # Write the C file on the target directly.
    err, _ = cijoe.run(
        "cat > /root/gran-probe/probe.c <<'EOF'\n"
        + PROBE_C
        + "\nEOF"
    )
    if err:
        return err

    # Compile against the CUDA toolkit. -lcuda for the driver API.
    err, _ = cijoe.run(
        "gcc -O2 -I/usr/local/cuda/include /root/gran-probe/probe.c "
        "-L/usr/local/cuda/lib64 -lcuda -o /root/gran-probe/probe"
    )
    if err:
        return err

    err, _ = cijoe.run("/root/gran-probe/probe")
    return err
