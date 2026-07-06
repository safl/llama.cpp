/*
 * xnvme_load: read a file into a DMA-capable host buffer using xNVMe's
 * io_uring file backend. Submits async reads of ``mdts_nbytes`` per
 * command up to a configurable queue depth. Reports throughput.
 *
 * First cut on the road to NVMe -> VRAM: this variant reads through the
 * kernel filesystem stack (open + io_uring + O_DIRECT) and lands in host
 * memory. Enough to bracket the ceiling of the async-pread approach
 * against llama.cpp's mmap + cudaMemcpyAsync baseline. A follow-up
 * variant swaps the file backend for xal on the raw block device to drop
 * the kernel out of the data path.
 */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>

#include <libxnvme.h>
#include <libxnvme_file.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

struct progress {
	uint64_t outstanding;
	uint64_t submitted;
	int err;
};

static void
cb_read(struct xnvme_cmd_ctx *ctx, void *cb_arg)
{
	struct progress *p = cb_arg;

	if (xnvme_cmd_ctx_cpl_status(ctx)) {
		xnvme_cmd_ctx_pr(ctx, XNVME_PR_DEF);
		p->err = -EIO;
	}
	p->outstanding -= 1;
	xnvme_queue_put_cmd_ctx(ctx->async.queue, ctx);
}

static uint64_t
now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int
main(int argc, char *argv[])
{
	const char *file_path = NULL;
	const char *be = "io_uring";
	uint16_t qd = 64;
	uint64_t io_size = 0;
	struct xnvme_dev *dev = NULL;
	struct xnvme_queue *queue = NULL;
	char *dst = NULL;
	int rc = 1;

	// Unbuffered stdout so early crashes still show progress prints
	setvbuf(stdout, NULL, _IONBF, 0);

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--file") && i + 1 < argc)
			file_path = argv[++i];
		else if (!strcmp(argv[i], "--async") && i + 1 < argc)
			be = argv[++i];
		else if (!strcmp(argv[i], "--qd") && i + 1 < argc)
			qd = (uint16_t)atoi(argv[++i]);
		else if (!strcmp(argv[i], "--io-size") && i + 1 < argc)
			io_size = strtoull(argv[++i], NULL, 0);
	}
	if (!file_path) {
		fprintf(stderr,
			"usage: %s --file <path> [--async <backend>] [--qd <N>] [--io-size <bytes>]\n",
			argv[0]);
		return 1;
	}

	// Sparse init like xnvme's own tools do. Do not call
	// xnvme_opts_default() here since it sets rdwr=1 and the file
	// backend rejects the ambiguous rdonly+rdwr combination with
	// -ENXIO.
	//
	// xnvme's platform config table registers io_uring_nvme before
	// io_uring_file. When only opts.async is set to "io_uring", the
	// resolver picks io_uring_nvme first, opens the regular file via
	// its stat-based dev_open (which does not reject S_IFREG), then
	// tries NVMe admin ioctls that ENOTTY on a filesystem file, and
	// downstream code segfaults. Pin the file config explicitly via
	// opts.be.
	struct xnvme_opts opts = {
		.be = "io_uring_file",
		.async = be,
		.rdonly = 1,
		.direct = 1,
	};

	dev = xnvme_file_open(file_path, &opts);
	if (!dev) {
		perror("xnvme_file_open");
		return 1;
	}

	const struct xnvme_geo *geo = xnvme_dev_get_geo(dev);

	// xNVMe's linux file backend does not propagate st_size into
	// geo->tbytes for regular files, and reports mdts_nbytes=0 since
	// a filesystem file has no NVMe MDTS. Fall back to stat() for the
	// file size and to a sensible default for the per-command chunk.
	struct stat st;
	if (stat(file_path, &st) < 0) {
		perror("stat");
		goto out;
	}
	uint64_t file_size = (uint64_t)st.st_size;

	if (io_size == 0)
		io_size = geo->mdts_nbytes ? geo->mdts_nbytes : (1ULL << 20);

	printf("file=%s size=%.2f MiB io_size=%" PRIu64
	       " KiB async=%s qd=%u\n",
	       file_path, (double)file_size / (1024.0 * 1024.0),
	       io_size / 1024, be, qd);

	uint64_t buf_size = (file_size + io_size - 1) & ~(io_size - 1);

	dst = xnvme_buf_alloc(dev, buf_size);
	if (!dst) {
		perror("xnvme_buf_alloc");
		goto out;
	}

	int err = xnvme_queue_init(dev, qd, 0, &queue);

	if (err) {
		fprintf(stderr, "xnvme_queue_init: %d\n", err);
		goto out;
	}

	struct progress p = {0};
	uint64_t t0 = now_ns();
	uint64_t offset = 0;

	while (offset < file_size) {
		// Always submit io_size per read. Under O_DIRECT the length
		// must be a multiple of the drive's logical sector size,
		// and shrinking the tail to (file_size - offset) breaks
		// that alignment. The kernel returns a short cqe->res on
		// the last chunk when offset+io_size crosses EOF; dst is
		// already padded to io_size alignment.
		uint64_t nbytes = io_size;

		struct xnvme_cmd_ctx *ctx = xnvme_cmd_ctx_from_queue(queue);

		if (!ctx) {
			fprintf(stderr,
				"xnvme_cmd_ctx_from_queue returned NULL "
				"(errno=%d, outstanding=%" PRIu64
				", submitted=%" PRIu64 ")\n",
				errno, p.outstanding, p.submitted);
			goto out;
		}

		ctx->async.cb = cb_read;
		ctx->async.cb_arg = &p;

submit:
		err = xnvme_file_pread(ctx, dst + offset, nbytes, offset);
		if (err == -EBUSY) {
			xnvme_queue_poke(queue, 0);
			goto submit;
		}
		if (err) {
			fprintf(stderr, "xnvme_file_pread: %d\n", err);
			goto out;
		}

		p.outstanding += 1;
		p.submitted += 1;
		offset += nbytes;
	}

	// xnvme_queue_drain returns the number of completions processed on
	// success, negative errno on failure. Non-zero is not an error.
	err = xnvme_queue_drain(queue);
	if (err < 0) {
		fprintf(stderr, "xnvme_queue_drain: %d\n", err);
		goto out;
	}

	uint64_t t1 = now_ns();
	double sec = (double)(t1 - t0) / 1e9;
	double mib = (double)file_size / (1024.0 * 1024.0);

	printf("read %.2f MiB in %.3f s -> %.1f MiB/s (%.2f GB/s) target=host, %" PRIu64
	       " cmds\n",
	       mib, sec, mib / sec, mib / sec / 1024.0, p.submitted);

#ifdef WITH_CUDA
	void *dev_buf = NULL;
	cudaError_t cerr = cudaMalloc(&dev_buf, buf_size);

	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(cerr));
		goto out;
	}
	cerr = cudaMemcpy(dev_buf, dst, file_size, cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy: %s\n", cudaGetErrorString(cerr));
		cudaFree(dev_buf);
		goto out;
	}
	cudaDeviceSynchronize();

	uint64_t t2 = now_ns();
	double sec_vram = (double)(t2 - t0) / 1e9;

	printf("read %.2f MiB in %.3f s -> %.1f MiB/s (%.2f GB/s) target=vram\n",
	       mib, sec_vram, mib / sec_vram, mib / sec_vram / 1024.0);

	cudaFree(dev_buf);
#endif

	if (p.err) {
		fprintf(stderr, "one or more commands failed: %d\n", p.err);
		goto out;
	}

	{
		struct rusage ru = {0};

		if (getrusage(RUSAGE_SELF, &ru) == 0) {
			printf("max_rss=%ld KiB (%.2f MiB)\n", ru.ru_maxrss,
			       (double)ru.ru_maxrss / 1024.0);
		}
	}

	rc = 0;

out:
	if (queue)
		xnvme_queue_term(queue);
	if (dst)
		xnvme_buf_free(dev, dst);
	if (dev)
		xnvme_file_close(dev);
	return rc;
}
