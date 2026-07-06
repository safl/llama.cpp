/*
 * xal_load: read a file's byte ranges directly from the underlying NVMe
 * device via xNVMe's io_uring_cmd passthrough, and pipeline
 * cudaMemcpyAsync per chunk into a VRAM buffer using a bounded
 * ring of pinned host slots.
 *
 * Shape:
 *   1. Open the .gguf, call FS_IOC_FIEMAP to get its extent list.
 *   2. Open the NVMe namespace char device via xNVMe's io_uring_cmd
 *      backend.
 *   3. Allocate a device buffer sized to the file (rounded up to
 *      io_size), a dedicated CUDA stream, and a ring of `qd` pinned
 *      host slots each of `io_size` bytes. Peak pinned host memory =
 *      qd * io_size (e.g. 64 * 128 KiB = 8 MiB) rather than the
 *      full file size.
 *   4. For each extent, chunk into io_size-sized reads. Each submit
 *      picks a free ring slot (blocking until one is available via a
 *      poll loop over cudaEventQuery + xnvme_queue_poke), records the
 *      slot's file offset and size, and issues xnvme_nvm_read into
 *      the slot's host buffer.
 *   5. The io_uring completion callback immediately fires
 *      cudaMemcpyAsync(dev_buf+offset, slot->host_buf, size, stream)
 *      and cudaEventRecord(slot->event, stream). The slot stays "in
 *      use" until the event fires, at which point the next
 *      pick_free_slot call recycles it.
 *   6. After xnvme_queue_drain (all reads done, all copies queued),
 *      cudaStreamSynchronize waits for the tail copies. Report both
 *      target=host (drain time) and target=vram (stream sync time)
 *      plus peak RSS.
 *
 * Naming is historical; extents come from FS_IOC_FIEMAP, not xal.
 */
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <libxnvme.h>
#include <libxnvme_nvm.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

struct progress;

struct slot {
	char *host_buf;
#ifdef WITH_CUDA
	cudaEvent_t event;
#endif
	int in_use;
	uint64_t file_offset;
	uint64_t size;
	struct progress *p;
};

struct progress {
	uint64_t outstanding;
	uint64_t submitted;
	int err;
	struct slot *slots;
	uint32_t n_slots;
#ifdef WITH_CUDA
	char *dev_buf;
	cudaStream_t stream;
	int use_cuda;
#endif
};

static void
cb_read(struct xnvme_cmd_ctx *ctx, void *cb_arg)
{
	struct slot *s = cb_arg;
	struct progress *p = s->p;
	struct xnvme_queue *q = ctx->async.queue;

	if (xnvme_cmd_ctx_cpl_status(ctx)) {
		xnvme_cmd_ctx_pr(ctx, XNVME_PR_DEF);
		p->err = -EIO;
		s->in_use = 0;
		goto put;
	}
#ifdef WITH_CUDA
	if (p->use_cuda) {
		cudaMemcpyAsync(p->dev_buf + s->file_offset,
				s->host_buf, s->size,
				cudaMemcpyHostToDevice, p->stream);
		cudaEventRecord(s->event, p->stream);
		/* slot stays in_use until its event fires */
	} else {
		s->in_use = 0;
	}
#else
	s->in_use = 0;
#endif
put:
	p->outstanding -= 1;
	xnvme_queue_put_cmd_ctx(q, ctx);
}

static struct slot *
pick_free_slot(struct xnvme_queue *queue, struct progress *p)
{
	while (1) {
		for (uint32_t i = 0; i < p->n_slots; i++) {
			struct slot *s = &p->slots[i];

			if (!s->in_use) {
				return s;
			}
#ifdef WITH_CUDA
			if (p->use_cuda &&
			    cudaEventQuery(s->event) == cudaSuccess) {
				s->in_use = 0;
				return s;
			}
#endif
		}
		xnvme_queue_poke(queue, 0);
	}
}

static uint64_t
now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static struct fiemap *
get_fiemap(int fd, uint64_t file_size)
{
	struct fiemap head = {
		.fm_start = 0,
		.fm_length = file_size,
		.fm_flags = FIEMAP_FLAG_SYNC,
		.fm_extent_count = 0,
	};

	if (ioctl(fd, FS_IOC_FIEMAP, &head) < 0) {
		perror("FS_IOC_FIEMAP (count)");
		return NULL;
	}

	uint32_t nex = head.fm_mapped_extents;
	size_t nbytes = sizeof(struct fiemap) +
			(size_t)nex * sizeof(struct fiemap_extent);
	struct fiemap *fm = calloc(1, nbytes);

	if (!fm) {
		perror("calloc(fiemap)");
		return NULL;
	}
	fm->fm_start = 0;
	fm->fm_length = file_size;
	fm->fm_flags = FIEMAP_FLAG_SYNC;
	fm->fm_extent_count = nex;

	if (ioctl(fd, FS_IOC_FIEMAP, fm) < 0) {
		perror("FS_IOC_FIEMAP (fetch)");
		free(fm);
		return NULL;
	}
	return fm;
}

int
main(int argc, char *argv[])
{
	const char *dev_uri = NULL;
	const char *file_path = NULL;
	uint16_t qd = 64;
	uint64_t io_size = 0;
	struct xnvme_dev *dev = NULL;
	struct xnvme_queue *queue = NULL;
	struct fiemap *fm = NULL;
	struct progress p = {0};
	int fd = -1;
	int rc = 1;

	setvbuf(stdout, NULL, _IONBF, 0);

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--dev-uri") && i + 1 < argc)
			dev_uri = argv[++i];
		else if (!strcmp(argv[i], "--file") && i + 1 < argc)
			file_path = argv[++i];
		else if (!strcmp(argv[i], "--qd") && i + 1 < argc)
			qd = (uint16_t)atoi(argv[++i]);
		else if (!strcmp(argv[i], "--io-size") && i + 1 < argc)
			io_size = strtoull(argv[++i], NULL, 0);
	}
	if (!dev_uri || !file_path) {
		fprintf(stderr,
			"usage: %s --dev-uri <ng-dev> --file <mount-path> "
			"[--qd <N>] [--io-size <bytes>]\n",
			argv[0]);
		return 1;
	}

	fd = open(file_path, O_RDONLY);
	if (fd < 0) {
		perror("open");
		return 1;
	}

	struct stat st;

	if (fstat(fd, &st) < 0) {
		perror("fstat");
		goto out;
	}

	uint64_t file_size = (uint64_t)st.st_size;

	fm = get_fiemap(fd, file_size);
	if (!fm) {
		goto out;
	}

	struct xnvme_opts opts = {
		.async = "io_uring_cmd",
		.rdonly = 1,
		.direct = 1,
	};

	dev = xnvme_dev_open(dev_uri, &opts);
	if (!dev) {
		perror("xnvme_dev_open");
		goto out;
	}

	const struct xnvme_geo *geo = xnvme_dev_get_geo(dev);
	uint32_t nsid = xnvme_dev_get_nsid(dev);
	uint32_t lba_nbytes = geo->lba_nbytes;
	uint32_t mdts_nbytes_dev = geo->mdts_nbytes;

	if (io_size == 0)
		io_size = mdts_nbytes_dev ? mdts_nbytes_dev : (1ULL << 20);

	uint32_t mdts_nlb = (uint32_t)(io_size / lba_nbytes);
	uint64_t buf_size = (file_size + io_size - 1) & ~(io_size - 1);

	p.n_slots = qd;
	p.slots = calloc(p.n_slots, sizeof(struct slot));
	if (!p.slots) {
		perror("calloc(slots)");
		goto out;
	}

#ifdef WITH_CUDA
	cudaError_t cerr;

	cerr = cudaMalloc((void **)&p.dev_buf, buf_size);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(%.2f MiB): %s\n",
			(double)buf_size / (1024.0 * 1024.0),
			cudaGetErrorString(cerr));
		goto out;
	}
	cerr = cudaStreamCreate(&p.stream);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudaStreamCreate: %s\n",
			cudaGetErrorString(cerr));
		goto out;
	}
	for (uint32_t i = 0; i < p.n_slots; i++) {
		cerr = cudaHostAlloc((void **)&p.slots[i].host_buf,
				     io_size, cudaHostAllocDefault);
		if (cerr != cudaSuccess) {
			fprintf(stderr, "cudaHostAlloc slot %u: %s\n", i,
				cudaGetErrorString(cerr));
			goto out;
		}
		cerr = cudaEventCreateWithFlags(&p.slots[i].event,
						cudaEventDisableTiming);
		if (cerr != cudaSuccess) {
			fprintf(stderr, "cudaEventCreate slot %u: %s\n", i,
				cudaGetErrorString(cerr));
			goto out;
		}
	}
	p.use_cuda = 1;
#else
	for (uint32_t i = 0; i < p.n_slots; i++) {
		p.slots[i].host_buf = xnvme_buf_alloc(dev, io_size);
		if (!p.slots[i].host_buf) {
			perror("xnvme_buf_alloc");
			goto out;
		}
	}
#endif

	int err = xnvme_queue_init(dev, qd, 0, &queue);

	if (err) {
		fprintf(stderr, "xnvme_queue_init: %d\n", err);
		goto out;
	}

	printf("file=%s size=%.2f MiB extents=%u lba=%u io_size=%" PRIu64
	       " KiB qd=%u n_slots=%u\n",
	       file_path, (double)file_size / (1024.0 * 1024.0),
	       fm->fm_mapped_extents, lba_nbytes, io_size / 1024, qd,
	       p.n_slots);

	uint64_t t0 = now_ns();

	for (uint32_t i = 0; i < fm->fm_mapped_extents; i++) {
		const struct fiemap_extent *fe = &fm->fm_extents[i];

		if (fe->fe_flags & FIEMAP_EXTENT_UNKNOWN) {
			fprintf(stderr,
				"extent %u marked UNKNOWN; cannot read direct\n",
				i);
			goto out;
		}

		uint64_t slba = fe->fe_physical / lba_nbytes;
		uint64_t nlb_ext = fe->fe_length / lba_nbytes;
		uint64_t file_off_ext = fe->fe_logical;

		while (nlb_ext) {
			uint32_t nlb = (nlb_ext < mdts_nlb) ? (uint32_t)nlb_ext
							    : mdts_nlb;
			struct slot *s = pick_free_slot(queue, &p);
			struct xnvme_cmd_ctx *ctx =
				xnvme_cmd_ctx_from_queue(queue);

			while (!ctx) {
				xnvme_queue_poke(queue, 0);
				ctx = xnvme_cmd_ctx_from_queue(queue);
			}

			s->file_offset = file_off_ext;
			s->size = (uint64_t)nlb * lba_nbytes;
			s->p = &p;
			s->in_use = 1;

			ctx->async.cb = cb_read;
			ctx->async.cb_arg = s;

submit:
			err = xnvme_nvm_read(ctx, nsid, slba, nlb - 1,
					     s->host_buf, NULL);
			if (err == -EBUSY) {
				xnvme_queue_poke(queue, 0);
				goto submit;
			}
			if (err) {
				fprintf(stderr, "xnvme_nvm_read: %d\n", err);
				goto out;
			}

			p.outstanding += 1;
			p.submitted += 1;

			slba += nlb;
			file_off_ext += s->size;
			nlb_ext -= nlb;
		}
	}

	err = xnvme_queue_drain(queue);
	if (err < 0) {
		fprintf(stderr, "xnvme_queue_drain: %d\n", err);
		goto out;
	}

	uint64_t t_read = now_ns();
	double sec_read = (double)(t_read - t0) / 1e9;
	double mib = (double)file_size / (1024.0 * 1024.0);

	printf("read %.2f MiB in %.3f s -> %.1f MiB/s (%.2f GB/s) target=host, "
	       "%" PRIu64 " cmds\n",
	       mib, sec_read, mib / sec_read, mib / sec_read / 1024.0,
	       p.submitted);

	if (p.err) {
		fprintf(stderr, "one or more commands failed: %d\n", p.err);
		goto out;
	}

#ifdef WITH_CUDA
	if (p.use_cuda) {
		cudaStreamSynchronize(p.stream);

		uint64_t t_vram = now_ns();
		double sec_vram = (double)(t_vram - t0) / 1e9;

		printf("read %.2f MiB in %.3f s -> %.1f MiB/s (%.2f GB/s) "
		       "target=vram\n",
		       mib, sec_vram, mib / sec_vram,
		       mib / sec_vram / 1024.0);
	}
#endif

	struct rusage ru = {0};

	if (getrusage(RUSAGE_SELF, &ru) == 0) {
		printf("max_rss=%ld KiB (%.2f MiB)\n", ru.ru_maxrss,
		       (double)ru.ru_maxrss / 1024.0);
	}

	rc = 0;

out:
	if (queue)
		xnvme_queue_term(queue);
#ifdef WITH_CUDA
	if (p.use_cuda) {
		for (uint32_t i = 0; i < p.n_slots; i++) {
			if (p.slots && p.slots[i].host_buf)
				cudaFreeHost(p.slots[i].host_buf);
			if (p.slots && p.slots[i].event)
				cudaEventDestroy(p.slots[i].event);
		}
		if (p.stream)
			cudaStreamDestroy(p.stream);
		if (p.dev_buf)
			cudaFree(p.dev_buf);
	}
#else
	if (p.slots) {
		for (uint32_t i = 0; i < p.n_slots; i++) {
			if (p.slots[i].host_buf)
				xnvme_buf_free(dev, p.slots[i].host_buf);
		}
	}
#endif
	if (p.slots)
		free(p.slots);
	if (dev)
		xnvme_dev_close(dev);
	if (fm)
		free(fm);
	if (fd >= 0)
		close(fd);
	return rc;
}
