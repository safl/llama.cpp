/*
 * iou_ring_load: read a file into a VRAM buffer through liburing on the
 * open file descriptor, using a bounded ring of pinned host slots and
 * per-chunk cudaMemcpyAsync fired from cqe handling.
 *
 * Same architecture as xal-load minus the raw-device path: instead of
 * NVMe passthrough over io_uring_cmd on /dev/ngXnY with FIEMAP extents,
 * this variant reads the mounted-FS file via plain liburing + O_DIRECT.
 * All the memory + pipelining machinery is identical.
 *
 * Shape:
 *   1. open(file, O_RDONLY | O_DIRECT). fstat for size.
 *   2. Allocate a device buffer sized to the file (rounded up to
 *      io_size), a CUDA stream, and a ring of `qd` pinned host slots
 *      each of `io_size` bytes.
 *   3. Loop: fill io_uring SQ with reads into free slots; io_uring_submit;
 *      reap cqes eagerly. Each cqe fires cudaMemcpyAsync on the stream
 *      + cudaEventRecord tied to the slot. Slots recycle when their
 *      event has fired.
 *   4. After all reads complete, cudaStreamSynchronize.
 */
#define _GNU_SOURCE 1

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
#include <unistd.h>

#include <liburing.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

struct slot {
	char *host_buf;
#ifdef WITH_CUDA
	cudaEvent_t event;
#endif
	int in_use;
	uint64_t file_offset;
	uint64_t size;
};

struct progress {
	uint64_t outstanding;
	uint64_t submitted;
	uint64_t completed;
	int err;
	struct slot *slots;
	uint32_t n_slots;
#ifdef WITH_CUDA
	char *dev_buf;
	cudaStream_t stream;
	int use_cuda;
#endif
};

static uint64_t
now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void
handle_cqe(struct io_uring_cqe *cqe, struct progress *p)
{
	struct slot *s = (struct slot *)(uintptr_t)cqe->user_data;

	if (cqe->res < 0) {
		fprintf(stderr, "cqe->res=%d (%s) for slot at offset %" PRIu64
			"\n", cqe->res, strerror(-cqe->res), s->file_offset);
		p->err = -EIO;
		s->in_use = 0;
		goto done;
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
done:
	p->outstanding -= 1;
	p->completed += 1;
}

static struct slot *
try_free_slot(struct progress *p)
{
	for (uint32_t i = 0; i < p->n_slots; i++) {
		struct slot *s = &p->slots[i];

		if (!s->in_use) {
			return s;
		}
#ifdef WITH_CUDA
		if (p->use_cuda && cudaEventQuery(s->event) == cudaSuccess) {
			s->in_use = 0;
			return s;
		}
#endif
	}
	return NULL;
}

int
main(int argc, char *argv[])
{
	const char *file_path = NULL;
	uint32_t qd = 64;
	uint64_t io_size = 1ULL << 20;
	int use_direct = 1;
	struct io_uring ring;
	int ring_initialised = 0;
	struct progress p = {0};
	int fd = -1;
	int rc = 1;

	setvbuf(stdout, NULL, _IONBF, 0);

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--file") && i + 1 < argc)
			file_path = argv[++i];
		else if (!strcmp(argv[i], "--qd") && i + 1 < argc)
			qd = (uint32_t)atoi(argv[++i]);
		else if (!strcmp(argv[i], "--io-size") && i + 1 < argc)
			io_size = strtoull(argv[++i], NULL, 0);
		else if (!strcmp(argv[i], "--no-direct"))
			use_direct = 0;
	}
	if (!file_path) {
		fprintf(stderr,
			"usage: %s --file <path> [--qd <N>] [--io-size <bytes>] "
			"[--no-direct]\n",
			argv[0]);
		return 1;
	}

	int oflags = O_RDONLY | (use_direct ? O_DIRECT : 0);

	fd = open(file_path, oflags);
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
	uint64_t buf_size = (file_size + io_size - 1) & ~(io_size - 1);
	uint64_t total_chunks = (file_size + io_size - 1) / io_size;

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
		if (posix_memalign((void **)&p.slots[i].host_buf, 4096,
				   io_size)) {
			perror("posix_memalign");
			goto out;
		}
	}
#endif

	int err = io_uring_queue_init(qd, &ring, 0);

	if (err) {
		fprintf(stderr, "io_uring_queue_init: %s\n", strerror(-err));
		goto out;
	}
	ring_initialised = 1;

	printf("file=%s size=%.2f MiB io_size=%" PRIu64 " KiB qd=%u "
	       "n_slots=%u direct=%d\n",
	       file_path, (double)file_size / (1024.0 * 1024.0),
	       io_size / 1024, qd, p.n_slots, use_direct);

	uint64_t offset = 0;
	uint64_t t0 = now_ns();

	while (p.completed < total_chunks) {
		int submitted_this_round = 0;

		while (offset < file_size && p.outstanding < qd) {
			struct slot *s = try_free_slot(&p);

			if (!s) {
				break;
			}
			struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);

			if (!sqe) {
				break;
			}
			s->in_use = 1;
			s->file_offset = offset;
			s->size = io_size;

			io_uring_prep_read(sqe, fd, s->host_buf, io_size,
					   offset);
			io_uring_sqe_set_data(sqe, s);

			offset += io_size;
			p.outstanding += 1;
			p.submitted += 1;
			submitted_this_round += 1;
		}
		if (submitted_this_round) {
			io_uring_submit(&ring);
		}

		if (p.outstanding == 0 && offset >= file_size) {
			break;
		}

		struct io_uring_cqe *cqe = NULL;

		if (p.outstanding) {
			err = io_uring_wait_cqe(&ring, &cqe);
			if (err) {
				fprintf(stderr, "io_uring_wait_cqe: %s\n",
					strerror(-err));
				goto out;
			}
			handle_cqe(cqe, &p);
			io_uring_cqe_seen(&ring, cqe);

			while (io_uring_peek_cqe(&ring, &cqe) == 0) {
				handle_cqe(cqe, &p);
				io_uring_cqe_seen(&ring, cqe);
			}
		}
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
	if (ring_initialised)
		io_uring_queue_exit(&ring);
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
				free(p.slots[i].host_buf);
		}
	}
#endif
	if (p.slots)
		free(p.slots);
	if (fd >= 0)
		close(fd);
	return rc;
}
