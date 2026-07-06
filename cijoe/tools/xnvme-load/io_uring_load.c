/*
 * io_uring_load: minimal async file reader using liburing directly,
 * no xNVMe involvement. Same shape and CLI flags as xnvme-load so we
 * can compare against the xnvme-based path on the same file.
 *
 * Used to bracket where the ceiling actually is: if this tool hits
 * the drive's read bandwidth but xnvme-load does not, the wrapper is
 * the variable.
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
	uint32_t qd = 64;
	uint64_t io_size = 1ULL << 20;
	int use_direct = 1;

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
			"usage: %s --file <path> [--qd <N>] [--io-size <bytes>] [--no-direct]\n",
			argv[0]);
		return 1;
	}

	int oflags = O_RDONLY | (use_direct ? O_DIRECT : 0);
	int fd = open(file_path, oflags);

	if (fd < 0) {
		perror("open");
		return 1;
	}

	struct stat st;

	if (fstat(fd, &st) < 0) {
		perror("fstat");
		close(fd);
		return 1;
	}

	uint64_t file_size = (uint64_t)st.st_size;
	uint64_t buf_size = (file_size + io_size - 1) & ~(io_size - 1);
	void *dst = NULL;

	if (posix_memalign(&dst, 4096, buf_size)) {
		perror("posix_memalign");
		close(fd);
		return 1;
	}

	printf("file=%s size=%.2f MiB io_size=%" PRIu64
	       " KiB qd=%u direct=%d\n",
	       file_path, (double)file_size / (1024.0 * 1024.0),
	       io_size / 1024, qd, use_direct);

	struct io_uring ring;
	int err = io_uring_queue_init(qd, &ring, 0);

	if (err) {
		fprintf(stderr, "io_uring_queue_init: %s\n", strerror(-err));
		free(dst);
		close(fd);
		return 1;
	}

	uint64_t offset = 0;
	uint64_t outstanding = 0;
	uint64_t submitted = 0;
	uint64_t completed = 0;
	int had_err = 0;

	uint64_t t0 = now_ns();

	uint64_t total_chunks = (file_size + io_size - 1) / io_size;

	while (completed < total_chunks) {
		// Always submit io_size per SQE, even at EOF. O_DIRECT
		// requires the READ LENGTH to be sector-aligned; a short
		// tail read (e.g. 845 KiB when io_size is 1 MiB) fails with
		// EINVAL. The kernel gracefully returns a short cqe->res for
		// the last chunk when offset+io_size crosses EOF, and the
		// destination buffer is already padded to io_size alignment.
		while (offset < file_size && outstanding < qd) {
			struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);

			if (!sqe) {
				break;
			}
			io_uring_prep_read(sqe, fd, (char *)dst + offset,
					   io_size, offset);
			offset += io_size;
			outstanding += 1;
			submitted += 1;
		}
		io_uring_submit(&ring);

		struct io_uring_cqe *cqe;
		err = io_uring_wait_cqe(&ring, &cqe);
		if (err) {
			fprintf(stderr, "io_uring_wait_cqe: %s\n",
				strerror(-err));
			had_err = 1;
			break;
		}
		if (cqe->res < 0) {
			fprintf(stderr, "read cqe: %s\n", strerror(-cqe->res));
			had_err = 1;
			io_uring_cqe_seen(&ring, cqe);
			break;
		}
		io_uring_cqe_seen(&ring, cqe);
		outstanding -= 1;
		completed += 1;
	}

	uint64_t t1 = now_ns();
	double sec = (double)(t1 - t0) / 1e9;
	double mib = (double)file_size / (1024.0 * 1024.0);

	printf("read %.2f MiB in %.3f s -> %.1f MiB/s (%.2f GB/s) target=host, "
	       "%" PRIu64 " cmds\n",
	       mib, sec, mib / sec, mib / sec / 1024.0, submitted);

#ifdef WITH_CUDA
	void *dev_buf = NULL;
	cudaError_t cerr = cudaMalloc(&dev_buf, buf_size);

	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(cerr));
		had_err = 1;
		goto cleanup;
	}
	cerr = cudaMemcpy(dev_buf, dst, file_size, cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy: %s\n", cudaGetErrorString(cerr));
		cudaFree(dev_buf);
		had_err = 1;
		goto cleanup;
	}
	cudaDeviceSynchronize();

	uint64_t t2 = now_ns();
	double sec_vram = (double)(t2 - t0) / 1e9;

	printf("read %.2f MiB in %.3f s -> %.1f MiB/s (%.2f GB/s) target=vram\n",
	       mib, sec_vram, mib / sec_vram, mib / sec_vram / 1024.0);

	cudaFree(dev_buf);

cleanup:
#endif
	{
		struct rusage ru = {0};

		if (getrusage(RUSAGE_SELF, &ru) == 0) {
			printf("max_rss=%ld KiB (%.2f MiB)\n", ru.ru_maxrss,
			       (double)ru.ru_maxrss / 1024.0);
		}
	}

	io_uring_queue_exit(&ring);
	free(dst);
	close(fd);
	return had_err ? 1 : 0;
}
