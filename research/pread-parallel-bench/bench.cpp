// Parallel pread bandwidth benchmark.
//
// Usage: pread-parallel-bench <file> <threads>
//
// Divides the file evenly across N threads. Each thread allocates an
// anonymous MADV_HUGEPAGE buffer and preads its chunk into it, then reports
// bytes read and elapsed time. An overall wall-clock time and bandwidth are
// also reported.

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

struct ThreadResult {
  size_t bytes = 0;
  double ms = 0.0;
};

[[noreturn]] void fail(const std::string& what) {
  std::fprintf(stderr, "error: %s: %s\n", what.c_str(), std::strerror(errno));
  std::exit(1);
}

ThreadResult worker(int fd, off_t offset, size_t size) {
  void* buf =
      mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
           -1, 0);
  if (buf == MAP_FAILED) fail("mmap failed");
  // Best-effort THP; not fatal if the kernel ignores it.
  (void)madvise(buf, size, MADV_HUGEPAGE);

  auto t0 = Clock::now();
  char* p = static_cast<char*>(buf);
  size_t got = 0;
  while (got < size) {
    ssize_t n =
        pread(fd, p + got, size - got, offset + static_cast<off_t>(got));
    if (n <= 0) fail("pread failed");
    got += static_cast<size_t>(n);
  }
  auto t1 = Clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  (void)munmap(buf, size);

  return {got, ms};
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <file> <threads>\n", argv[0]);
    return 1;
  }
  const char* path = argv[1];
  int nthreads = std::atoi(argv[2]);
  if (nthreads <= 0) fail("threads must be > 0");

  int fd = open(path, O_RDONLY);
  if (fd < 0) fail("open failed");

  off_t fsize_off = lseek(fd, 0, SEEK_END);
  if (fsize_off < 0) fail("lseek failed");
  if (lseek(fd, 0, SEEK_SET) < 0) fail("lseek rewind failed");
  const size_t total = static_cast<size_t>(fsize_off);

  const size_t base = total / static_cast<size_t>(nthreads);
  const size_t rem = total % static_cast<size_t>(nthreads);

  std::printf("file:   %s\n", path);
  std::printf("size:   %.2f GiB (%zu bytes)\n",
              static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0), total);
  std::printf("threads: %d\n", nthreads);
  std::printf("chunk:  ~%.2f MiB per thread\n\n",
              static_cast<double>(base) / (1024.0 * 1024.0));

  std::vector<std::thread> threads;
  threads.reserve(nthreads);
  std::vector<ThreadResult> results(static_cast<size_t>(nthreads));

  auto t0 = Clock::now();
  off_t offset = 0;
  for (int i = 0; i < nthreads; ++i) {
    size_t size = base + (static_cast<size_t>(i) < rem ? 1 : 0);
    threads.emplace_back([&, i, offset, size]() {
      results[static_cast<size_t>(i)] = worker(fd, offset, size);
    });
    offset += static_cast<off_t>(size);
  }
  for (auto& t : threads) t.join();
  auto t1 = Clock::now();

  for (int i = 0; i < nthreads; ++i) {
    const auto& r = results[static_cast<size_t>(i)];
    double bw_gbs =
        (static_cast<double>(r.bytes) / (1024.0 * 1024.0 * 1024.0)) /
        (r.ms / 1000.0);
    std::printf("thread %2d: %8.2f MiB in %8.2f ms -> %6.2f GB/s\n", i,
                static_cast<double>(r.bytes) / (1024.0 * 1024.0), r.ms,
                bw_gbs);
  }

  double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double total_bw_gbs =
      (static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0)) /
      (total_ms / 1000.0);
  std::printf("\noverall:   %8.2f GiB in %8.2f ms -> %6.2f GB/s\n",
              static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0),
              total_ms, total_bw_gbs);

  (void)close(fd);
  return 0;
}
