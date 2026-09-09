// alice_bench_ffn — per-block microbenchmark for DenseFfnBF16 implementations.
//
// Times forward() directly on realistic shapes (default: the bring-up 1B's
// d=2048 / inter=8192) with deterministic synthetic weights — no model load,
// no tokenizer, no attention in the measurement. Exists so kernel changes
// get precise per-block before/after numbers (and as the pattern for future
// per-operator benches; attention stays naive until its turn, design §10).
//
// usage: alice_bench_ffn [hw=all|cpu_generic|cpu_avx2] [iters_s1=100] [iters_pre=3] [S_pre=128]
// Thread count via OMP_NUM_THREADS; NUMA via numactl wrapper.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "alice/dense_ffn_bf16.hpp"
#include "alice/engine_config.hpp"
#include "alice/manifest.hpp"

namespace {

using alice::BF16View;
using alice::F32View;

struct Bf16Buf {
  std::vector<alice::bf16> data;
  BF16View view;
  Bf16Buf(uint32_t r, uint32_t c, uint32_t seed) : data(size_t(r) * c) {
    for (size_t i = 0; i < data.size(); ++i)  // bf16-exact pattern
      data[i] = alice::bf16::from_f32(float(int((i * 7 + seed * 13) % 17) - 8) * 0.03125f);
    view = BF16View{data.data(), r, c, 0};
  }
};

double bench(alice::DenseFfnBF16& ffn, F32View io, int iters) {
  for (int k = 0; k < 3; ++k) ffn.forward(io);  // warmup (threads, pages, caches)
  const auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < iters; ++k) ffn.forward(io);
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count() / double(iters);
}

void report(const char* hw, uint32_t S, double secs, uint32_t d, uint32_t inter) {
  const double weight_bytes = 3.0 * d * inter * sizeof(alice::bf16);  // streamed once
  const double flops = 2.0 * 3.0 * S * d * inter;
  std::printf("%-12s S=%-4u %8.3f ms/call | %7.1f GB/s weights | %8.1f GFLOP/s\n", hw, S,
              secs * 1e3, weight_bytes / secs / 1e9, flops / secs / 1e9);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string hw = argc > 1 ? argv[1] : "all";
  const int iters_s1 = argc > 2 ? std::atoi(argv[2]) : 100;
  const int iters_pre = argc > 3 ? std::atoi(argv[3]) : 3;
  const uint32_t s_pre = argc > 4 ? uint32_t(std::atoi(argv[4])) : 128;

  alice::ModelConfig mc;
  mc.hidden_size = 2048;
  mc.intermediate_size = 8192;
  alice::EngineConfig ec;
  ec.max_prefill_chunk = 512;

  const bool want_generic = hw == "all" || hw == "cpu_generic";
  const bool want_avx2 = hw == "all" || hw == "cpu_avx2";

  std::vector<float> xbuf(size_t(ec.max_prefill_chunk) * mc.hidden_size);
  for (size_t i = 0; i < xbuf.size(); ++i)
    xbuf[i] = float(int((i * 11 + 5) % 23) - 11) * 0.015625f;

  for (const bool is_avx2 : {false, true}) {
    if (is_avx2 && !want_avx2) continue;
    if (!is_avx2 && !want_generic) continue;
    const alice::HardwareClass cls =
        is_avx2 ? alice::HardwareClass::CPU_AVX2 : alice::HardwareClass::CPU_GENERIC;
    const char* name = is_avx2 ? "cpu_avx2" : "cpu_generic";
    Bf16Buf gate(mc.intermediate_size, mc.hidden_size, 1);
    Bf16Buf up(mc.intermediate_size, mc.hidden_size, 2);
    Bf16Buf down(mc.hidden_size, mc.intermediate_size, 3);
    std::unique_ptr<alice::DenseFfnBF16> ffn;
    try {
      ffn = alice::DenseFfnBF16::create(cls, gate.view, up.view, down.view, mc, ec);
    } catch (const std::exception& e) {
      std::printf("%-12s skipped: %s\n", name, e.what());
      continue;
    }
    {
      F32View io{xbuf.data(), 1, mc.hidden_size, 0};
      report(name, 1, bench(*ffn, io, iters_s1), mc.hidden_size, mc.intermediate_size);
    }
    {
      F32View io{xbuf.data(), s_pre, mc.hidden_size, 0};
      report(name, s_pre, bench(*ffn, io, iters_pre), mc.hidden_size, mc.intermediate_size);
    }
  }
  return 0;
}
