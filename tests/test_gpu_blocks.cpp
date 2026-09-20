// GPU-resident block tests (design §13): device blocks vs the CPU_GENERIC
// oracle blocks on toy shapes, plus incremental == batched state behavior.

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "../src/blocks/cpu_generic/dense_ffn.hpp"
#include "../src/blocks/cpu_generic/gqa_attention.hpp"
#include "../src/blocks/gpu_rocm/dense_ffn.hpp"
#include "../src/blocks/gpu_rocm/gqa_attention.hpp"
#include "alice/gpu_model.hpp"

namespace {

using alice::bf16;
using alice::EngineConfig;
using alice::gpu_rocm::Context;
using alice::ModelConfig;
using alice::Tensor;

// Host fixture data + a factory producing fresh owning Tensors per
// construction (blocks take staged weights by value).
struct Bf16Mat {
  std::vector<bf16> data;
  uint32_t rows, cols;
  Bf16Mat(uint32_t r, uint32_t c, const std::vector<float>& vals)
      : data(vals.size()), rows(r), cols(c) {
    for (size_t i = 0; i < vals.size(); ++i) data[i] = bf16::from_f32(vals[i]);
  }
  Tensor<bf16> t() const { return Tensor<bf16>::copy_of(data.data(), rows, cols); }
};

float pat(size_t r, size_t c) { return float((int((r * 7 + c * 13) % 17) - 8)) / 8.0f; }

std::vector<float> mat_vals(uint32_t r, uint32_t c) {
  std::vector<float> v(size_t(r) * c);
  for (uint32_t i = 0; i < r; ++i)
    for (uint32_t j = 0; j < c; ++j) v[i * c + j] = pat(i, j);
  return v;
}

// Allocate the shared v4 prefill partials scratch for a test config.
float* alloc_prefill_scratch(const ModelConfig& mc, const EngineConfig& ec) {
  float* p = nullptr;
  const size_t bytes = size_t(mc.num_heads) * ec.max_prefill_chunk *
                       ((ec.max_seq_len + 63) / 64) * (mc.head_dim + 2) * 4;
  if (hipMalloc(&p, bytes) != hipSuccess) throw std::bad_alloc();
  return p;
}

// Run a gpu::Block on host-staged io: H2D -> forward -> D2H.
std::vector<float> run_gpu_block(alice::gpu::Block& b, const std::vector<float>& io,
                                 uint32_t S, uint32_t d) {
  float* d_io = nullptr;
  if (hipMalloc(&d_io, io.size() * 4) != hipSuccess) throw std::bad_alloc();
  if (hipMemcpy(d_io, io.data(), io.size() * 4, hipMemcpyHostToDevice) != hipSuccess)
    throw std::runtime_error("H2D failed");
  b.forward(d_io, S);
  std::vector<float> out(io.size());
  if (hipMemcpy(out.data(), d_io, io.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
    throw std::runtime_error("D2H failed");
  (void)hipFree(d_io);
  return out;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

TEST(GpuBlocks, DenseFfnApproxVsGeneric) {
  ModelConfig mc;
  mc.hidden_size = 8;
  mc.intermediate_size = 600;
  EngineConfig ec;
  ec.max_prefill_chunk = 8;
  Bf16Mat g1(600, 8, mat_vals(600, 8)), u1(600, 8, mat_vals(601, 9)),
      d1(8, 600, mat_vals(602, 10));
  alice::cpu_generic::DenseFfnBF16CpuGeneric oracle(g1.t(), u1.t(), d1.t(), mc, ec);
  Bf16Mat g2(600, 8, mat_vals(600, 8)), u2(600, 8, mat_vals(601, 9)),
      d2(8, 600, mat_vals(602, 10));
  Context ctx(0);
  alice::gpu_rocm::DenseFfnBF16 gpu(g2.t(), u2.t(), d2.t(), mc, ec, ctx);

  std::vector<float> io = mat_vals(5, 8);
  oracle.forward(io.data(), 5);
  auto got = run_gpu_block(gpu, mat_vals(5, 8), 5, 8);
  // bf16 rounding of x and the SiLU intermediate; observed ~0.096 at these
  // magnitudes (600-wide reductions of O(1) pattern values).
  EXPECT_LT(max_abs_diff(io, got), 0.15f) << "max|generic - gpu|";
}

// S=1 exercises the split-K GEMV dispatch (design §15): same tolerance as
// the S>1 gemm_ex path, on the same toy weights.
TEST(GpuBlocks, DenseFfnDecodeVsGeneric) {
  ModelConfig mc;
  mc.hidden_size = 8;
  mc.intermediate_size = 600;
  EngineConfig ec;
  ec.max_prefill_chunk = 8;
  Bf16Mat g1(600, 8, mat_vals(600, 8)), u1(600, 8, mat_vals(601, 9)),
      d1(8, 600, mat_vals(602, 10));
  alice::cpu_generic::DenseFfnBF16CpuGeneric oracle(g1.t(), u1.t(), d1.t(), mc, ec);
  Bf16Mat g2(600, 8, mat_vals(600, 8)), u2(600, 8, mat_vals(601, 9)),
      d2(8, 600, mat_vals(602, 10));
  Context ctx(0);
  alice::gpu_rocm::DenseFfnBF16 gpu(g2.t(), u2.t(), d2.t(), mc, ec, ctx);

  std::vector<float> io = mat_vals(1, 8);
  oracle.forward(io.data(), 1);
  auto got = run_gpu_block(gpu, mat_vals(1, 8), 1, 8);
  EXPECT_LT(max_abs_diff(io, got), 0.15f) << "max|generic - gpu| at S=1";
}

TEST(GpuBlocks, AttentionApproxVsGeneric) {
  ModelConfig mc;
  mc.num_heads = 2;
  mc.num_kv_heads = 1;
  mc.head_dim = 4;
  mc.hidden_size = 8;
  mc.rope_theta = 10000.0f;
  mc.has_rope_scaling = false;
  EngineConfig ec;
  ec.max_seq_len = 8;
  ec.max_prefill_chunk = 4;
  Bf16Mat wq1(8, 8, mat_vals(8, 8)), wk1(4, 8, mat_vals(9, 9)), wv1(4, 8, mat_vals(10, 10)),
      wo1(8, 8, mat_vals(11, 11));
  alice::cpu_generic::GqaAttentionBF16CpuGeneric oracle(wq1.t(), wk1.t(), wv1.t(),
                                                        wo1.t(), mc, ec);
  Bf16Mat wq2(8, 8, mat_vals(8, 8)), wk2(4, 8, mat_vals(9, 9)), wv2(4, 8, mat_vals(10, 10)),
      wo2(8, 8, mat_vals(11, 11));
  Context ctx(0);
  alice::gpu_rocm::GqaAttentionBF16 gpu(wq2.t(), wk2.t(), wv2.t(), wo2.t(), mc, ec, ctx,
                                      alloc_prefill_scratch(mc, ec));

  std::vector<float> io = mat_vals(3, 8);
  oracle.forward(io.data(), 3);
  auto got = run_gpu_block(gpu, mat_vals(3, 8), 3, 8);
  // bf16 projections + online softmax; observed ~1e-3.
  EXPECT_LT(max_abs_diff(io, got), 0.05f) << "max|generic - gpu|";
}

TEST(GpuBlocks, AttentionIncrementalEqualsBatched) {
  Context ctx(0);
  ModelConfig mc;
  mc.num_heads = 2;
  mc.num_kv_heads = 1;
  mc.head_dim = 4;
  mc.hidden_size = 8;
  mc.rope_theta = 10000.0f;
  EngineConfig ec;
  ec.max_seq_len = 8;
  ec.max_prefill_chunk = 4;
  auto mk = [&]() {
    Bf16Mat* wq = new Bf16Mat(8, 8, mat_vals(8, 8));
    Bf16Mat* wk = new Bf16Mat(4, 8, mat_vals(9, 9));
    Bf16Mat* wv = new Bf16Mat(4, 8, mat_vals(10, 10));
    Bf16Mat* wo = new Bf16Mat(8, 8, mat_vals(11, 11));
    auto b = std::make_unique<alice::gpu_rocm::GqaAttentionBF16>(
        wq->t(), wk->t(), wv->t(), wo->t(), mc, ec, ctx, alloc_prefill_scratch(mc, ec));
    return std::tuple{
        std::unique_ptr<Bf16Mat>(wq), std::unique_ptr<Bf16Mat>(wk),
        std::unique_ptr<Bf16Mat>(wv), std::unique_ptr<Bf16Mat>(wo), std::move(b)};
  };
  auto [wq1, wk1, wv1, wo1, a] = mk();
  auto batched = run_gpu_block(*a, mat_vals(3, 8), 3, 8);

  auto [wq2, wk2, wv2, wo2, b] = mk();
  auto part1 = run_gpu_block(*b, mat_vals(2, 8), 2, 8);
  auto row2src = mat_vals(3, 8);
  std::vector<float> row2(row2src.begin() + 2 * 8, row2src.end());
  auto part2 = run_gpu_block(*b, row2, 1, 8);

  part1.insert(part1.end(), part2.begin(), part2.end());
  // Deterministic kernels: batched == incremental bit-exact.
  EXPECT_EQ(0.0f, max_abs_diff(batched, part1)) << "batched diverged from incremental";
}

// Flash-decoding split-T path (kernels.hip dispatch: S <= 4 and T >= 256):
// decode steps past the split threshold must match the CPU oracle.
TEST(GpuBlocks, AttentionSplitDecodeMatchesOracle) {
  ModelConfig mc;
  mc.num_heads = 4;
  mc.num_kv_heads = 2;
  mc.head_dim = 128;  // exercises the 2-elements-per-lane path
  mc.hidden_size = 256;
  mc.rope_theta = 10000.0f;
  mc.has_rope_scaling = false;
  EngineConfig ec;
  ec.max_seq_len = 512;
  ec.max_prefill_chunk = 512;
  const uint32_t d = 256, qd = 512, kvd = 256, T0 = 300, ND = 4;
  // Scale weights by 2^-4 (bf16-exact): unscaled patterns would make
  // attention scores ~+/-250, saturate softmax to near-one-hot, and amplify
  // cross-impl fp noise into winner flips (an ill-conditioned comparison,
  // not a correctness signal).
  auto scaled = [](std::vector<float> v) {
    for (auto& x : v) x *= 0.0625f;
    return v;
  };
  auto wv_vals = scaled(mat_vals(kvd, d));
  for (auto& x : wv_vals) x = -0.5f * x + 0.125f;  // distinct from wk
  Bf16Mat wq1(qd, d, scaled(mat_vals(qd, d))), wk1(kvd, d, scaled(mat_vals(kvd, d))),
      wv1(kvd, d, wv_vals), wo1(d, qd, scaled(mat_vals(d, qd)));
  alice::cpu_generic::GqaAttentionBF16CpuGeneric oracle(wq1.t(), wk1.t(), wv1.t(),
                                                        wo1.t(), mc, ec);
  Bf16Mat wq2(qd, d, scaled(mat_vals(qd, d))), wk2(kvd, d, scaled(mat_vals(kvd, d))),
      wv2(kvd, d, wv_vals), wo2(d, qd, scaled(mat_vals(d, qd)));
  Context ctx(0);
  alice::gpu_rocm::GqaAttentionBF16 gpu(wq2.t(), wk2.t(), wv2.t(), wo2.t(), mc, ec, ctx,
                                      alloc_prefill_scratch(mc, ec));

  const std::vector<float> rows = mat_vals(T0 + ND, d);
  std::vector<float> ref = rows;  // oracle: one batched run over all rows
  oracle.forward(ref.data(), T0 + ND);

  run_gpu_block(gpu, std::vector<float>(rows.begin(), rows.begin() + T0 * d), T0, d);
  for (uint32_t k = 0; k < ND; ++k) {  // decode at pos 300..303: split-T path
    std::vector<float> one(rows.begin() + (T0 + k) * d, rows.begin() + (T0 + k + 1) * d);
    auto got = run_gpu_block(gpu, one, 1, d);
    std::vector<float> want(ref.begin() + (T0 + k) * d, ref.begin() + (T0 + k + 1) * d);
    EXPECT_LT(max_abs_diff(want, got), 0.1f)
        << "split-T decode diverged at step " << k;
  }
}

// v4 split-T prefill path (dispatch: S > 4): chunked prefill outputs must
// match the CPU oracle, including continuation chunks (pos0 > 0).
TEST(GpuBlocks, AttentionPrefillSplitMatchesOracle) {
  ModelConfig mc;
  mc.num_heads = 4;
  mc.num_kv_heads = 2;
  mc.head_dim = 128;
  mc.hidden_size = 256;
  mc.rope_theta = 10000.0f;
  mc.has_rope_scaling = false;
  EngineConfig ec;
  ec.max_seq_len = 512;
  ec.max_prefill_chunk = 512;
  const uint32_t d = 256, qd = 512, kvd = 256, T0 = 300, T1 = 200;
  auto scaled = [](std::vector<float> v) {  // bf16-exact 2^-4; see split test
    for (auto& x : v) x *= 0.0625f;
    return v;
  };
  auto wv_vals = scaled(mat_vals(kvd, d));
  for (auto& x : wv_vals) x = -0.5f * x + 0.125f;
  Bf16Mat wq1(qd, d, scaled(mat_vals(qd, d))), wk1(kvd, d, scaled(mat_vals(kvd, d))),
      wv1(kvd, d, wv_vals), wo1(d, qd, scaled(mat_vals(d, qd)));
  alice::cpu_generic::GqaAttentionBF16CpuGeneric oracle(wq1.t(), wk1.t(), wv1.t(),
                                                        wo1.t(), mc, ec);
  Bf16Mat wq2(qd, d, scaled(mat_vals(qd, d))), wk2(kvd, d, scaled(mat_vals(kvd, d))),
      wv2(kvd, d, wv_vals), wo2(d, qd, scaled(mat_vals(d, qd)));
  Context ctx(0);
  alice::gpu_rocm::GqaAttentionBF16 gpu(wq2.t(), wk2.t(), wv2.t(), wo2.t(), mc, ec, ctx,
                                        alloc_prefill_scratch(mc, ec));

  const std::vector<float> rows = mat_vals(T0 + T1, d);
  std::vector<float> ref = rows;
  oracle.forward(ref.data(), T0 + T1);

  auto got0 = run_gpu_block(
      gpu, std::vector<float>(rows.begin(), rows.begin() + T0 * d), T0, d);
  EXPECT_LT(max_abs_diff(std::vector<float>(ref.begin(), ref.begin() + T0 * d), got0),
            0.1f)
      << "v4 prefill diverged (initial chunk)";
  auto got1 = run_gpu_block(gpu, std::vector<float>(rows.begin() + T0 * d, rows.end()),
                            T1, d);
  EXPECT_LT(max_abs_diff(std::vector<float>(ref.begin() + T0 * d, ref.end()), got1),
            0.1f)
      << "v4 prefill diverged (continuation chunk, pos0=300)";
}

TEST(GpuBlocks, AttentionResetRestartsPosition) {
  ModelConfig mc;
  mc.num_heads = 2;
  mc.num_kv_heads = 1;
  mc.head_dim = 4;
  mc.hidden_size = 8;
  mc.rope_theta = 10000.0f;
  EngineConfig ec;
  ec.max_seq_len = 8;
  ec.max_prefill_chunk = 4;
  Bf16Mat wq(8, 8, mat_vals(8, 8)), wk(4, 8, mat_vals(9, 9)), wv(4, 8, mat_vals(10, 10)),
      wo(8, 8, mat_vals(11, 11));
  Context ctx(0);
  alice::gpu_rocm::GqaAttentionBF16 b(wq.t(), wk.t(), wv.t(), wo.t(), mc, ec, ctx,
                                    alloc_prefill_scratch(mc, ec));
  auto first = run_gpu_block(b, mat_vals(3, 8), 3, 8);
  EXPECT_EQ(b.position(), 3u);
  b.reset_state();
  EXPECT_EQ(b.position(), 0u);
  auto again = run_gpu_block(b, mat_vals(3, 8), 3, 8);
  EXPECT_EQ(0.0f, max_abs_diff(first, again)) << "reset did not restore initial state";
}

}  // namespace
