// GPU-resident block structural tests: stateful behaviour (incremental == batched,
// reset freshness, position tracking) that have no HuggingFace equivalent.
//
// Numerical correctness against a canonical implementation is covered by
// test_hf_block_oracle.cpp, which compares alice::gpu_rocm blocks against
// the exact LlamaAttention / LlamaMLP submodules from HuggingFace.

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "../src/blocks/gpu_rocm/gqa_attention.hpp"
#include "../src/gpu/model.hpp"

namespace {

using alice::bf16;
using alice::EngineConfig;
using alice::gpu::Resources;
using alice::block::GqaAttentionConfig;
using alice::Tensor;
using alice::TensorView;

struct Bf16Mat {
  std::vector<bf16> data;
  uint32_t rows, cols;
  Bf16Mat(uint32_t r, uint32_t c, const std::vector<float>& vals)
      : data(vals.size()), rows(r), cols(c) {
    for (size_t i = 0; i < vals.size(); ++i) data[i] = bf16::from_f32(vals[i]);
  }
  Tensor<bf16> t() const { return Tensor<bf16>::copy_of(data.data(), rows, cols); }
  TensorView<bf16> v() const { return TensorView<bf16>(data.data(), rows, cols); }
};

float pat(size_t r, size_t c) { return float((int((r * 7 + c * 13) % 17) - 8)) / 8.0f; }

std::vector<float> mat_vals(uint32_t r, uint32_t c) {
  std::vector<float> v(size_t(r) * c);
  for (uint32_t i = 0; i < r; ++i)
    for (uint32_t j = 0; j < c; ++j) v[i * c + j] = pat(i, j);
  return v;
}

size_t prefill_scratch_bytes(const GqaAttentionConfig& cfg, const EngineConfig& ec) {
  return size_t(cfg.num_q_heads) * ec.max_prefill_chunk *
         ((ec.max_seq_len + 63) / 64) * (cfg.qk_dim + 2) * 4;
}

std::vector<float> run_gpu_block(alice::block::Block& b, const std::vector<float>& io,
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
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabsf(a[i] - b[i]));
  return m;
}

// ---------------------------------------------------------------------------
// Attention structural tests
// ---------------------------------------------------------------------------
TEST(GpuBlocks, AttentionIncrementalEqualsBatched) {
  Resources res(0);
  GqaAttentionConfig cfg;
  cfg.io_dim = 8;
  cfg.qk_dim = 4;
  cfg.v_dim = 4;
  cfg.num_q_heads = 2;
  cfg.num_kv_heads = 1;
  cfg.rope_theta = 10000.0;
  EngineConfig ec;
  ec.max_seq_len = 8;
  ec.max_prefill_chunk = 4;
  res.init_scratch(prefill_scratch_bytes(cfg, ec));
  auto mk = [&]() {
    Bf16Mat* wq = new Bf16Mat(8, 8, mat_vals(8, 8));
    Bf16Mat* wk = new Bf16Mat(4, 8, mat_vals(9, 9));
    Bf16Mat* wv = new Bf16Mat(4, 8, mat_vals(10, 10));
    Bf16Mat* wo = new Bf16Mat(8, 8, mat_vals(11, 11));
    auto b = std::make_unique<alice::gpu_rocm::GqaAttentionBF16>(
        wq->v(), wk->v(), wv->v(), wo->v(), cfg, ec, res);
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
  EXPECT_LT(max_abs_diff(batched, part1), 1e-4f)
      << "batched diverged from incremental (tolerance-based)";
}

TEST(GpuBlocks, AttentionResetRestartsPosition) {
  GqaAttentionConfig cfg;
  cfg.io_dim = 8;
  cfg.qk_dim = 4;
  cfg.v_dim = 4;
  cfg.num_q_heads = 2;
  cfg.num_kv_heads = 1;
  cfg.rope_theta = 10000.0;
  EngineConfig ec;
  ec.max_seq_len = 8;
  ec.max_prefill_chunk = 4;
  Bf16Mat wq(8, 8, mat_vals(8, 8)), wk(4, 8, mat_vals(9, 9)), wv(4, 8, mat_vals(10, 10)),
      wo(8, 8, mat_vals(11, 11));
  Resources res(0);
  res.init_scratch(prefill_scratch_bytes(cfg, ec));
  alice::gpu_rocm::GqaAttentionBF16 b(wq.v(), wk.v(), wv.v(), wo.v(), cfg, ec, res);
  auto first = run_gpu_block(b, mat_vals(3, 8), 3, 8);
  EXPECT_EQ(b.position(), 3u);
  b.reset_state();
  EXPECT_EQ(b.position(), 0u);
  auto again = run_gpu_block(b, mat_vals(3, 8), 3, 8);
  EXPECT_LT(max_abs_diff(first, again), 1e-4f)
      << "reset did not restore initial state";
}

}  // namespace
