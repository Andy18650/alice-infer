// CPU_GENERIC oracle tests: naive kernels, FFN wiring, and the attention
// block's core properties (incremental == batched, causality, reset, bounds).

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "alice/dense_ffn_bf16.hpp"
#include "alice/engine_config.hpp"
#include "alice/gqa_attention_bf16.hpp"
#include "alice/manifest.hpp"

// reach into the naive kernels directly (they are the functional tier)
#include "../src/blocks/cpu_generic/kernels.hpp"

namespace {

using alice::bf16;
using alice::BF16View;
using alice::EngineConfig;
using alice::F32View;
using alice::HardwareClass;
using alice::ModelConfig;
namespace cg = alice::cpu_generic;

struct Bf16Mat {
  std::vector<bf16> data;
  BF16View view;
  Bf16Mat(uint32_t r, uint32_t c, const std::vector<float>& vals)
      : data(vals.size()), view{nullptr, r, c, 0} {
    for (size_t i = 0; i < vals.size(); ++i) data[i] = bf16::from_f32(vals[i]);
    view.data = data.data();
  }
};

struct F32Mat {
  std::vector<float> data;
  F32View view;
  F32Mat(uint32_t r, uint32_t c, std::vector<float> vals)
      : data(std::move(vals)), view{nullptr, r, c, 0} {
    view.data = data.data();
  }
  F32Mat(uint32_t r, uint32_t c) : F32Mat(r, c, std::vector<float>(size_t(r) * c, 0.0f)) {}
};

TEST(Kernels, MatmulWt) {
  F32Mat in(2, 3, {1, 2, 3, 4, 5, 6});
  Bf16Mat w(2, 3, {1, 0, 0, 0, 1, 0});
  F32Mat out(2, 2);
  cg::matmul_wt(in.view, w.view, out.view);
  EXPECT_FLOAT_EQ(out.data[0], 1.0f);
  EXPECT_FLOAT_EQ(out.data[1], 2.0f);
  EXPECT_FLOAT_EQ(out.data[2], 4.0f);
  EXPECT_FLOAT_EQ(out.data[3], 5.0f);
}

TEST(Kernels, Rmsnorm) {
  F32Mat x(1, 2, {3, 4});
  float g[2] = {1, 1};
  cg::rmsnorm_inplace(x.view, g, 0.0f);
  const float rms = std::sqrt(12.5f);
  EXPECT_NEAR(x.data[0], 3.0f / rms, 1e-6);
  EXPECT_NEAR(x.data[1], 4.0f / rms, 1e-6);
}

TEST(Kernels, RopeTablesPlain) {
  const uint32_t max_seq = 4, hd = 64, half = hd / 2;
  std::vector<float> c(max_seq * half), s(max_seq * half);
  alice::RopeScaling rs;
  cg::build_rope_tables(c.data(), s.data(), max_seq, hd, 500000.0f, false, rs);
  for (uint32_t i = 0; i < half; ++i) {  // position 0: cos=1, sin=0
    EXPECT_FLOAT_EQ(c[i], 1.0f);
    EXPECT_FLOAT_EQ(s[i], 0.0f);
  }
  EXPECT_NEAR(s[half + 0], std::sin(1.0), 1e-6);  // inv_freq[0] == 1
}

TEST(Kernels, RopeTablesLlama3Scaling) {
  const uint32_t max_seq = 4, hd = 64, half = hd / 2;
  std::vector<float> c(max_seq * half), s(max_seq * half);
  alice::RopeScaling rs;
  rs.factor = 32.0f;
  rs.low_freq_factor = 1.0f;
  rs.high_freq_factor = 4.0f;
  rs.original_max_position_embeddings = 8192;
  cg::build_rope_tables(c.data(), s.data(), max_seq, hd, 500000.0f, true, rs);
  // reference values from the HF llama3 formula (computed offline):
  EXPECT_NEAR(s[half + 0], 8.41470985e-01, 1e-6);   // high band: unchanged
  EXPECT_NEAR(s[half + 17], 9.70828780e-05, 1e-9);  // medium band: smoothed
  EXPECT_NEAR(s[half + 31], 9.41830673e-08, 1e-12); // low band: /32
}

TEST(Kernels, RopeRotate) {
  float x[4] = {1, 0, 2, 0};
  const float c[2] = {0.6f, 1.0f}, s[2] = {0.8f, 0.0f};
  cg::rope_rotate(x, 4, c, s);
  EXPECT_FLOAT_EQ(x[0], 1.0f * 0.6f - 2.0f * 0.8f);
  EXPECT_FLOAT_EQ(x[2], 2.0f * 0.6f + 1.0f * 0.8f);
  EXPECT_FLOAT_EQ(x[1], 0.0f);
  EXPECT_FLOAT_EQ(x[3], 0.0f);
}

// --- FFN -------------------------------------------------------------------

ModelConfig ffn_mc() {
  ModelConfig mc;
  mc.hidden_size = 2;
  mc.intermediate_size = 3;
  return mc;
}

TEST(DenseFfn, ForwardMatchesHandCompute) {
  Bf16Mat gate(3, 2, {1, 0, 0, 1, 1, 1});
  Bf16Mat up(3, 2, {1, 1, 0, 1, 1, 0});
  Bf16Mat down(2, 3, {1, 0, 1, 0, 1, 0});
  EngineConfig ec;
  ec.max_prefill_chunk = 2;
  auto ffn = alice::DenseFfnBF16::create(HardwareClass::CPU_GENERIC, gate.view, up.view,
                                         down.view, ffn_mc(), ec);

  F32Mat io(1, 2, {1, 2});
  ffn->forward(io.view);
  // gate=[1,2,3], up=[3,2,1], act=[silu(1)*3, silu(2)*2, silu(3)*3],
  // silu(x) = x*sigmoid(x): silu(1)=0.7310586, silu(2)=1.7615942, silu(3)=2.8577223
  const float a0 = 0.7310586f * 3, a1 = 1.7615942f * 2, a2 = 2.8577223f * 1;
  EXPECT_NEAR(io.data[0], a0 + a2, 1e-4);
  EXPECT_NEAR(io.data[1], a1, 1e-4);
}

TEST(DenseFfn, ChunkLimitHardError) {
  Bf16Mat gate(3, 2, {1, 0, 0, 1, 1, 1});
  Bf16Mat up(3, 2, {1, 1, 0, 1, 1, 0});
  Bf16Mat down(2, 3, {1, 0, 1, 0, 1, 0});
  EngineConfig ec;
  ec.max_prefill_chunk = 1;
  auto ffn = alice::DenseFfnBF16::create(HardwareClass::CPU_GENERIC, gate.view, up.view,
                                         down.view, ffn_mc(), ec);
  F32Mat io(2, 2, {1, 2, 3, 4});
  EXPECT_THROW(ffn->forward(io.view), std::runtime_error);
}

// --- CPU_AVX2 FFN ------------------------------------------------------------
//
// Numerics policy (user-signed-off, design §10): cross-impl equivalence is
// TOLERANCE-based (chunked accumulation reorders sums); within-impl
// determinism and incremental==batched stay BIT-EXACT (one fixed loop
// schedule for all S).

#if defined(__x86_64__) && defined(__GNUC__)
static bool HasAvx2() {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#else
static bool HasAvx2() { return false; }
#endif

// Multi-chunk dims: I = 600 = 2*256 + 88 forces the chunk loop + tail.
ModelConfig ffn_mc_600() {
  ModelConfig mc;
  mc.hidden_size = 8;
  mc.intermediate_size = 600;
  return mc;
}

std::vector<float> ffn_pattern(uint32_t r, uint32_t c, uint32_t seed) {
  std::vector<float> v(size_t(r) * c);
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = float(int((i * 7 + seed * 13) % 17) - 8) * 0.03125f;  // bf16-exact
  return v;
}

TEST(DenseFfnAvx2, ForwardMatchesHandCompute) {
  if (!HasAvx2()) GTEST_SKIP() << "no AVX2 on this machine";
  Bf16Mat gate(3, 2, {1, 0, 0, 1, 1, 1});
  Bf16Mat up(3, 2, {1, 1, 0, 1, 1, 0});
  Bf16Mat down(2, 3, {1, 0, 1, 0, 1, 0});
  EngineConfig ec;
  ec.max_prefill_chunk = 2;
  auto ffn = alice::DenseFfnBF16::create(HardwareClass::CPU_AVX2, gate.view, up.view,
                                         down.view, ffn_mc(), ec);
  F32Mat io(1, 2, {1, 2});
  ffn->forward(io.view);
  const float a0 = 0.7310586f * 3, a1 = 1.7615942f * 2, a2 = 2.8577223f * 1;
  EXPECT_NEAR(io.data[0], a0 + a2, 1e-4);
  EXPECT_NEAR(io.data[1], a1, 1e-4);
}

TEST(DenseFfnAvx2, ApproxVsGenericMultiChunk) {
  if (!HasAvx2()) GTEST_SKIP() << "no AVX2 on this machine";
  const auto mc = ffn_mc_600();
  EngineConfig ec;
  ec.max_prefill_chunk = 4;
  Bf16Mat g1(600, 8, ffn_pattern(600, 8, 1)), u1(600, 8, ffn_pattern(600, 8, 2)),
      d1(8, 600, ffn_pattern(8, 600, 3));
  auto oracle = alice::DenseFfnBF16::create(HardwareClass::CPU_GENERIC, g1.view, u1.view,
                                            d1.view, mc, ec);
  Bf16Mat g2(600, 8, ffn_pattern(600, 8, 1)), u2(600, 8, ffn_pattern(600, 8, 2)),
      d2(8, 600, ffn_pattern(8, 600, 3));
  auto avx2 = alice::DenseFfnBF16::create(HardwareClass::CPU_AVX2, g2.view, u2.view,
                                          d2.view, mc, ec);
  std::vector<float> input = ffn_pattern(4, 8, 4);
  F32Mat io1(4, 8, input), io2(4, 8, input);
  oracle->forward(io1.view);
  avx2->forward(io2.view);
  float max_diff = 0.0f;
  for (size_t i = 0; i < io1.data.size(); ++i) {
    const float d = std::fabs(io1.data[i] - io2.data[i]);
    max_diff = d > max_diff ? d : max_diff;
  }
  // Observed: ~1e-6 for these magnitudes. Threshold has ~100x margin.
  EXPECT_LT(max_diff, 1e-4f) << "max |generic - avx2| = " << max_diff;
}

TEST(DenseFfnAvx2, IncrementalEqualsBatchedBitExact) {
  if (!HasAvx2()) GTEST_SKIP() << "no AVX2 on this machine";
  const auto mc = ffn_mc_600();
  EngineConfig ec;
  ec.max_prefill_chunk = 4;
  Bf16Mat g(600, 8, ffn_pattern(600, 8, 1)), u(600, 8, ffn_pattern(600, 8, 2)),
      dn(8, 600, ffn_pattern(8, 600, 3));
  auto ffn = alice::DenseFfnBF16::create(HardwareClass::CPU_AVX2, g.view, u.view, dn.view,
                                         mc, ec);
  F32Mat batched(4, 8, ffn_pattern(4, 8, 4));
  ffn->forward(batched.view);
  F32Mat acc(4, 8);
  for (uint32_t s = 0; s < 4; ++s) {
    std::vector<float> row(8);
    for (uint32_t c = 0; c < 8; ++c)  // same formula as ffn_pattern(4, 8, 4)
      row[c] = float(int(((s * 8 + c) * 7 + 4 * 13) % 17) - 8) * 0.03125f;
    F32Mat one(1, 8, row);
    ffn->forward(one.view);
    std::copy(one.data.begin(), one.data.end(), acc.data.begin() + s * 8);
  }
  EXPECT_EQ(batched.data, acc.data);  // bit-exact within impl (fixed schedule)
}

TEST(DenseFfnAvx2, ChunkLimitHardError) {
  if (!HasAvx2()) GTEST_SKIP() << "no AVX2 on this machine";
  Bf16Mat g(600, 8, ffn_pattern(600, 8, 1)), u(600, 8, ffn_pattern(600, 8, 2)),
      dn(8, 600, ffn_pattern(8, 600, 3));
  EngineConfig ec;
  ec.max_prefill_chunk = 1;
  auto ffn = alice::DenseFfnBF16::create(HardwareClass::CPU_AVX2, g.view, u.view, dn.view,
                                         ffn_mc_600(), ec);
  F32Mat io(2, 8, std::vector<float>(16, 0.5f));
  EXPECT_THROW(ffn->forward(io.view), std::runtime_error);
}

// --- attention --------------------------------------------------------------

struct AttnFixture {
  ModelConfig mc;
  EngineConfig ec;
  AttnFixture() {
    mc.num_heads = 2;
    mc.num_kv_heads = 1;
    mc.head_dim = 4;
    mc.hidden_size = 8;
    mc.rope_theta = 10000.0f;
    mc.has_rope_scaling = false;
    ec.max_seq_len = 8;
    ec.max_prefill_chunk = 4;
  }
  // deterministic, exactly-representable pattern in [-1, 1]
  static float pat(size_t r, size_t c) { return float((int((r * 7 + c * 13) % 17) - 8)) / 8.0f; }
  static Bf16Mat w(uint32_t r, uint32_t c) {
    std::vector<float> v(size_t(r) * c);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j) v[i * c + j] = pat(i, j);
    return Bf16Mat(r, c, v);
  }
  // rows r0 .. r0+S-1 of an implicit token stream: element (r, c) = pat(r, c)
  static F32Mat rows(uint32_t r0, uint32_t S, uint32_t d) {
    std::vector<float> v(size_t(S) * d);
    for (size_t r = 0; r < S; ++r)
      for (size_t c = 0; c < d; ++c) v[r * d + c] = pat(r0 + r, c);
    return F32Mat(S, d, v);
  }
  std::unique_ptr<alice::GqaAttentionBF16> make(Bf16Mat& wq, Bf16Mat& wk, Bf16Mat& wv,
                                                Bf16Mat& wo) {
    return alice::GqaAttentionBF16::create(HardwareClass::CPU_GENERIC, wq.view, wk.view,
                                           wv.view, wo.view, mc, ec);
  }
};

TEST(Attention, IncrementalEqualsBatched) {
  AttnFixture f;
  auto wq = AttnFixture::w(8, 8), wk = AttnFixture::w(4, 8), wv = AttnFixture::w(4, 8),
       wo = AttnFixture::w(8, 8);

  auto a = f.make(wq, wk, wv, wo);
  auto io_a = AttnFixture::rows(0, 3, 8);
  a->forward(io_a.view);

  auto b = f.make(wq, wk, wv, wo);
  auto io_b1 = AttnFixture::rows(0, 2, 8);
  b->forward(io_b1.view);
  auto io_b2 = AttnFixture::rows(2, 1, 8);  // same absolute row 2 as the batched run
  b->forward(io_b2.view);

  const size_t d = 8;
  EXPECT_EQ(0, std::memcmp(io_b1.data.data(), io_a.data.data(), 2 * d * sizeof(float)))
      << "prefill[0:2] diverged from batched prefill";
  EXPECT_EQ(0, std::memcmp(io_b2.data.data(), io_a.data.data() + 2 * d, d * sizeof(float)))
      << "decode step diverged from batched prefill row 2";
}

TEST(Attention, ResetRestoresFreshBehavior) {
  AttnFixture f;
  auto wq = AttnFixture::w(8, 8), wk = AttnFixture::w(4, 8), wv = AttnFixture::w(4, 8),
       wo = AttnFixture::w(8, 8);
  auto a = f.make(wq, wk, wv, wo);

  auto io1 = AttnFixture::rows(0, 2, 8);
  a->forward(io1.view);
  const auto first = io1.data;

  auto io2 = AttnFixture::rows(4, 2, 8);  // dirty the state
  a->forward(io2.view);

  a->reset_state();
  auto io3 = AttnFixture::rows(0, 2, 8);
  a->forward(io3.view);
  EXPECT_EQ(0, std::memcmp(first.data(), io3.data.data(), first.size() * sizeof(float)));
}

TEST(Attention, CausalityPrefixStability) {
  AttnFixture f;
  auto wq = AttnFixture::w(8, 8), wk = AttnFixture::w(4, 8), wv = AttnFixture::w(4, 8),
       wo = AttnFixture::w(8, 8);
  auto a = f.make(wq, wk, wv, wo);
  auto io = AttnFixture::rows(0, 3, 8);
  a->forward(io.view);

  auto b = f.make(wq, wk, wv, wo);
  auto io1 = AttnFixture::rows(0, 1, 8);
  b->forward(io1.view);

  EXPECT_EQ(0, std::memcmp(io1.data.data(), io.data.data(), 8 * sizeof(float)))
      << "row 0 changed when future tokens were appended";
}

TEST(Attention, MaxSeqHardError) {
  AttnFixture f;
  f.ec.max_seq_len = 2;
  auto wq = AttnFixture::w(8, 8), wk = AttnFixture::w(4, 8), wv = AttnFixture::w(4, 8),
       wo = AttnFixture::w(8, 8);
  auto a = f.make(wq, wk, wv, wo);
  auto io = AttnFixture::rows(0, 2, 8);
  a->forward(io.view);
  auto io2 = AttnFixture::rows(2, 1, 8);
  EXPECT_THROW(a->forward(io2.view), std::runtime_error);
}

}  // namespace
