// GPU-resident SIMT kernel tests (design §13): each device kernel compared
// against the oracle (tests/oracle/math — the ground-truth host reference).

#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include "oracle/math.hpp"
#include "alice/tensor.hpp"
#include "../src/blocks/gpu_rocm/kernels.hpp"
#include "../src/gpu/kernels.hpp"

namespace {

namespace om = oracle;
namespace gk = alice::gpu::kernels;
namespace gbk = alice::gpu_rocm::kernels;
using alice::bf16;


template <typename T>
T* dev_upload(const std::vector<T>& host) {
  T* p = nullptr;
  if (hipMalloc(&p, host.size() * sizeof(T)) != hipSuccess) throw std::bad_alloc();
  if (hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice) !=
      hipSuccess)
    throw std::runtime_error("H2D failed");
  return p;
}

template <typename T>
std::vector<T> dev_download(const T* p, size_t n) {
  std::vector<T> host(n);
  if (hipMemcpy(host.data(), p, n * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess)
    throw std::runtime_error("D2H failed");
  return host;
}

void dev_free(void* p) { (void)hipFree(p); }

std::vector<float> pattern(size_t n, uint32_t seed) {  // exact-representable
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = float(int(((i * 7 + seed * 13) % 17)) - 8) * 0.03125f;
  return v;
}

TEST(GpuKernels, F32ToBf16MatchesFromF32) {
  auto host = pattern(1000, 1);
  float* d_in = dev_upload(host);
  unsigned short* d_out = nullptr;
  ASSERT_EQ(hipMalloc(&d_out, host.size() * 2), hipSuccess);
  gk::f32_to_bf16(0, d_in, d_out, host.size());
  auto got = dev_download(d_out, host.size());
  for (size_t i = 0; i < host.size(); ++i)
    EXPECT_EQ(got[i], bf16::from_f32(host[i]).bits) << "element " << i;
  dev_free(d_in);
  dev_free(d_out);
}

TEST(GpuKernels, RmsnormMatchesOracle) {
  const uint32_t S = 5, d = 300;  // non-multiple of block size on purpose
  auto x = pattern(size_t(S) * d, 2);
  auto g = pattern(d, 3);
  std::vector<float> oracle = x;
  om::rmsnorm_inplace(oracle.data(), S, d, g.data(), 1e-5f);
  float* d_in = dev_upload(x);
  float* d_g = dev_upload(g);
  float* d_out = nullptr;
  ASSERT_EQ(hipMalloc(&d_out, x.size() * 4), hipSuccess);
  gk::rmsnorm(0, d_in, d_g, d_out, S, d, 1e-5f);
  auto got = dev_download(d_out, x.size());
  for (size_t i = 0; i < x.size(); ++i) EXPECT_NEAR(got[i], oracle[i], 1e-5f) << i;
  dev_free(d_in);
  dev_free(d_g);
  dev_free(d_out);
}

TEST(GpuKernels, AddRowsMatchesOracle) {
  auto a = pattern(777, 4), b = pattern(777, 5);
  std::vector<float> oracle = a;
  om::add_rows(oracle.data(), b.data(), 1, 777);
  float* d_a = dev_upload(a);
  float* d_b = dev_upload(b);
  gk::add_rows(0, d_a, d_b, 777);
  auto got = dev_download(d_a, 777);
  for (size_t i = 0; i < 777; ++i) EXPECT_FLOAT_EQ(got[i], oracle[i]) << i;
  dev_free(d_a);
  dev_free(d_b);
}

TEST(GpuKernels, EmbedGatherMatchesHost) {
  const uint32_t vocab = 50, d = 64;
  auto tab = pattern(size_t(vocab) * d, 6);
  std::vector<bf16> tab_b(tab.size());
  for (size_t i = 0; i < tab.size(); ++i) tab_b[i] = bf16::from_f32(tab[i]);
  std::vector<int32_t> ids = {0, 49, 7, 7, 33};
  bf16* d_tab = dev_upload(tab_b);
  int32_t* d_ids = dev_upload(ids);
  float* d_out = nullptr;
  ASSERT_EQ(hipMalloc(&d_out, ids.size() * d * 4), hipSuccess);
  gk::embed_gather(0, d_tab, d_ids, d_out, uint32_t(ids.size()), d);
  auto got = dev_download(d_out, ids.size() * d);
  for (size_t s = 0; s < ids.size(); ++s)
    for (uint32_t i = 0; i < d; ++i)
      EXPECT_FLOAT_EQ(got[s * d + i], tab_b[size_t(ids[s]) * d + i].to_f32()) << s << "," << i;
  dev_free(d_tab);
  dev_free(d_ids);
  dev_free(d_out);
}

TEST(GpuKernels, SiluMulMatchesHost) {
  const uint32_t S = 3, I = 100;
  // gu as the GEMM writes it: col-major [2I, S] == row-major [S, 2I].
  auto gu = pattern(size_t(2) * I * S, 7);
  float* d_gu = dev_upload(gu);
  unsigned short* d_act = nullptr;
  ASSERT_EQ(hipMalloc(&d_act, size_t(S) * I * 2), hipSuccess);
  gbk::silu_mul(0, d_gu, d_act, S, I);
  auto got = dev_download(d_act, size_t(S) * I);
  for (uint32_t s = 0; s < S; ++s)
    for (uint32_t i = 0; i < I; ++i) {
      const float g = gu[size_t(s) * 2 * I + i], u = gu[size_t(s) * 2 * I + I + i];
      const float want = bf16::from_f32(om::silu(g) * u).to_f32();
      EXPECT_NEAR(bf16{got[size_t(s) * I + i]}.to_f32(), want, 1e-3f) << s << "," << i;
    }
  dev_free(d_gu);
  dev_free(d_act);
}

TEST(GpuKernels, ArgmaxFindsMaxAndBreaksTiesLow) {
  std::vector<float> x(1000, 1.0f);
  x[511] = 2.0f;
  x[13] = 2.0f;  // tie: lowest index wins
  x[999] = 1.999f;
  float* d_x = dev_upload(x);
  int32_t* d_out = nullptr;
  ASSERT_EQ(hipMalloc(&d_out, 4), hipSuccess);
  gk::argmax(0, d_x, x.size(), d_out);
  auto got = dev_download(d_out, 1);
  EXPECT_EQ(got[0], 13);
  dev_free(d_x);
  dev_free(d_out);
}

TEST(GpuKernels, RopeAppendMatchesOracle) {
  const uint32_t nh = 2, nkv = 1, hd = 4, S = 3, pos0 = 2, max_seq = 8;
  const uint32_t qd = nh * hd, kvd = nkv * hd, qkvd = qd + 2 * kvd, half = hd / 2;
  auto qkv = pattern(size_t(S) * qkvd, 8);
  std::vector<float> oracle_qkv = qkv;
  std::vector<float> cos_t(max_seq * half), sin_t(max_seq * half);
  om::build_rope_tables(cos_t.data(), sin_t.data(), max_seq, hd, 10000.0f);

  // oracle: rotate q/k in place, then memcpy rows into caches
  std::vector<float> okc(size_t(max_seq) * kvd, 0.0f), ovc(size_t(max_seq) * kvd, 0.0f);
  for (uint32_t s = 0; s < S; ++s) {
    const uint32_t pos = pos0 + s;
    for (uint32_t h = 0; h < nh; ++h)
      om::rope_rotate(oracle_qkv.data() + size_t(s) * qkvd + h * hd, hd,
                      cos_t.data() + size_t(pos) * half, sin_t.data() + size_t(pos) * half);
    for (uint32_t g = 0; g < nkv; ++g)
      om::rope_rotate(oracle_qkv.data() + size_t(s) * qkvd + qd + g * hd, hd,
                      cos_t.data() + size_t(pos) * half, sin_t.data() + size_t(pos) * half);
    std::memcpy(okc.data() + size_t(pos) * kvd, oracle_qkv.data() + size_t(s) * qkvd + qd,
                kvd * sizeof(float));
    std::memcpy(ovc.data() + size_t(pos) * kvd,
                oracle_qkv.data() + size_t(s) * qkvd + qd + kvd, kvd * sizeof(float));
  }

  float* d_qkv = dev_upload(qkv);
  float* d_cos = dev_upload(cos_t);
  float* d_sin = dev_upload(sin_t);
  float *d_kc = nullptr, *d_vc = nullptr;
  ASSERT_EQ(hipMalloc(&d_kc, okc.size() * 4), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_vc, ovc.size() * 4), hipSuccess);
  ASSERT_EQ(hipMemset(d_kc, 0, okc.size() * 4), hipSuccess);
  ASSERT_EQ(hipMemset(d_vc, 0, ovc.size() * 4), hipSuccess);
  gbk::rope_append(0, d_qkv, d_kc, d_vc, d_cos, d_sin, pos0, S, nh, nkv, hd);
  auto got_q = dev_download(d_qkv, qkv.size());
  auto got_kc = dev_download(d_kc, okc.size());
  auto got_vc = dev_download(d_vc, ovc.size());
  // q region is rotated in place; k/v regions of qkv are dead after append
  // (the block reads k/v from the caches, compared below).
  for (uint32_t s = 0; s < S; ++s)
    for (uint32_t e = 0; e < qd; ++e)
      EXPECT_NEAR(got_q[size_t(s) * qkvd + e], oracle_qkv[size_t(s) * qkvd + e], 1e-6f)
          << s << "," << e;
  for (size_t i = 0; i < okc.size(); ++i) EXPECT_NEAR(got_kc[i], okc[i], 1e-6f) << i;
  for (size_t i = 0; i < ovc.size(); ++i) EXPECT_NEAR(got_vc[i], ovc[i], 1e-6f) << i;
  dev_free(d_qkv);
  dev_free(d_cos);
  dev_free(d_sin);
  dev_free(d_kc);
  dev_free(d_vc);
}

TEST(GpuKernels, DecodeGraphFixedGridMatchesDynamicAttentionGeometry) {
  constexpr uint32_t nh = 2, nkv = 1, max_seq = 4160;
  const std::vector<uint32_t> lengths{1, 2, 63, 64, 65, 127, 128, 129,
                                      4095, 4096, 4097, 4159};
  for (uint32_t hd : {64u, 128u}) {
    float* qkv = dev_upload(pattern(size_t(nh + 2 * nkv) * hd, 31));
    float* keys = dev_upload(pattern(size_t(max_seq) * nkv * hd, 32));
    float* values = dev_upload(pattern(size_t(max_seq) * nkv * hd, 33));
    uint32_t* position = dev_upload(std::vector<uint32_t>{0});
    const size_t scratch_bytes = size_t(gbk::kAttnMaxSplits) * nh * (hd + 2) * 4;
    float* scratch = nullptr;
    unsigned short *eager = nullptr, *replay = nullptr;
    ASSERT_EQ(hipMalloc(&scratch, scratch_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&eager, size_t(nh) * hd * 2), hipSuccess);
    ASSERT_EQ(hipMalloc(&replay, size_t(nh) * hd * 2), hipSuccess);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    const float scale = 1.0f / std::sqrt(float(hd));
    ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal), hipSuccess);
    gbk::attention_decode(stream, qkv, keys, values, replay, scratch, position,
                           nh, nkv, hd, scale);
    ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    ASSERT_EQ(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0), hipSuccess);
    for (uint32_t T : lengths) {
      const uint32_t pos = T - 1;
      gbk::attention(stream, qkv, keys, values, eager, scratch, nullptr, pos, 1,
                     nh, nkv, hd, scale);
      // Poison all partials: replay must not read inactive fixed-grid slots.
      ASSERT_EQ(hipMemsetAsync(scratch, 0xff, scratch_bytes, stream), hipSuccess);
      ASSERT_EQ(hipMemcpyAsync(position, &pos, sizeof(pos), hipMemcpyHostToDevice,
                               stream), hipSuccess);
      ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess);
      ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
      const auto want = dev_download(eager, size_t(nh) * hd);
      const auto got = dev_download(replay, size_t(nh) * hd);
      EXPECT_EQ(got, want) << "head dim " << hd << ", context " << T;
    }
    EXPECT_EQ(hipGraphExecDestroy(exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    dev_free(qkv); dev_free(keys); dev_free(values); dev_free(position);
    dev_free(scratch); dev_free(eager); dev_free(replay);
  }
}

}  // namespace
