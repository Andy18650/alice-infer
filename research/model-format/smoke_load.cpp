// Smoke harness for the alice model format loader (research tooling, not
// ctest): loads a packed .model with the new loader, then constructs the
// same blocks through the OLD proven path (manifest + safetensors staging)
// and compares forward() outputs bit-exactly on identical input. This is
// the cross-check that packer layout == factory layout == old-path bytes,
// and that the allocator's scratch (NaN-poisoned) is written before read.
//
// Usage: smoke_load <packed_dir> <hf_model_dir> <manifest.json> <max_seq> <mpc>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "../../src/manifest.hpp"
#include "../../src/model_format.hpp"
#include "../../src/safetensors.hpp"
#include "blocks/block.hpp"
#include "gpu/resources.hpp"
#include "blocks/gpu_rocm/dense_ffn.hpp"
#include "blocks/gpu_rocm/gqa_attention.hpp"
#include "loader_staging.hpp"

using namespace alice;
using gpu::Resources;

#define HIP_CHECK(x)                                                            \
  do {                                                                          \
    hipError_t e_ = (x);                                                        \
    if (e_ != hipSuccess) {                                                     \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),    \
                   __FILE__, __LINE__);                                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <packed_dir> <hf_model_dir> <manifest.json> "
                         "<max_seq> <mpc>\n", argv[0]);
    return 2;
  }
  const std::filesystem::path packed = argv[1];
  const std::string hf_dir = argv[2];
  const std::string manifest_path = argv[3];
  EngineConfig ec;
  ec.max_seq_len = uint32_t(std::stoul(argv[4]));
  ec.max_prefill_chunk = uint32_t(std::stoul(argv[5]));
  const uint32_t S = 16;  // > 4: exercises the v4 prefill path + shared scratch

  Resources res(0);
  auto t0 = std::chrono::steady_clock::now();

  // --- new path ---------------------------------------------------------------
  modelfmt::LoadStats st;
  modelfmt::LoadedBlocks lb =
      modelfmt::load_blocks(packed / "model.model", res, ec, &st);
  const double load_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  std::printf(
      "[smoke] loaded %zu blocks (%zu specs) from %s\n"
      "[smoke]   %.3f s total: parse %.1f ms, read %.1f ms (%.2f GB/s), "
      "hash %.1f ms (%.2f GB/s), construct %.1f ms\n"
      "[smoke]   scratch %.2f MiB, %zu hash verifications\n",
      st.blocks, st.specs, packed.c_str(), load_s, st.parse_ms, st.read_ms,
      st.bytes_read / st.read_ms / 1e6, st.hash_ms,
      st.bytes_read / st.hash_ms / 1e6, st.construct_ms,
      double(st.scratch_bytes) / (1 << 20), st.hash_verifications);

  // --- old path (manifest + safetensors): the reference ------------------------
  const Manifest m = Manifest::parse(manifest_path);
  auto files = safetensors::Files::open({hf_dir + "/model.safetensors"});
  const ModelConfig& mc = m.config;

  // Reference blocks reuse res.scratch() (init'd by load_blocks from the
  // factory scratch_bytes, same formula); no separate ref scratch.

  // io buffer [S, d] fp32 with a deterministic pattern.
  std::vector<float> host_io(size_t(S) * mc.hidden_size);
  std::mt19937 rng(7);
  for (auto& v : host_io) v = float(rng()) / float(rng.max()) * 2.f - 1.f;
  float *io_a = nullptr, *io_b = nullptr;
  HIP_CHECK(hipMalloc((void**)&io_a, host_io.size() * 4));
  HIP_CHECK(hipMalloc((void**)&io_b, host_io.size() * 4));

  size_t compared = 0;
  for (size_t i = 0; i < lb.blocks.size() && i < m.blocks.size(); ++i) {
    const auto& bs = m.blocks[i];
    std::unique_ptr<block::Block> ref;
    if (bs.type == "gqa_attention_bf16") {
      auto wq = loader::stage<bf16>(files, bs.weights.at("wq"));
      auto wk = loader::stage<bf16>(files, bs.weights.at("wk"));
      auto wv = loader::stage<bf16>(files, bs.weights.at("wv"));
      auto wo = loader::stage<bf16>(files, bs.weights.at("wo"));
      ref = std::make_unique<gpu_rocm::GqaAttentionBF16>(wq.view(), wk.view(),
                                                         wv.view(), wo.view(),
                                                         mc, ec, res);
    } else {
      auto g = loader::stage<bf16>(files, bs.weights.at("w_gate"));
      auto u = loader::stage<bf16>(files, bs.weights.at("w_up"));
      auto dn = loader::stage<bf16>(files, bs.weights.at("w_down"));
      ref = std::make_unique<gpu_rocm::DenseFfnBF16>(g.view(), u.view(), dn.view(),
                                                     mc, ec, res);
    }

    HIP_CHECK(hipMemcpy(io_a, host_io.data(), host_io.size() * 4,
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(io_b, host_io.data(), host_io.size() * 4,
                        hipMemcpyHostToDevice));
    lb.blocks[i]->forward(io_a, S);  // fresh blocks: position 0, same schedule
    ref->forward(io_b, S);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> a(host_io.size()), b(host_io.size());
    HIP_CHECK(hipMemcpy(a.data(), io_a, a.size() * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(b.data(), io_b, b.size() * 4, hipMemcpyDeviceToHost));
    if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
      size_t first = 0;
      while (first < a.size() && a[first] == b[first]) ++first;
      std::fprintf(stderr,
                   "[smoke] FAIL block %zu (%s): outputs diverge at element %zu "
                   "(new %g vs ref %g)\n",
                   i, bs.type.c_str(), first, a[first], b[first]);
      return 1;
    }
    ++compared;
  }

  std::printf("[smoke] PASS: %zu/%zu blocks bit-exact vs the safetensors path "
              "(S=%u, %s scratch path)\n",
              compared, lb.blocks.size(), S,
              mc.num_heads ? "v4 prefill" : "?");
  HIP_CHECK(hipFree(io_a));
  HIP_CHECK(hipFree(io_b));
  return 0;
}
