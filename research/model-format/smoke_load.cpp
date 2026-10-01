// Smoke harness for the alice model format loader (research tooling, not
// ctest): loads a packed .model via init_engine (the new path) and the same
// model via the manifest + safetensors path (the proven path), then runs
// both forward_logits() on identical tokens and compares bit-exactly.
//
// This is the cross-check that the model-format path produces a Model
// equivalent to the manifest path: same weights (content-addressed, same
// bytes), same component constructors, same compute -> same logits. Any
// divergence in packer layout, factory construction, member loading, or the
// arena/scratch path shows up as a logits mismatch.
//
// Usage: smoke_load <packed_dir> <hf_model_dir> <manifest.json> <max_seq> <mpc>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <vector>

#include "../../src/gpu/gpu_model.hpp"
#include "../../src/model_format.hpp"

using namespace alice;

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

  // --- new path: model-format (init_engine) ---------------------------------
  modelfmt::LoadStats st;
  modelfmt::Engine eng = modelfmt::init_engine(packed / "model.model", ec, &st);
  std::printf(
      "[smoke] format path: %zu blocks (%zu specs), %zu hash verifications, "
      "scratch %.2f MiB\n",
      st.blocks, st.specs, st.hash_verifications,
      double(st.scratch_bytes) / (1 << 20));

  // --- old path: manifest + safetensors (Model::load) -----------------------
  auto ref = gpu::Model::load(hf_dir, manifest_path, ec);

  // --- compare whole-model forward_logits on identical tokens ----------------
  // ids 1..S (all < vocab; the values are irrelevant — same ids -> same logits,
  // so any divergence is a load-path difference, not a token-value effect).
  std::vector<int32_t> tokens(S);
  std::iota(tokens.begin(), tokens.end(), int32_t(1));

  auto lf = eng.model->forward_logits(tokens);
  auto lr = ref->forward_logits(tokens);

  if (lf.size() != lr.size()) {
    std::fprintf(stderr,
                 "[smoke] FAIL: logits length mismatch (format %zu vs manifest %zu)\n",
                 lf.size(), lr.size());
    return 1;
  }
  if (std::memcmp(lf.data(), lr.data(), lf.size() * sizeof(float)) != 0) {
    size_t first = 0;
    while (first < lf.size() && lf[first] == lr[first]) ++first;
    std::fprintf(stderr,
                 "[smoke] FAIL: logits diverge at element %zu (format %g vs manifest %g)\n",
                 first, lf[first], lr[first]);
    return 1;
  }
  std::printf("[smoke] PASS: forward_logits bit-exact, format vs manifest "
              "(S=%u, %zu logits)\n", S, lf.size());
  return 0;
}
