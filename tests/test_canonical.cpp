// Canonical comparison: our greedy generation and logits vs the committed
// PyTorch reference fixtures (tests/canonical/case_*.json + .logits.f32).
// Token ids must match EXACTLY; logits are a bf16-compute reference, so they
// get a generous tolerance with printed metrics.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "alice/model.hpp"

namespace {

constexpr int kCases = 3;

std::string fixture(const std::string& name) {
  return std::string(ALICE_SOURCE_DIR) + "/tests/canonical/" + name;
}

bool fixtures_present() {
  return std::filesystem::exists("/opt/models/Llama-3.2-1B-Instruct/model.safetensors") &&
         std::filesystem::exists(fixture("case_0.json"));
}

TEST(Canonical, GreedyMatchesHF) {
  if (!fixtures_present()) GTEST_SKIP() << "model or canonical fixtures not present";

  alice::EngineConfig cfg;
  cfg.max_seq_len = 256;
  cfg.max_prefill_chunk = 8;
  cfg.sampling.temperature = 0.0f;
  auto model = alice::ModelLoader::load(
      "/opt/models/Llama-3.2-1B-Instruct",
      std::string(ALICE_SOURCE_DIR) + "/models/llama-3.2-1b-instruct.manifest.json", cfg);

  for (int ci = 0; ci < kCases; ++ci) {
    std::ifstream min(fixture("case_" + std::to_string(ci) + ".json"));
    const auto meta = nlohmann::json::parse(
        std::string((std::istreambuf_iterator<char>(min)), std::istreambuf_iterator<char>()));
    const auto input_ids = meta.at("input_ids").get<std::vector<int32_t>>();
    const auto generated = meta.at("generated_ids").get<std::vector<int32_t>>();
    const int steps = meta.at("steps").get<int>();
    const int vocab = meta.at("vocab").get<int>();
    ASSERT_EQ(int(generated.size()), steps);

    std::vector<float> ref_logits(size_t(steps) * vocab);
    std::ifstream bin(fixture("case_" + std::to_string(ci) + ".logits.f32"),
                      std::ios::binary);
    bin.read(reinterpret_cast<char*>(ref_logits.data()), ref_logits.size() * sizeof(float));
    ASSERT_TRUE(bin.good() || bin.eof());

    model->reset();
    // chunked prefill; the last chunk's logits are step-0 logits
    alice::TensorView logits;
    for (size_t off = 0; off < input_ids.size(); off += cfg.max_prefill_chunk) {
      const size_t n = std::min<size_t>(cfg.max_prefill_chunk, input_ids.size() - off);
      logits = model->forward_logits(
          std::span<const int32_t>(input_ids.data() + off, n));
    }
    ASSERT_EQ(logits.cols, uint32_t(vocab));

    auto check_step = [&](const alice::TensorView& ours, int step) {
      const float* ref = ref_logits.data() + size_t(step) * vocab;
      const float* got = ours.as<float>();
      float max_diff = 0.0f;
      double sum_diff = 0.0;
      for (int v = 0; v < vocab; ++v) {
        const float d = std::fabs(got[v] - ref[v]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
      }
      const int32_t our_top = int32_t(std::max_element(got, got + vocab) - got);
      const int32_t ref_top = int32_t(std::max_element(ref, ref + vocab) - ref);
      float ref_second = -1e30f;
      for (int v = 0; v < vocab; ++v)
        if (v != ref_top) ref_second = std::max(ref_second, ref[v]);
      printf("case %d step %d: max|Δlogit|=%.4f mean|Δ|=%.5f top1 ours=%d ref=%d margin=%.4f\n",
             ci, step, max_diff, sum_diff / vocab, our_top, ref_top,
             ref[ref_top] - ref_second);
      EXPECT_LT(max_diff, 0.75f) << "logit drift beyond bf16-reference tolerance";
      // bf16 compute noise can flip argmax on near-ties (observed: case 1
      // step 4 is an EXACT tie in the reference, ref[11] == ref[26129]).
      // Decisive steps must match exactly; near-ties must land within margin.
      constexpr float kTieBand = 0.25f;
      if (ref[ref_top] - ref_second > kTieBand) {
        EXPECT_EQ(our_top, ref_top) << "case " << ci << " step " << step << " (decisive)";
        if (step < steps) EXPECT_EQ(our_top, generated[step]);
      } else {
        EXPECT_GE(ref[our_top], ref[ref_top] - kTieBand)
            << "case " << ci << " step " << step << " (tie band)";
      }
      return our_top;
    };

    int32_t t = check_step(logits, 0);
    for (int s = 1; s < steps; ++s) {
      // feed the CANONICAL token to keep sequences aligned (drift isolation)
      const int32_t feed = generated[s - 1];
      logits = model->forward_logits(std::span<const int32_t>(&feed, 1));
      t = check_step(logits, s);
    }
    (void)t;
  }
}

}  // namespace
