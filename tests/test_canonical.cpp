// Canonical comparison, teacher-forced: the reference sequence (prompt +
// canonical greedy continuation) is replayed through our engine and the
// LOGIT VECTORS at every position are compared — no argmax assertions, so
// reference near-ties are irrelevant by construction.
//
// Logits are a bf16-compute reference (ours is fp32 compute with bf16
// weights), so the comparison is within-tolerance with printed metrics.

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
constexpr float kMaxLogitDrift = 0.75f;  // generous; observed ~0.1-0.2

std::string fixture(const std::string& name) {
  return std::string(ALICE_SOURCE_DIR) + "/tests/canonical/" + name;
}

bool fixtures_present() {
  return std::filesystem::exists("/opt/models/Llama-3.2-1B-Instruct/model.safetensors") &&
         std::filesystem::exists(fixture("case_0.json"));
}

TEST(Canonical, TeacherForcedLogitsMatchHF) {
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
    auto seq = meta.at("input_ids").get<std::vector<int32_t>>();
    for (const int32_t t : meta.at("generated_ids").get<std::vector<int32_t>>())
      seq.push_back(t);
    const int seq_len = meta.at("seq_len").get<int>();
    const int vocab = meta.at("vocab").get<int>();
    ASSERT_EQ(int(seq.size()), seq_len);

    std::vector<float> ref(size_t(seq_len) * vocab);
    std::ifstream bin(fixture("case_" + std::to_string(ci) + ".teacher.f32"),
                      std::ios::binary);
    bin.read(reinterpret_cast<char*>(ref.data()), ref.size() * sizeof(float));
    ASSERT_EQ(size_t(bin.gcount()), ref.size() * sizeof(float));

    model->reset();
    float max_drift = 0.0f;
    double sum_drift = 0.0;
    int worst_pos = -1;
    for (size_t off = 0; off < seq.size(); off += cfg.max_prefill_chunk) {
      const size_t n = std::min<size_t>(cfg.max_prefill_chunk, seq.size() - off);
      const alice::TensorView logits =
          model->forward_logits_all(std::span<const int32_t>(seq.data() + off, n));
      ASSERT_EQ(logits.rows, uint32_t(n));
      ASSERT_EQ(logits.cols, uint32_t(vocab));
      for (size_t r = 0; r < n; ++r) {
        const size_t pos = off + r;
        const float* got = logits.as<float>() + r * logits.stride();
        const float* want = ref.data() + pos * vocab;
        for (int v = 0; v < vocab; ++v) {
          const float d = std::fabs(got[v] - want[v]);
          if (d > max_drift) {
            max_drift = d;
            worst_pos = int(pos);
          }
          sum_drift += d;
        }
      }
    }
    printf("case %d: %d positions, max|Δlogit|=%.4f (pos %d), mean|Δ|=%.5f\n", ci,
           seq_len, max_drift, worst_pos, sum_drift / (double(seq_len) * vocab));
    EXPECT_LT(max_drift, kMaxLogitDrift) << "case " << ci << ": logit drift beyond tolerance";
  }
}

}  // namespace
