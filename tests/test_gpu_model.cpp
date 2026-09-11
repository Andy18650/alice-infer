// GPU-resident model tests (design §13): determinism, and the teacher-forced
// canonical gate vs HF fixtures on the device-resident path.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "alice/engine_config.hpp"
#include "alice/gpu_model.hpp"

namespace {

constexpr float kMaxLogitDrift = 0.75f;  // the canonical gate (design §13)

std::string fixture(const std::string& dir, const std::string& name) {
  return std::string(ALICE_SOURCE_DIR) + "/tests/" + dir + "/" + name;
}

bool have_gpu_model(const std::string& dir, const std::string& fixture_dir) {
  return alice::gpu::resident_available() && std::filesystem::exists(dir) &&
         std::filesystem::exists(fixture(fixture_dir, "case_0.json"));
}

void run_teacher_forced(alice::gpu::Model& model, const std::string& fixture_dir,
                        uint32_t chunk, int cases) {
  for (int ci = 0; ci < cases; ++ci) {
    std::ifstream min(fixture_dir + "/case_" + std::to_string(ci) + ".json");
    const auto meta = nlohmann::json::parse(
        std::string((std::istreambuf_iterator<char>(min)), std::istreambuf_iterator<char>()));
    auto seq = meta.at("input_ids").get<std::vector<int32_t>>();
    for (const int32_t t : meta.at("generated_ids").get<std::vector<int32_t>>())
      seq.push_back(t);
    const int seq_len = meta.at("seq_len").get<int>();
    const int vocab = meta.at("vocab").get<int>();
    ASSERT_EQ(int(seq.size()), seq_len);

    std::vector<float> ref(size_t(seq_len) * vocab);
    std::ifstream bin(fixture_dir + "/case_" + std::to_string(ci) + ".teacher.f32",
                      std::ios::binary);
    bin.read(reinterpret_cast<char*>(ref.data()), ref.size() * sizeof(float));
    ASSERT_EQ(size_t(bin.gcount()), ref.size() * sizeof(float));

    model.reset();
    float max_drift = 0.0f;
    double sum_drift = 0.0;
    int worst_pos = -1;
    for (size_t off = 0; off < seq.size(); off += chunk) {
      const size_t n = std::min<size_t>(chunk, seq.size() - off);
      const auto logits =
          model.forward_logits_all(std::span<const int32_t>(seq.data() + off, n));
      ASSERT_EQ(logits.size(), n * size_t(vocab));
      for (size_t r = 0; r < n; ++r) {
        const size_t pos = off + r;
        const float* got = logits.data() + r * vocab;
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
    printf("%s case %d: %d positions, max|Δlogit|=%.4f (pos %d), mean|Δ|=%.5f\n",
           fixture_dir.c_str(), ci, seq_len, max_drift, worst_pos,
           sum_drift / (double(seq_len) * vocab));
    fflush(stdout);
    EXPECT_LT(max_drift, kMaxLogitDrift) << "case " << ci << ": logit drift beyond tolerance";
  }
}

TEST(GpuModel, DeterminismBitExact) {
  const std::string dir = "/opt/models/Llama-3.2-1B-Instruct";
  if (!alice::gpu::resident_available() || !std::filesystem::exists(dir))
    GTEST_SKIP() << "no GPU or 1B model";
  alice::EngineConfig cfg;
  cfg.max_seq_len = 512;
  auto model = alice::gpu::Model::load(
      dir, std::string(ALICE_SOURCE_DIR) + "/models/llama-3.2-1b-instruct.manifest.json", cfg);
  const std::vector<int32_t> tokens = {128000, 791, 6864, 315, 9822, 374};
  auto a = model->forward_logits(tokens);
  model->reset();
  auto b = model->forward_logits(tokens);
  ASSERT_EQ(a.size(), b.size());
  EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
      << "same tokens + fresh state produced different logits";
}

TEST(GpuModel, Canonical1BTeacherForced) {
  const std::string dir = "/opt/models/Llama-3.2-1B-Instruct";
  if (!have_gpu_model(dir, "canonical"))
    GTEST_SKIP() << "no GPU, 1B model, or canonical fixtures";
  alice::EngineConfig cfg;
  cfg.max_seq_len = 4096;
  auto model = alice::gpu::Model::load(
      dir, std::string(ALICE_SOURCE_DIR) + "/models/llama-3.2-1b-instruct.manifest.json", cfg);
  run_teacher_forced(*model, fixture("canonical", ""), cfg.max_prefill_chunk, 3);
}

TEST(GpuModel, Canonical8BTeacherForced) {
  const std::string dir = "/opt/models/deepseek-ai/deepseek-r1-distill-llama-8b";
  if (!have_gpu_model(dir, "canonical_8b"))
    GTEST_SKIP() << "no GPU, 8B model, or canonical fixtures";
  alice::EngineConfig cfg;
  cfg.max_seq_len = 4096;
  auto model = alice::gpu::Model::load(
      dir, std::string(ALICE_SOURCE_DIR) + "/models/deepseek-r1-distill-llama-8b.manifest.json", cfg);
  run_teacher_forced(*model, fixture("canonical_8b", ""), cfg.max_prefill_chunk, 3);
}

// Ad-hoc model validation without recompile (mirrors CanonicalEnv for the
// host path): ALICE_CANONICAL_{MODEL_DIR,MANIFEST,FIXTURES[,CHUNK]}.
TEST(GpuModel, CanonicalEnvTeacherForced) {
  const char* dir = getenv("ALICE_CANONICAL_MODEL_DIR");
  const char* man = getenv("ALICE_CANONICAL_MANIFEST");
  const char* fix = getenv("ALICE_CANONICAL_FIXTURES");
  if (!dir || !man || !fix) GTEST_SKIP() << "ALICE_CANONICAL_* not set";
  if (!alice::gpu::resident_available()) GTEST_SKIP() << "no GPU";
  alice::EngineConfig cfg;
  cfg.max_seq_len = 8192;
  const char* chunk_env = getenv("ALICE_CANONICAL_CHUNK");
  if (chunk_env) cfg.max_prefill_chunk = uint32_t(atoi(chunk_env));
  auto model = alice::gpu::Model::load(dir, man, cfg);
  run_teacher_forced(*model, fix, cfg.max_prefill_chunk, 3);
}

}  // namespace
