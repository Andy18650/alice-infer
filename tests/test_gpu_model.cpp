// GPU-resident model tests (design §13): determinism, and the teacher-forced
// canonical gate vs HF fixtures on the device-resident path.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "alice/engine_config.hpp"
#include "../src/gpu/gpu_model.hpp"

namespace {

constexpr float kMaxLogitDrift = 0.75f;  // the canonical gate (design §13)

std::string fixture(const std::string& dir, const std::string& name) {
  return std::string(ALICE_SOURCE_DIR) + "/tests/" + dir + "/" + name;
}

bool have_gpu_model(const std::string& dir, const std::string& fixture_dir) {
  return std::filesystem::exists(dir) &&
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
  if (!std::filesystem::exists(dir))
    GTEST_SKIP() << "no 1B model";
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

// Long-sequence gate: one ~1500-token WikiText prompt per model — exercises
// chunked prefill across chunk boundaries (512+512+tail) AND the S=1 decode
// path at long context (greedy continuation must match HF token-for-token).
// Fixtures are regenerable (tests/python/dump_canonical.py --prompt-file)
// and NOT committed (777 MB each; see .gitignore) — skips when absent.
void run_long_case(const std::string& model_dir, const std::string& manifest,
                   const std::string& fixture_dir) {
  std::ifstream min(fixture_dir + "/case_0.json");
  const auto meta = nlohmann::json::parse(
      std::string((std::istreambuf_iterator<char>(min)), std::istreambuf_iterator<char>()));
  auto prompt = meta.at("input_ids").get<std::vector<int32_t>>();
  const auto generated = meta.at("generated_ids").get<std::vector<int32_t>>();
  const int vocab = meta.at("vocab").get<int>();
  auto seq = prompt;
  for (const int32_t t : generated) seq.push_back(t);
  const int seq_len = meta.at("seq_len").get<int>();
  ASSERT_EQ(int(seq.size()), seq_len);

  std::vector<float> ref(size_t(seq_len) * vocab);
  std::ifstream bin(fixture_dir + "/case_0.teacher.f32", std::ios::binary);
  bin.read(reinterpret_cast<char*>(ref.data()), ref.size() * sizeof(float));
  ASSERT_EQ(size_t(bin.gcount()), ref.size() * sizeof(float));

  alice::EngineConfig cfg;
  cfg.max_seq_len = 8192;
  auto model = alice::gpu::Model::load(model_dir, manifest, cfg);

  // (a) teacher-forced, chunked across boundaries
  model->reset();
  float max_drift = 0.0f;
  double sum_drift = 0.0;
  int worst_pos = -1;
  const uint32_t chunk = cfg.max_prefill_chunk;
  for (size_t off = 0; off < seq.size(); off += chunk) {
    const size_t n = std::min<size_t>(chunk, seq.size() - off);
    const auto logits =
        model->forward_logits_all(std::span<const int32_t>(seq.data() + off, n));
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
  const double mean_drift = sum_drift / (double(seq_len) * vocab);
  printf("%s long case: %d positions, max|Δlogit|=%.4f (pos %d), mean|Δ|=%.5f\n",
         fixture_dir.c_str(), seq_len, max_drift, worst_pos, mean_drift);
  fflush(stdout);
  // Long-case gate (user-signed 2026-09, fp32-ideal reference): the mean
  // bounds systematic error; the max bounds the bf16 GEMM-input quantization
  // budget at massive-activation positions. Observed: 1B 0.76 max / 0.018
  // mean, 8B 1.52 max / 0.018 mean. The CPU oracle is exact against this
  // reference (0.0002) — the GPU drift is bf16-input error, nothing else.
  EXPECT_LT(max_drift, 2.0f) << "long case: bf16-input drift beyond budget";
  EXPECT_LT(mean_drift, 0.05) << "long case: systematic drift beyond budget";

  // (b) greedy decode from the prompt: token ids must match HF exactly
  model->reset();
  int32_t t = -1;
  for (size_t off = 0; off < prompt.size(); off += chunk) {
    const size_t n = std::min<size_t>(chunk, prompt.size() - off);
    t = model->forward(std::span<const int32_t>(prompt.data() + off, n));
  }
  for (size_t i = 0; i < generated.size(); ++i) {
    if (t != generated[i]) {
      // Coin-flip rule (user-signed 2026-09): a divergence is a bug only if
      // OUR token is decisively worse in the fp32 reference; near-ties flip
      // legitimately within the bf16-input noise budget.
      const float* row = ref.data() + (prompt.size() + i - 1) * vocab;
      float m1 = -INFINITY;
      for (int v = 0; v < vocab; ++v) m1 = std::max(m1, row[v]);
      const float deficit = m1 - row[t];
      EXPECT_LT(deficit, 0.25f)
          << "greedy diverged at step " << i << " (ours " << t << " vs HF "
          << generated[i] << ") and our token is " << deficit
          << " below the reference argmax — not a near-tie";
      printf("  greedy: tolerated near-tie flip at step %zu (our token %.4f "
             "below ref argmax)\n",
             i, deficit);
      break;  // sequences fork after a flip; stop comparing
    }
    t = model->forward(std::span<const int32_t>(&t, 1));
  }
}

TEST(GpuModel, CanonicalLong1B) {
  const std::string dir = "/opt/models/Llama-3.2-1B-Instruct";
  if (!have_gpu_model(dir, "canonical_long"))
    GTEST_SKIP() << "no GPU, 1B model, or long fixtures (regenerate: dump_canonical.py "
                    "--prompt-file)";
  run_long_case(dir,
                std::string(ALICE_SOURCE_DIR) +
                    "/models/llama-3.2-1b-instruct.manifest.json",
                fixture("canonical_long", ""));
}

TEST(GpuModel, CanonicalLong8B) {
  const std::string dir = "/opt/models/deepseek-ai/deepseek-r1-distill-llama-8b";
  if (!have_gpu_model(dir, "canonical_long_8b"))
    GTEST_SKIP() << "no GPU, 8B model, or long fixtures (regenerate: dump_canonical.py "
                    "--prompt-file)";
  run_long_case(dir,
                std::string(ALICE_SOURCE_DIR) +
                    "/models/deepseek-r1-distill-llama-8b.manifest.json",
                fixture("canonical_long_8b", ""));
}

// Ad-hoc model validation without recompile (mirrors CanonicalEnv for the
// host path): ALICE_CANONICAL_{MODEL_DIR,MANIFEST,FIXTURES[,CHUNK]}.
TEST(GpuModel, CanonicalEnvTeacherForced) {
  const char* dir = getenv("ALICE_CANONICAL_MODEL_DIR");
  const char* man = getenv("ALICE_CANONICAL_MANIFEST");
  const char* fix = getenv("ALICE_CANONICAL_FIXTURES");
  if (!dir || !man || !fix) GTEST_SKIP() << "ALICE_CANONICAL_* not set";
  alice::EngineConfig cfg;
  cfg.max_seq_len = 8192;
  const char* chunk_env = getenv("ALICE_CANONICAL_CHUNK");
  if (chunk_env) cfg.max_prefill_chunk = uint32_t(atoi(chunk_env));
  auto model = alice::gpu::Model::load(dir, man, cfg);
  run_teacher_forced(*model, fix, cfg.max_prefill_chunk, 3);
}

}  // namespace
