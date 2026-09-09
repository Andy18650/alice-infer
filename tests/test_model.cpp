// End-to-end smoke: the real bring-up model loads via the manifest contract
// and generates deterministically. Gated on the model being present.

#include <filesystem>
#include <vector>

#include <gtest/gtest.h>

#include "alice/model.hpp"

namespace {

TEST(Model, RealBringUpSmoke) {
  const std::string dir = "/opt/models/Llama-3.2-1B-Instruct";
  const std::string manifest =
      std::string(ALICE_SOURCE_DIR) + "/models/llama-3.2-1b-instruct.manifest.json";
  if (!std::filesystem::exists(dir + "/model.safetensors"))
    GTEST_SKIP() << "bring-up model not present";

  alice::EngineConfig cfg;
  cfg.max_seq_len = 64;
  cfg.max_prefill_chunk = 8;
  cfg.sampling.temperature = 0.0f;  // greedy: determinism
  auto model = alice::ModelLoader::load(dir, manifest, cfg);
  EXPECT_EQ(model->vocab_size(), 128256u);

  constexpr int32_t kBos = 128000;
  auto run = [&]() {
    model->reset();
    std::vector<int32_t> out;
    std::vector<int32_t> prompt = {kBos};
    int32_t t = model->forward(prompt);
    out.push_back(t);
    for (int i = 0; i < 2; ++i) {
      std::vector<int32_t> one = {t};
      t = model->forward(one);
      out.push_back(t);
    }
    return out;
  };

  const auto r1 = run();
  const auto r2 = run();
  EXPECT_EQ(r1, r2) << "greedy generation not deterministic across reset";
  for (int32_t t : r1) {
    EXPECT_GE(t, 0);
    EXPECT_LT(uint32_t(t), model->vocab_size());
  }
  for (size_t i = 0; i < r1.size(); ++i)
    printf("  generated[%zu] = %d\n", i, r1[i]);  // visible with --gtest_output

  std::vector<int32_t> too_long(cfg.max_prefill_chunk + 1, 1);
  EXPECT_THROW(model->forward(too_long), std::runtime_error);
}

}  // namespace
