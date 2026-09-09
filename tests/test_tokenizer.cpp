// Tokenizer battery: our BPE must match HF `tokenizers` exactly on the
// committed battery fixture (tests/canonical/tokenizer_battery.json).

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "../src/frontend/tokenizer.hpp"

namespace {

using alice::frontend::Tokenizer;

TEST(Tokenizer, BatteryMatchesHF) {
  const std::string tok_path = "/opt/models/Llama-3.2-1B-Instruct/tokenizer.json";
  const std::string battery_path =
      std::string(ALICE_SOURCE_DIR) + "/tests/canonical/tokenizer_battery.json";
  if (!std::filesystem::exists(tok_path) || !std::filesystem::exists(battery_path))
    GTEST_SKIP() << "tokenizer or battery fixture not present";

  Tokenizer tok(tok_path);
  EXPECT_EQ(tok.vocab_size(), 128000u);
  EXPECT_EQ(tok.decode_token(128000), "<|begin_of_text|>");

  std::ifstream in(battery_path);
  const auto battery = nlohmann::json::parse(std::string(
      (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));

  size_t case_i = 0;
  for (const auto& c : battery) {
    const std::string text = c.at("text").get<std::string>();
    const std::vector<int32_t> want = c.at("ids").get<std::vector<int32_t>>();
    const std::vector<int32_t> got = tok.encode(text);
    EXPECT_EQ(got, want) << "case " << case_i << ": encode mismatch for " << std::quoted(text);
    if (got == want) {
      const std::string roundtrip = tok.decode(got);
      EXPECT_EQ(roundtrip, text) << "case " << case_i << ": decode roundtrip mismatch";
    }
    ++case_i;
  }
}

}  // namespace
