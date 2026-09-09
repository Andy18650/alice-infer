#include "alice/manifest.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "alice/safetensors.hpp"

namespace {

using alice::Manifest;
using alice::safetensors::File;

// --- toy weight source: one attention + one ffn block, tiny dims -----------
// hidden=8, intermediate=16, heads=2, kv_heads=1, head_dim=4, vocab=16
struct ToyTensor {
  std::string name;
  std::vector<uint32_t> shape;
};

std::string toy_file(const std::vector<ToyTensor>& extra = {}) {
  std::vector<ToyTensor> ts = {
      {"model.embed_tokens.weight", {16, 8}},
      {"model.layers.0.self_attn.q_proj.weight", {8, 8}},
      {"model.layers.0.self_attn.k_proj.weight", {4, 8}},
      {"model.layers.0.self_attn.v_proj.weight", {4, 8}},
      {"model.layers.0.self_attn.o_proj.weight", {8, 8}},
      {"model.layers.0.mlp.gate_proj.weight", {16, 8}},
      {"model.layers.0.mlp.up_proj.weight", {16, 8}},
      {"model.layers.0.mlp.down_proj.weight", {8, 16}},
      {"model.layers.0.input_layernorm.weight", {8}},
      {"model.layers.0.post_attention_layernorm.weight", {8}},
      {"model.norm.weight", {8}},
  };
  ts.insert(ts.end(), extra.begin(), extra.end());

  std::string path = std::string(::testing::TempDir()) + "/alice_mani_toy2.safetensors";
  std::string json = "{";
  size_t off = 0;
  for (size_t i = 0; i < ts.size(); ++i) {
    size_t elems = 1;
    for (auto d : ts[i].shape) elems *= d;
    if (i) json += ",";
    json += "\"" + ts[i].name + "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (size_t d = 0; d < ts[i].shape.size(); ++d) {
      if (d) json += ",";
      json += std::to_string(ts[i].shape[d]);
    }
    json += "],\"data_offsets\":[" + std::to_string(off) + "," +
            std::to_string(off + elems * 2) + "]}";
    off += elems * 2;
  }
  json += "}";
  std::ofstream out(path, std::ios::binary);
  uint64_t len = json.size();
  out.write(reinterpret_cast<const char*>(&len), 8);
  out.write(json.data(), json.size());
  std::vector<char> zeros(off, 0);
  out.write(zeros.data(), zeros.size());
  return path;
}

std::string spec(const std::string& name, const std::string& shape) {
  return "{\"name\":\"" + name + "\",\"shape\":" + shape + ",\"dtype\":\"bf16\"}";
}

std::string toy_manifest_json() {
  return std::string(R"({
 "model": "toy",
 "config": {"hidden_size": 8, "intermediate_size": 16, "num_layers": 1,
   "num_heads": 2, "num_kv_heads": 1, "head_dim": 4, "vocab_size": 16,
   "rms_norm_eps": 1e-5, "rope_theta": 500000.0},
 "embedder": {"token_embedding": )") +
         spec("model.embed_tokens.weight", "[16, 8]") + R"(},
 "trunk": {"type": "rmsnorm_add",
   "pre_norms": [)" +
         spec("model.layers.0.input_layernorm.weight", "[8]") + "," +
         spec("model.layers.0.post_attention_layernorm.weight", "[8]") + R"(],
   "final_norm": )" +
         spec("model.norm.weight", "[8]") + R"(},
 "blocks": [
  {"type": "gqa_attention_bf16", "weights": {
    "wq": )" + spec("model.layers.0.self_attn.q_proj.weight", "[8, 8]") + R"(,
    "wk": )" + spec("model.layers.0.self_attn.k_proj.weight", "[4, 8]") + R"(,
    "wv": )" + spec("model.layers.0.self_attn.v_proj.weight", "[4, 8]") + R"(,
    "wo": )" + spec("model.layers.0.self_attn.o_proj.weight", "[8, 8]") + R"(}},
  {"type": "dense_ffn_bf16", "weights": {
    "w_gate": )" + spec("model.layers.0.mlp.gate_proj.weight", "[16, 8]") + R"(,
    "w_up":   )" + spec("model.layers.0.mlp.up_proj.weight", "[16, 8]") + R"(,
    "w_down": )" + spec("model.layers.0.mlp.down_proj.weight", "[8, 16]") + R"(}}],
 "lm_head": {"weight": {"alias_of": "model.embed_tokens.weight",
   "shape": [16, 8], "dtype": "bf16"}}
})";
}

void expect_error(const std::vector<std::string>& errors, const std::string& needle) {
  for (const auto& e : errors)
    if (e.find(needle) != std::string::npos) return;
  FAIL() << "expected error containing '" << needle << "'; got "
         << (errors.empty() ? std::string("no errors") : errors.front());
}

std::string replace_first(std::string s, const std::string& from, const std::string& to) {
  auto pos = s.find(from);
  EXPECT_NE(pos, std::string::npos);
  return s.replace(pos, from.size(), to);
}

TEST(Manifest, ToyContractHolds) {
  auto f = File::open(toy_file());
  auto m = Manifest::parse_string(toy_manifest_json());
  EXPECT_EQ(m.blocks.size(), 2u);
  EXPECT_EQ(m.pre_norms.size(), 2u);
  EXPECT_TRUE(m.lm_head.is_alias());
  auto errors = validate(m, f);
  for (const auto& e : errors) ADD_FAILURE() << "unexpected: " << e;
  EXPECT_TRUE(errors.empty());
}

TEST(Manifest, MissingTensor) {
  auto f = File::open(toy_file());
  auto j = replace_first(toy_manifest_json(), "q_proj.weight", "qq_proj.weight");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "not found");
}

TEST(Manifest, ShapeMismatch) {
  auto f = File::open(toy_file());
  auto j = toy_manifest_json();
  auto pos = j.find("q_proj.weight");
  ASSERT_NE(pos, std::string::npos);
  auto spos = j.find("[8, 8]", pos);
  ASSERT_NE(spos, std::string::npos);
  j.replace(spos, 6, "[7, 8]");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "shape mismatch");
}

TEST(Manifest, DtypeMismatch) {
  auto f = File::open(toy_file());
  auto j = toy_manifest_json();
  auto pos = j.find("q_proj.weight");
  ASSERT_NE(pos, std::string::npos);
  auto dpos = j.find("bf16", pos);
  ASSERT_NE(dpos, std::string::npos);
  j.replace(dpos, 4, "f32");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "dtype mismatch");
}

TEST(Manifest, UnclaimedTensor) {
  auto f = File::open(toy_file({{"model.surprise.weight", {4}}}));
  auto errors = validate(Manifest::parse_string(toy_manifest_json()), f);
  expect_error(errors, "unclaimed");
}

TEST(Manifest, DuplicateClaim) {
  auto f = File::open(toy_file());
  // lm_head directly claims embed_tokens instead of aliasing it
  auto j = replace_first(toy_manifest_json(),
                         "{\"alias_of\": \"model.embed_tokens.weight\",",
                         "{\"name\": \"model.embed_tokens.weight\",");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "claimed twice");
}

TEST(Manifest, BadAliasTarget) {
  auto f = File::open(toy_file());
  auto j = replace_first(toy_manifest_json(),
                         "\"alias_of\": \"model.embed_tokens.weight\"",
                         "\"alias_of\": \"model.nonexistent.weight\"");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "alias target");
}

TEST(Manifest, ConfigInconsistency) {
  auto f = File::open(toy_file());
  // heads 2 -> 4 would make wq expected [16, 8] vs manifest [8, 8]
  auto j = replace_first(toy_manifest_json(), "\"num_heads\": 2", "\"num_heads\": 4");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "from config");
}

TEST(Manifest, PreNormCountMismatch) {
  auto f = File::open(toy_file());
  auto j = toy_manifest_json();
  auto pos = j.find("post_attention_layernorm");
  ASSERT_NE(pos, std::string::npos);
  // delete the second pre_norm entry (find its enclosing {...})
  auto begin = j.rfind('{', pos);
  auto end = j.find('}', pos);
  ASSERT_NE(begin, std::string::npos);
  ASSERT_NE(end, std::string::npos);
  j.erase(begin - 1, end - begin + 2);  // ",{...}"
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "pre_norms");
}

TEST(Manifest, UnknownBlockType) {
  auto f = File::open(toy_file());
  auto j = replace_first(toy_manifest_json(), "gqa_attention_bf16", "mamba_bf16");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "unknown block type");
}

TEST(Manifest, MissingRole) {
  auto f = File::open(toy_file());
  auto j = replace_first(toy_manifest_json(), "\"wo\":", "\"w_oops\":");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "missing required role");
  expect_error(errors, "unknown role");
}

// --- the real bring-up model: the contract must hold against the real file -
TEST(Manifest, RealBringUpModel) {
  const std::string weights = "/opt/models/Llama-3.2-1B-Instruct/model.safetensors";
  const std::string manifest =
      std::string(ALICE_SOURCE_DIR) + "/models/llama-3.2-1b-instruct.manifest.json";
  if (!std::filesystem::exists(weights) || !std::filesystem::exists(manifest))
    GTEST_SKIP() << "bring-up model or manifest not present";

  auto f = File::open(weights);
  EXPECT_EQ(f.tensors().size(), 146u);
  auto m = Manifest::parse(manifest);
  EXPECT_EQ(m.blocks.size(), 32u);
  EXPECT_EQ(m.pre_norms.size(), 32u);

  auto errors = validate(m, f);
  for (const auto& e : errors) ADD_FAILURE() << e;
  EXPECT_TRUE(errors.empty());

  // load-time read smoke: the final norm is 4 KiB of BF16
  const auto& norm = f.at("model.norm.weight");
  std::vector<std::byte> buf(norm.nbytes);
  EXPECT_NO_THROW(f.read(norm, buf.data()));
}

}  // namespace
