#include "alice/manifest.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "alice/safetensors.hpp"

namespace {

using alice::Manifest;
using alice::safetensors::Files;

// --- toy weight source: one attention + one ffn block, tiny dims -----------
// hidden=8, intermediate=16, heads=2, kv_heads=1, head_dim=4, vocab=16
struct ToyTensor {
  std::string name;
  std::vector<uint32_t> shape;
  std::string dtype = "BF16";  // "BF16" | "F32" (audit dtype test)
};

std::vector<ToyTensor> toy_tensors() {
  return {
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
}

std::string write_shard(const std::string& tag, const std::vector<ToyTensor>& ts) {
  std::string path =
      std::string(::testing::TempDir()) + "/alice_mani_" + tag + ".safetensors";
  std::string json = "{";
  size_t off = 0;
  for (size_t i = 0; i < ts.size(); ++i) {
    size_t elems = 1;
    for (auto d : ts[i].shape) elems *= d;
    const size_t esz = ts[i].dtype == "BF16" ? 2 : 4;
    if (i) json += ",";
    json += "\"" + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype + "\",\"shape\":[";
    for (size_t d = 0; d < ts[i].shape.size(); ++d) {
      if (d) json += ",";
      json += std::to_string(ts[i].shape[d]);
    }
    json += "],\"data_offsets\":[" + std::to_string(off) + "," +
            std::to_string(off + elems * esz) + "]}";
    off += elems * esz;
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

std::string toy_file(const std::vector<ToyTensor>& extra = {}) {
  auto ts = toy_tensors();
  ts.insert(ts.end(), extra.begin(), extra.end());
  return write_shard("toy2", ts);
}

std::string spec(const std::string& name, const std::string& shape) {
  return "{\"name\":\"" + name + "\",\"shape\":" + shape + "}";
}

std::string toy_manifest_json() {
  return std::string(R"({
 "model": "toy",
 "config": {"hidden_size": 8, "intermediate_size": 16,
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
 "lm_head": {"weight": {"name": "model.embed_tokens.weight",
   "shape": [16, 8]}}
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
  auto f = Files::open({toy_file()});
  auto m = Manifest::parse_string(toy_manifest_json());
  EXPECT_EQ(m.blocks.size(), 2u);
  EXPECT_EQ(m.pre_norms.size(), 2u);
  // toy lm_head is tied (names the embedder's tensor — the one legal shared
  // claim; the alias mechanism is gone, manifest diet 2026-09)
  EXPECT_EQ(m.lm_head.name, m.token_embedding.name);
  auto errors = validate(m, f);
  for (const auto& e : errors) ADD_FAILURE() << "unexpected: " << e;
  EXPECT_TRUE(errors.empty());
}

TEST(Manifest, MultiShardContractHolds) {
  // Split the toy tensors across two shards; the audit is global.
  auto ts = toy_tensors();
  auto mid = ts.begin() + 5;
  auto a = write_shard("shard_a", {ts.begin(), mid});
  auto b = write_shard("shard_b", {mid, ts.end()});
  auto f = Files::open({a, b});
  EXPECT_EQ(f.num_shards(), 2u);
  auto errors = validate(Manifest::parse_string(toy_manifest_json()), f);
  for (const auto& e : errors) ADD_FAILURE() << "unexpected: " << e;
  EXPECT_TRUE(errors.empty());
}

TEST(Manifest, MultiShardUnclaimedInSecondShard) {
  auto ts = toy_tensors();
  auto mid = ts.begin() + 5;
  auto a = write_shard("shard_a", {ts.begin(), mid});
  std::vector<ToyTensor> rest(mid, ts.end());
  rest.push_back({"model.surprise.weight", {4}});
  auto b = write_shard("shard_b", rest);
  auto f = Files::open({a, b});
  auto errors = validate(Manifest::parse_string(toy_manifest_json()), f);
  expect_error(errors, "unclaimed");
}

TEST(Manifest, MissingTensor) {
  auto f = Files::open({toy_file()});
  auto j = replace_first(toy_manifest_json(), "q_proj.weight", "qq_proj.weight");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "not found");
}

TEST(Manifest, ShapeMismatch) {
  auto f = Files::open({toy_file()});
  auto j = toy_manifest_json();
  auto pos = j.find("q_proj.weight");
  ASSERT_NE(pos, std::string::npos);
  auto spos = j.find("[8, 8]", pos);
  ASSERT_NE(spos, std::string::npos);
  j.replace(spos, 6, "[7, 8]");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "shape mismatch");
}

// Dtype is not declared per weight anymore (derived: all current block types
// are bf16); the audit still checks the FILE side, so the mismatch case is an
// F32 tensor in the weight source.
TEST(Manifest, DtypeMismatch) {
  auto ts = toy_tensors();
  for (auto& t : ts)
    if (t.name.find("q_proj") != std::string::npos) t.dtype = "F32";
  auto f = Files::open({write_shard("toy_f32", ts)});
  auto errors = validate(Manifest::parse_string(toy_manifest_json()), f);
  expect_error(errors, "dtype mismatch");
  expect_error(errors, "expected bf16");
}

TEST(Manifest, UnclaimedTensor) {
  auto f = Files::open({toy_file({{"model.surprise.weight", {4}}})});
  auto errors = validate(Manifest::parse_string(toy_manifest_json()), f);
  expect_error(errors, "unclaimed");
}

TEST(Manifest, DuplicateClaim) {
  auto f = Files::open({toy_file()});
  // two block roles claiming one tensor: wv points at q_proj's tensor
  // (fires the duplicate-claim error; the shape mismatch also fires)
  auto j = replace_first(
      toy_manifest_json(),
      "{\"name\":\"model.layers.0.self_attn.v_proj.weight\",\"shape\":[4, 8]}",
      "{\"name\":\"model.layers.0.self_attn.q_proj.weight\",\"shape\":[4, 8]}");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "claimed twice");
}

TEST(Manifest, ConfigInconsistency) {
  auto f = Files::open({toy_file()});
  // heads 2 -> 4 would make wq expected [16, 8] vs manifest [8, 8]
  auto j = replace_first(toy_manifest_json(), "\"num_heads\": 2", "\"num_heads\": 4");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "from config");
}

TEST(Manifest, PreNormCountMismatch) {
  auto f = Files::open({toy_file()});
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
  auto f = Files::open({toy_file()});
  auto j = replace_first(toy_manifest_json(), "gqa_attention_bf16", "mamba_bf16");
  auto errors = validate(Manifest::parse_string(j), f);
  expect_error(errors, "unknown block type");
}

TEST(Manifest, MissingRole) {
  auto f = Files::open({toy_file()});
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

  auto f = Files::open({weights});
  EXPECT_EQ(f.shards().front().tensors().size(), 146u);
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

// --- HF model directory: config.json synthesis (EXPERIMENTAL path) --------

// Writes an HF-style config.json under a fresh temp dir; returns the dir.
std::filesystem::path write_hf_config(const std::string& json_text) {
  static int n = 0;
  const auto dir = std::filesystem::temp_directory_path() /
      ("alice_hf_cfg_" + std::to_string(::getpid()) + "_" + std::to_string(n++));
  std::filesystem::create_directories(dir);
  { std::ofstream(dir / "config.json") << json_text; }
  return dir;
}

TEST(Manifest, DirectorySynthesizesLlamaManifest) {
  // Minimal LlamaForCausalLM config: 1 layer, tied embeddings, llama3 rope.
  const auto dir = write_hf_config(R"({
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 100,
    "rms_norm_eps": 1e-05, "rope_theta": 500000.0,
    "tie_word_embeddings": true,
    "rope_scaling": {"rope_type": "llama3", "factor": 32.0,
      "low_freq_factor": 1.0, "high_freq_factor": 4.0,
      "original_max_position_embeddings": 8192}
  })");
  auto m = Manifest::parse(dir);
  EXPECT_EQ(m.model_name, dir.filename().string());
  EXPECT_EQ(m.config.hidden_size, 64u);
  EXPECT_EQ(m.config.head_dim, 16u);            // defaulted: d / nh
  EXPECT_EQ(m.config.num_kv_heads, 2u);
  EXPECT_TRUE(m.config.has_rope_scaling);       // accepted "rope_type" spelling
  EXPECT_EQ(m.config.rope_scaling.factor, 32.0f);
  EXPECT_EQ(m.blocks.size(), 2u);               // attn + ffn per layer
  EXPECT_EQ(m.pre_norms.size(), 2u);
  EXPECT_EQ(m.blocks[0].type, "gqa_attention_bf16");
  EXPECT_EQ(m.blocks[0].weights.at("wq").rows, 64u);      // nh * hd
  EXPECT_EQ(m.blocks[0].weights.at("wk").rows, 32u);      // nkv * hd
  EXPECT_EQ(m.blocks[1].type, "dense_ffn_bf16");
  EXPECT_EQ(m.blocks[1].weights.at("w_down").cols, 128u); // intermediate
  // tied: the lm_head role names the embedder's tensor
  EXPECT_EQ(m.lm_head.name, m.token_embedding.name);
  EXPECT_EQ(m.lm_head.rows, 100u);
}

TEST(Manifest, DirectoryConfigInsufficientReportsAllMissing) {
  const auto dir = write_hf_config(
      R"({"architectures": ["LlamaForCausalLM"], "hidden_size": 64})");
  try {
    Manifest::parse(dir);
    FAIL() << "parse should have thrown";
  } catch (const std::runtime_error& e) {
    // every missing field named at once, not first-miss aborts
    EXPECT_NE(std::string(e.what()).find("intermediate_size"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("num_key_value_heads"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("rope_theta"), std::string::npos);
  }
}

TEST(Manifest, DirectoryRejectsNonLlamaArchitecture) {
  const auto dir = write_hf_config(
      R"({"architectures": ["Qwen2ForCausalLM"], "hidden_size": 64})");
  try {
    Manifest::parse(dir);
    FAIL() << "parse should have thrown";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("Qwen2ForCausalLM"), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("only LlamaForCausalLM"), std::string::npos);
  }
}

TEST(Manifest, DirectoryMatchesAuthoredManifest) {
  // The synthesis must be the generator's document: parsing the real 1B
  // model directory must equal parsing the committed gen_manifest.py output.
  const std::string weights =
      "/opt/models/Llama-3.2-1B-Instruct/model.safetensors";
  const std::string manifest =
      std::string(ALICE_SOURCE_DIR) +
      "/models/llama-3.2-1b-instruct.manifest.json";
  const std::string dir = "/opt/models/Llama-3.2-1B-Instruct";
  if (!std::filesystem::exists(weights) || !std::filesystem::exists(manifest))
    GTEST_SKIP() << "bring-up model or manifest not present";

  auto authored = Manifest::parse(manifest);
  auto synth = Manifest::parse(dir);
  EXPECT_EQ(synth.model_name, authored.model_name);
  EXPECT_EQ(synth.config.vocab_size, authored.config.vocab_size);
  EXPECT_EQ(synth.config.head_dim, authored.config.head_dim);
  EXPECT_EQ(synth.config.rms_norm_eps, authored.config.rms_norm_eps);
  EXPECT_EQ(synth.config.rope_theta, authored.config.rope_theta);
  EXPECT_EQ(synth.config.has_rope_scaling, authored.config.has_rope_scaling);
  EXPECT_EQ(synth.blocks.size(), authored.blocks.size());
  EXPECT_EQ(synth.pre_norms.size(), authored.pre_norms.size());
  EXPECT_EQ(synth.lm_head.name, authored.lm_head.name);  // both tied
  for (size_t i = 0; i < synth.blocks.size(); ++i) {
    EXPECT_EQ(synth.blocks[i].type, authored.blocks[i].type);
    for (const auto& [role, w] : authored.blocks[i].weights) {
      EXPECT_EQ(synth.blocks[i].weights.at(role).name, w.name);
      EXPECT_EQ(synth.blocks[i].weights.at(role).rows, w.rows);
      EXPECT_EQ(synth.blocks[i].weights.at(role).cols, w.cols);
    }
  }

  // and the synthesized contract holds against the real weights
  auto f = Files::open({weights});
  auto errors = validate(synth, f);
  for (const auto& e : errors) ADD_FAILURE() << e;
}

}  // namespace
