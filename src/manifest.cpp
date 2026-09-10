#include "alice/manifest.hpp"

#include <fstream>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace alice {

namespace {

using nlohmann::json;

DType parse_dtype(const std::string& s, const std::string& ctx) {
  if (s == "bf16") return DType::BF16;
  if (s == "f32") return DType::F32;
  throw std::runtime_error("manifest: " + ctx + ": unknown dtype '" + s +
                           "' (expected bf16|f32)");
}

WeightSpec parse_spec(const json& j, const std::string& ctx) {
  WeightSpec w;
  if (j.contains("alias_of")) {
    w.alias_of = j.at("alias_of").get<std::string>();
  } else {
    w.name = j.at("name").get<std::string>();
  }
  const auto& shape = j.at("shape");
  if (!shape.is_array() || shape.size() < 1 || shape.size() > 2)
    throw std::runtime_error("manifest: " + ctx + ": shape must have rank 1-2");
  if (shape.size() == 1) {
    w.rows = 1;
    w.cols = shape[0].get<uint32_t>();
  } else {
    w.rows = shape[0].get<uint32_t>();
    w.cols = shape[1].get<uint32_t>();
  }
  w.dtype = parse_dtype(j.at("dtype").get<std::string>(), ctx);
  return w;
}

Manifest from_json(const json& j) {
  Manifest m;
  m.model_name = j.value("model", std::string{});

  const auto& c = j.at("config");
  auto& mc = m.config;
  mc.hidden_size = c.at("hidden_size").get<uint32_t>();
  mc.intermediate_size = c.at("intermediate_size").get<uint32_t>();
  mc.num_layers = c.at("num_layers").get<uint32_t>();
  mc.num_heads = c.at("num_heads").get<uint32_t>();
  mc.num_kv_heads = c.at("num_kv_heads").get<uint32_t>();
  mc.head_dim = c.at("head_dim").get<uint32_t>();
  mc.vocab_size = c.at("vocab_size").get<uint32_t>();
  mc.rms_norm_eps = c.at("rms_norm_eps").get<float>();
  mc.rope_theta = c.at("rope_theta").get<float>();
  if (c.contains("rope_scaling")) {
    const auto& rs = c.at("rope_scaling");
    if (rs.at("type").get<std::string>() != "llama3")
      throw std::runtime_error("manifest: unsupported rope_scaling type '" +
                               rs.at("type").get<std::string>() + "'");
    mc.has_rope_scaling = true;
    mc.rope_scaling.factor = rs.at("factor").get<float>();
    mc.rope_scaling.low_freq_factor = rs.at("low_freq_factor").get<float>();
    mc.rope_scaling.high_freq_factor = rs.at("high_freq_factor").get<float>();
    mc.rope_scaling.original_max_position_embeddings =
        rs.at("original_max_position_embeddings").get<uint32_t>();
  }
  if (c.contains("bos_token_id")) mc.bos_token_id = c.at("bos_token_id").get<uint32_t>();
  if (c.contains("eos_token_ids"))
    mc.eos_token_ids = c.at("eos_token_ids").get<std::vector<uint32_t>>();

  m.token_embedding = parse_spec(j.at("embedder").at("token_embedding"), "embedder");

  const auto& trunk = j.at("trunk");
  m.trunk_type = trunk.at("type").get<std::string>();
  if (m.trunk_type != "rmsnorm_add")
    throw std::runtime_error("manifest: unknown trunk type '" + m.trunk_type + "'");
  size_t i = 0;
  for (const auto& n : trunk.at("pre_norms"))
    m.pre_norms.push_back(parse_spec(n, "trunk.pre_norms[" + std::to_string(i++) + "]"));
  m.final_norm = parse_spec(trunk.at("final_norm"), "trunk.final_norm");

  i = 0;
  for (const auto& b : j.at("blocks")) {
    BlockSpec bs;
    bs.type = b.at("type").get<std::string>();
    for (auto& [role, spec] : b.at("weights").items())
      bs.weights[role] =
          parse_spec(spec, "blocks[" + std::to_string(i) + "]." + role);
    m.blocks.push_back(std::move(bs));
    ++i;
  }

  m.lm_head = parse_spec(j.at("lm_head").at("weight"), "lm_head");
  return m;
}

std::string dims(uint32_t r, uint32_t c) {
  return "[" + std::to_string(r) + ", " + std::to_string(c) + "]";
}

}  // namespace

Manifest Manifest::parse(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("manifest: cannot open " + path);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return parse_string(text);
}

Manifest Manifest::parse_string(const std::string& json_text) {
  try {
    return from_json(json::parse(json_text));
  } catch (const json::exception& e) {
    throw std::runtime_error(std::string("manifest: ") + e.what());
  }
}

std::vector<std::string> validate(const Manifest& m, const safetensors::Files& f) {
  std::vector<std::string> errors;
  std::set<std::string> direct_claims;
  std::set<std::string> all_claims;
  const ModelConfig& c = m.config;

  auto check = [&](const WeightSpec& w, const std::string& what) {
    const std::string& resolved = w.is_alias() ? w.alias_of : w.name;
    if (resolved.empty()) {
      errors.push_back(what + ": empty tensor name");
      return;
    }
    if (!w.is_alias() && !direct_claims.insert(resolved).second)
      errors.push_back(what + ": tensor '" + resolved + "' claimed twice");
    all_claims.insert(resolved);
    if (!f.contains(resolved)) {
      errors.push_back(what + ": tensor '" + resolved + "'" +
                       (w.is_alias() ? " (alias target)" : "") + " not found in weight source");
      return;
    }
    const auto& t = f.at(resolved);
    if (t.rows != w.rows || t.cols != w.cols)
      errors.push_back(what + ": shape mismatch on '" + resolved + "': manifest " +
                       dims(w.rows, w.cols) + " vs file " + dims(t.rows, t.cols));
    if (t.dtype != w.dtype)
      errors.push_back(what + ": dtype mismatch on '" + resolved + "'");
  };

  auto expect = [&](const WeightSpec& w, const std::string& what, uint32_t r, uint32_t cl) {
    if (w.rows != r || w.cols != cl)
      errors.push_back(what + ": expected " + dims(r, cl) + " from config, manifest says " +
                       dims(w.rows, w.cols));
  };

  // --- structural checks against config ---
  check(m.token_embedding, "embedder.token_embedding");
  expect(m.token_embedding, "embedder.token_embedding", c.vocab_size, c.hidden_size);
  check(m.lm_head, "lm_head.weight");
  expect(m.lm_head, "lm_head.weight", c.vocab_size, c.hidden_size);
  check(m.final_norm, "trunk.final_norm");
  expect(m.final_norm, "trunk.final_norm", 1, c.hidden_size);

  if (m.pre_norms.size() != m.blocks.size())
    errors.push_back("trunk: pre_norms has " + std::to_string(m.pre_norms.size()) +
                     " entries but there are " + std::to_string(m.blocks.size()) + " blocks");
  for (size_t i = 0; i < m.pre_norms.size(); ++i) {
    check(m.pre_norms[i], "trunk.pre_norms[" + std::to_string(i) + "]");
    expect(m.pre_norms[i], "trunk.pre_norms[" + std::to_string(i) + "]", 1, c.hidden_size);
  }

  const uint32_t q_rows = c.num_heads * c.head_dim;
  const uint32_t kv_rows = c.num_kv_heads * c.head_dim;
  for (size_t i = 0; i < m.blocks.size(); ++i) {
    const BlockSpec& b = m.blocks[i];
    const std::string ctx = "blocks[" + std::to_string(i) + "]";

    std::map<std::string, std::pair<uint32_t, uint32_t>> want;
    if (b.type == "gqa_attention_bf16") {
      want = {{"wq", {q_rows, c.hidden_size}},
              {"wk", {kv_rows, c.hidden_size}},
              {"wv", {kv_rows, c.hidden_size}},
              {"wo", {c.hidden_size, q_rows}}};
    } else if (b.type == "dense_ffn_bf16") {
      want = {{"w_gate", {c.intermediate_size, c.hidden_size}},
              {"w_up", {c.intermediate_size, c.hidden_size}},
              {"w_down", {c.hidden_size, c.intermediate_size}}};
    } else {
      errors.push_back(ctx + ": unknown block type '" + b.type + "'");
      continue;
    }
    for (const auto& [role, rc] : want) {
      auto it = b.weights.find(role);
      if (it == b.weights.end()) {
        errors.push_back(ctx + ": missing required role '" + role + "' for type " + b.type);
        continue;
      }
      check(it->second, ctx + "." + role);
      expect(it->second, ctx + "." + role, rc.first, rc.second);
    }
    for (const auto& [role, w] : b.weights)
      if (!want.count(role)) errors.push_back(ctx + ": unknown role '" + role + "'");
  }

  // --- unclaimed tensors in the weight source (global across shards) ---
  for (const auto& shard : f.shards())
    for (const auto& t : shard.tensors())
      if (!all_claims.count(t.name))
        errors.push_back("weight source: tensor '" + t.name + "' is unclaimed by the manifest");

  return errors;
}

}  // namespace alice
