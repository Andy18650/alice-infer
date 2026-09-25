#include "alice/manifest.hpp"

#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace alice {

namespace {

using nlohmann::json;

// ===========================================================================
// HF config.json synthesis (EXPERIMENTAL, pending design review)
// ===========================================================================
// Mirrors tools/gen_manifest.py field for field, so an engine-synthesized
// manifest and a generator-authored one are the same document. Known
// divergence from the runtime-trust design decision: HF config.json was
// deliberately NOT a runtime input — no schema stability, the wrong
// vocabulary (no role mapping or block composition), and a second input
// channel would break the audit's independence. This path is here to be
// judged as code, not to quietly amend the design.

WeightSpec hf_spec(std::string name, uint32_t rows, uint32_t cols) {
  return WeightSpec{std::move(name), rows, cols};
}

Manifest from_hf_llama_config(const json& c, const std::filesystem::path& dir) {
  // Insufficiency report: everything the Llama format needs, all at once
  // (the audit reports all problems in one pass; this matches it).
  static const char* kRequired[] = {
      "hidden_size", "intermediate_size", "num_hidden_layers",
      "num_attention_heads", "num_key_value_heads", "vocab_size",
      "rms_norm_eps", "rope_theta"};
  std::string missing;
  for (const char* k : kRequired)
    if (!c.contains(k))
      missing += std::string(missing.empty() ? "" : ", ") + k;
  if (!missing.empty())
    throw std::runtime_error(
        "manifest: config.json has insufficient information for "
        "LlamaForCausalLM, missing: " + missing);

  const uint32_t d = c.at("hidden_size").get<uint32_t>();
  const uint32_t inter = c.at("intermediate_size").get<uint32_t>();
  const uint32_t L = c.at("num_hidden_layers").get<uint32_t>();
  const uint32_t nh = c.at("num_attention_heads").get<uint32_t>();
  const uint32_t nkv = c.at("num_key_value_heads").get<uint32_t>();
  const uint32_t hd =
      c.contains("head_dim") ? c.at("head_dim").get<uint32_t>() : d / nh;

  Manifest m;
  m.model_name = c.value("_name_or_path", std::string{});
  if (m.model_name.empty()) {
    m.model_name = dir.filename().string();
  } else {
    // basename (gen_manifest.py: rstrip('/').split('/')[-1])
    while (m.model_name.size() > 1 && m.model_name.back() == '/')
      m.model_name.pop_back();
    m.model_name = m.model_name.substr(m.model_name.find_last_of('/') + 1);
  }

  ModelConfig& mc = m.config;
  mc.hidden_size = d;
  mc.intermediate_size = inter;
  mc.num_heads = nh;
  mc.num_kv_heads = nkv;
  mc.head_dim = hd;
  mc.vocab_size = c.at("vocab_size").get<uint32_t>();
  mc.rms_norm_eps = c.at("rms_norm_eps").get<float>();
  mc.rope_theta = c.at("rope_theta").get<float>();
  if (c.contains("rope_scaling")) {
    const auto& rs = c.at("rope_scaling");
    // HF renamed "type" to "rope_type" in the llama3 era; accept either.
    const std::string type = rs.value("rope_type", rs.value("type", ""));
    if (type != "llama3")
      throw std::runtime_error(
          "manifest: config.json: unsupported rope_scaling type '" + type +
          "' (only llama3)");
    mc.has_rope_scaling = true;
    mc.rope_scaling.factor = rs.at("factor").get<float>();
    mc.rope_scaling.low_freq_factor = rs.at("low_freq_factor").get<float>();
    mc.rope_scaling.high_freq_factor = rs.at("high_freq_factor").get<float>();
    mc.rope_scaling.original_max_position_embeddings =
        rs.at("original_max_position_embeddings").get<uint32_t>();
  }
  mc.bos_token_id = c.value("bos_token_id", 128000u);
  // HF writes eos as a scalar or a list; the engine wants the list.
  const json eos = c.contains("eos_token_id") ? c.at("eos_token_id")
                                              : json(128001);
  if (eos.is_array())
    for (const auto& e : eos) mc.eos_token_ids.push_back(e.get<uint32_t>());
  else
    mc.eos_token_ids.push_back(eos.get<uint32_t>());

  m.token_embedding = hf_spec("model.embed_tokens.weight", mc.vocab_size, d);

  m.trunk_type = "rmsnorm_add";
  for (uint32_t i = 0; i < L; ++i) {
    const std::string p = "model.layers." + std::to_string(i);
    m.pre_norms.push_back(hf_spec(p + ".input_layernorm.weight", 1, d));
    m.pre_norms.push_back(
        hf_spec(p + ".post_attention_layernorm.weight", 1, d));

    BlockSpec attn;
    attn.type = "gqa_attention_bf16";
    attn.weights["wq"] = hf_spec(p + ".self_attn.q_proj.weight", nh * hd, d);
    attn.weights["wk"] = hf_spec(p + ".self_attn.k_proj.weight", nkv * hd, d);
    attn.weights["wv"] = hf_spec(p + ".self_attn.v_proj.weight", nkv * hd, d);
    attn.weights["wo"] = hf_spec(p + ".self_attn.o_proj.weight", d, nh * hd);
    m.blocks.push_back(std::move(attn));

    BlockSpec ffn;
    ffn.type = "dense_ffn_bf16";
    ffn.weights["w_gate"] = hf_spec(p + ".mlp.gate_proj.weight", inter, d);
    ffn.weights["w_up"] = hf_spec(p + ".mlp.up_proj.weight", inter, d);
    ffn.weights["w_down"] = hf_spec(p + ".mlp.down_proj.weight", d, inter);
    m.blocks.push_back(std::move(ffn));
  }
  m.final_norm = hf_spec("model.norm.weight", 1, d);

  // Tied embeddings: name the EMBEDDER's tensor in the lm_head role — the one
  // legal shared claim (validate() exempts that pair). Same rule as
  // gen_manifest.py.
  m.lm_head = c.value("tie_word_embeddings", false)
                  ? hf_spec("model.embed_tokens.weight", mc.vocab_size, d)
                  : hf_spec("lm_head.weight", mc.vocab_size, d);
  return m;
}

// ===========================================================================
// Manifest JSON parsing (the engine's own vocabulary)
// ===========================================================================

WeightSpec parse_spec(const json& j, const std::string& ctx) {
  WeightSpec w;
  w.name = j.at("name").get<std::string>();
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
  return w;
}

Manifest from_json(const json& j) {
  Manifest m;
  m.model_name = j.value("model", std::string{});

  const auto& c = j.at("config");
  auto& mc = m.config;
  mc.hidden_size = c.at("hidden_size").get<uint32_t>();
  mc.intermediate_size = c.at("intermediate_size").get<uint32_t>();
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

Manifest Manifest::parse(const std::filesystem::path& path) {
  // Directory => HF model directory: synthesize the manifest from config.json
  // (EXPERIMENTAL path — see from_hf_llama_config above).
  std::error_code ec;
  if (std::filesystem::is_directory(path, ec)) {
    const std::filesystem::path cfg_path = path / "config.json";
    std::ifstream in(cfg_path);
    if (!in)
      throw std::runtime_error("manifest: no config.json under " +
                               path.string());
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    json c;
    try {
      c = json::parse(text);
    } catch (const json::exception& e) {
      throw std::runtime_error(std::string("manifest: config.json: ") +
                               e.what());
    }
    if (!c.contains("architectures") || !c.at("architectures").is_array())
      throw std::runtime_error(
          "manifest: config.json has no architectures list");
    std::string found;
    bool llama = false;
    for (const auto& a : c.at("architectures")) {
      const std::string arch = a.get<std::string>();
      found += (found.empty() ? "" : ", ") + arch;
      if (arch == "LlamaForCausalLM") llama = true;
    }
    if (!llama)
      throw std::runtime_error(
          "manifest: config.json architecture(s) [" + found +
          "] cannot be synthesized from a model directory "
          "(only LlamaForCausalLM)");
    return from_hf_llama_config(c, path);
  }
  if (ec)
    throw std::runtime_error("manifest: cannot open " + path.string() +
                             ": " + ec.message());

  std::ifstream in(path);
  if (!in) throw std::runtime_error("manifest: cannot open " + path.string());
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
    if (w.name.empty()) {
      errors.push_back(what + ": empty tensor name");
      return;
    }
    // Tied embeddings: the lm_head may claim the SAME tensor as the
    // embedder — the one legal shared claim (manifest diet 2026-09: this
    // rule replaces the former alias_of mechanism). Anything else claiming
    // a tensor twice is an error; a mistied lm_head still fails the shape
    // check below (only a [vocab, d] tensor passes).
    const bool tied_embedding =
        what == "lm_head.weight" && w.name == m.token_embedding.name;
    if (!tied_embedding && !direct_claims.insert(w.name).second)
      errors.push_back(what + ": tensor '" + w.name + "' claimed twice");
    all_claims.insert(w.name);
    if (!f.contains(w.name)) {
      errors.push_back(what + ": tensor '" + w.name + "' not found in weight source");
      return;
    }
    const auto& t = f.at(w.name);
    if (t.rows != w.rows || t.cols != w.cols)
      errors.push_back(what + ": shape mismatch on '" + w.name + "': manifest " +
                       dims(w.rows, w.cols) + " vs file " + dims(t.rows, t.cols));
    // Expected dtype is DERIVED, not declared (2026-09 manifest diet): every
    // weightful component today is bf16 (the block types say so), and the
    // load boundary enforces it via File::check_as<T>. The audit keeps the
    // file-side check so a wrong-dtype checkpoint fails all-errors-at-once
    // instead of on the first stage(). When a non-bf16 block type lands,
    // this becomes a per-type lookup.
    if (t.dtype != DType::BF16)
      errors.push_back(what + ": dtype mismatch on '" + w.name +
                       "': expected bf16 (all current block types are bf16)");
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
