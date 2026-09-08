#include "alice/manifest.hpp"

#include "dtype_string.hpp"

#include <nlohmann/json.hpp>

namespace alice {

namespace {

using nlohmann::json;

[[noreturn]] void fail(const std::string& msg) { throw ManifestError(msg); }

Params parse_params(const json& j, const std::string& path) {
  Params p;
  if (!j.is_object()) fail(path + ": params must be an object");
  for (const auto& [k, v] : j.items()) {
    if (v.is_number_integer() || v.is_number_unsigned()) p[k] = v.get<int64_t>();
    else if (v.is_number_float()) p[k] = v.get<double>();
    else if (v.is_boolean()) p[k] = v.get<bool>();
    else if (v.is_string()) p[k] = v.get<std::string>();
    else fail(path + "." + k + ": nested/array params are not supported yet");
  }
  return p;
}

WeightRef parse_weight(const json& j, const std::string& path) {
  if (!j.is_object()) fail(path + ": weight must be an object");
  const bool has_name = j.contains("name");
  const bool has_alias = j.contains("alias_of");
  if (has_name == has_alias)
    fail(path + ": exactly one of 'name' or 'alias_of' is required");

  WeightRef r;
  if (has_alias) {
    r.alias_of = j.at("alias_of").get<std::string>();
    return r;
  }

  r.name = j.at("name").get<std::string>();

  const auto dit = j.find("dtype");
  if (dit == j.end() || !dit->is_string() ||
      !detail::dtype_from_string(dit->get<std::string>(), r.dtype))
    fail(path + ": missing or unsupported 'dtype' (supported: BF16, F32)");

  const auto sit = j.find("shape");
  if (sit == j.end() || !sit->is_array() || sit->empty() || sit->size() > 2)
    fail(path + ": 'shape' must be an array of rank 1 or 2");
  const uint64_t d0 = sit->at(0).get<uint64_t>();
  const uint64_t d1 = sit->size() == 2 ? sit->at(1).get<uint64_t>() : 0;
  if (d0 > UINT32_MAX || d1 > UINT32_MAX) fail(path + ": dimension too large");
  if (sit->size() == 1) { r.rows = 1; r.cols = uint32_t(d0); }
  else                  { r.rows = uint32_t(d0); r.cols = uint32_t(d1); }
  return r;
}

ComponentSpec parse_component(const json& j, const std::string& path) {
  if (!j.is_object()) fail(path + ": component must be an object");
  ComponentSpec c;
  const auto tit = j.find("type");
  if (tit == j.end() || !tit->is_string()) fail(path + ": missing 'type'");
  c.type = tit->get<std::string>();
  if (const auto wit = j.find("weights"); wit != j.end()) {
    if (!wit->is_object()) fail(path + ": 'weights' must be an object");
    for (const auto& [role, wj] : wit->items())
      c.weights[role] = parse_weight(wj, path + "/" + role);
  }
  if (const auto pit = j.find("params"); pit != j.end())
    c.params = parse_params(*pit, path);
  return c;
}

const json& required_object(const json& j, const char* key, const char* what) {
  const auto it = j.find(key);
  if (it == j.end() || !it->is_object())
    fail(std::string("manifest: missing or invalid '") + key + "' (" + what + ")");
  return *it;
}

}  // namespace

Manifest Manifest::parse(std::string_view text) {
  json j;
  try {
    j = json::parse(text);
  } catch (const json::exception& e) {
    throw ManifestError(std::string("manifest JSON: ") + e.what());
  }
  try {
    Manifest m;
    m.format_version = j.at("format_version").get<uint32_t>();
    if (m.format_version != 1)
      fail("manifest: unsupported format_version " + std::to_string(m.format_version));
    if (const auto p = j.find("params"); p != j.end()) m.params = parse_params(*p, "params");

    const json& bb = required_object(j, "backbone", "wiring scheme");
    m.backbone.type = bb.at("type").get<std::string>();
    if (const auto p = bb.find("params"); p != bb.end())
      m.backbone.params = parse_params(*p, "backbone");

    m.embedder = parse_component(required_object(j, "embedder", "component"), "embedder");
    m.lm_head = parse_component(required_object(j, "lm_head", "component"), "lm_head");

    const auto& bl = j.at("blocks");
    if (!bl.is_array() || bl.empty())
      fail("manifest: 'blocks' must be a non-empty array");
    for (size_t i = 0; i < bl.size(); ++i)
      m.blocks.push_back(parse_component(bl[i], "blocks." + std::to_string(i)));
    return m;
  } catch (const json::exception& e) {
    throw ManifestError(std::string("manifest: ") + e.what());
  }
}

std::vector<std::string> Manifest::validate(const TensorSource& source) const {
  std::vector<std::string> errors;

  std::vector<std::pair<std::string, const ComponentSpec*>> comps;
  comps.emplace_back("embedder", &embedder);
  for (size_t i = 0; i < blocks.size(); ++i)
    comps.emplace_back("blocks." + std::to_string(i), &blocks[i]);
  comps.emplace_back("lm_head", &lm_head);

  // Role-path index for alias resolution.
  std::map<std::string, const WeightRef*> roles;
  for (const auto& [path, spec] : comps)
    for (const auto& [role, ref] : spec->weights)
      roles[path + "/" + role] = &ref;

  std::map<std::string, std::string> claimed;  // tensor name -> claiming role path
  for (const auto& [path, spec] : comps) {
    for (const auto& [role, ref] : spec->weights) {
      const std::string rp = path + "/" + role;

      if (!ref.alias_of.empty()) {
        const auto it = roles.find(ref.alias_of);
        if (it == roles.end())
          errors.push_back(rp + ": alias target '" + ref.alias_of + "' does not exist");
        else if (!it->second->alias_of.empty())
          errors.push_back(rp + ": chained aliases are not supported");
        continue;
      }

      if (!source.contains(ref.name)) {
        errors.push_back(rp + ": tensor '" + ref.name + "' not found in weights");
        continue;
      }
      const TensorView v = source.get(ref.name);
      if (v.dtype != ref.dtype)
        errors.push_back(rp + ": dtype mismatch for '" + ref.name + "' (manifest " +
                         detail::dtype_name(ref.dtype) + ", file " +
                         detail::dtype_name(v.dtype) + ")");
      if (v.rows != ref.rows || v.cols != ref.cols)
        errors.push_back(rp + ": shape mismatch for '" + ref.name + "' (manifest [" +
                         std::to_string(ref.rows) + ", " + std::to_string(ref.cols) +
                         "], file [" + std::to_string(v.rows) + ", " +
                         std::to_string(v.cols) + "])");
      const auto [it, inserted] = claimed.emplace(ref.name, rp);
      if (!inserted)
        errors.push_back("tensor '" + ref.name + "' claimed by both " + it->second +
                         " and " + rp + " (use an alias for shared weights)");
    }
  }

  for (const auto& n : source.names())
    if (!claimed.count(n))
      errors.push_back("unclaimed tensor '" + n +
                       "': present in weights but absent from manifest");

  return errors;
}

}  // namespace alice
