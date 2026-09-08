#include <gtest/gtest.h>

#include "alice/manifest.hpp"

#include <map>

using namespace alice;

namespace {

// In-memory TensorSource for validation tests: metadata only, no data.
class FakeSource : public TensorSource {
 public:
  struct E {
    DType dt;
    uint32_t r, c;
  };
  std::map<std::string, E> m;

  bool contains(std::string_view n) const override { return m.count(std::string(n)); }
  TensorView get(std::string_view n) const override {
    const auto& e = m.at(std::string(n));
    return TensorView{nullptr, e.dt, e.r, e.c, 0};
  }
  std::vector<std::string> names() const override {
    std::vector<std::string> v;
    for (const auto& [k, _] : m) v.push_back(k);
    return v;
  }
};

const char* kManifest = R"JSON({
  "format_version": 1,
  "params": {"rms_eps": 1e-5},
  "backbone": {"type": "add"},
  "embedder": {"type": "token_embedding",
    "weights": {"table": {"name": "emb.weight", "shape": [10, 4], "dtype": "F32"}}},
  "blocks": [
    {"type": "ffn", "weights": {"w": {"name": "b0.w", "shape": [4, 4], "dtype": "BF16"}}}
  ],
  "lm_head": {"type": "linear_head",
    "weights": {"w": {"alias_of": "embedder/table"}}}
})JSON";

FakeSource good_source() {
  FakeSource s;
  s.m["emb.weight"] = {DType::F32, 10, 4};
  s.m["b0.w"] = {DType::BF16, 4, 4};
  return s;
}

bool has_error_containing(const std::vector<std::string>& errs, const std::string& sub) {
  for (const auto& e : errs)
    if (e.find(sub) != std::string::npos) return true;
  return false;
}

}  // namespace

TEST(Manifest, ParsesAndValidates) {
  const Manifest m = Manifest::parse(kManifest);
  EXPECT_EQ(m.format_version, 1u);
  EXPECT_EQ(m.backbone.type, "add");
  EXPECT_EQ(m.blocks.size(), 1u);
  EXPECT_EQ(m.blocks[0].type, "ffn");
  EXPECT_TRUE(m.lm_head.weights.at("w").alias_of == "embedder/table");
  EXPECT_TRUE(m.validate(good_source()).empty());
}

TEST(Manifest, MissingTensorReported) {
  FakeSource s;
  s.m["emb.weight"] = {DType::F32, 10, 4};
  // b0.w absent
  const auto errs = Manifest::parse(kManifest).validate(s);
  EXPECT_TRUE(has_error_containing(errs, "b0.w"));
  EXPECT_TRUE(has_error_containing(errs, "not found"));
}

TEST(Manifest, ShapeMismatchReported) {
  FakeSource s = good_source();
  s.m["b0.w"] = {DType::BF16, 4, 8};
  const auto errs = Manifest::parse(kManifest).validate(s);
  EXPECT_TRUE(has_error_containing(errs, "shape mismatch"));
}

TEST(Manifest, DtypeMismatchReported) {
  FakeSource s = good_source();
  s.m["b0.w"] = {DType::F32, 4, 4};
  const auto errs = Manifest::parse(kManifest).validate(s);
  EXPECT_TRUE(has_error_containing(errs, "dtype mismatch"));
}

TEST(Manifest, UnclaimedTensorReported) {
  FakeSource s = good_source();
  s.m["mystery.tensor"] = {DType::F32, 1, 1};
  const auto errs = Manifest::parse(kManifest).validate(s);
  EXPECT_TRUE(has_error_containing(errs, "unclaimed tensor 'mystery.tensor'"));
}

TEST(Manifest, BadAliasTargetReported) {
  const std::string bad = std::string(kManifest).replace(
      std::string(kManifest).find("embedder/table"), 14, "embedder/nope");
  const auto errs = Manifest::parse(bad).validate(good_source());
  EXPECT_TRUE(has_error_containing(errs, "alias target"));
}

TEST(Manifest, DuplicateClaimReported) {
  const char* dup = R"JSON({
    "format_version": 1,
    "backbone": {"type": "add"},
    "embedder": {"type": "token_embedding",
      "weights": {"table": {"name": "emb.weight", "shape": [10, 4], "dtype": "F32"},
                  "other": {"name": "emb.weight", "shape": [10, 4], "dtype": "F32"}}},
    "blocks": [{"type": "ffn"}],
    "lm_head": {"type": "linear_head"}
  })JSON";
  FakeSource s;
  s.m["emb.weight"] = {DType::F32, 10, 4};
  const auto errs = Manifest::parse(dup).validate(s);
  EXPECT_TRUE(has_error_containing(errs, "claimed by both"));
}

TEST(Manifest, ParseErrorsThrow) {
  EXPECT_THROW(Manifest::parse("{not json"), ManifestError);
  EXPECT_THROW(Manifest::parse(R"({"format_version": 99})"), ManifestError);
  EXPECT_THROW(Manifest::parse(R"({"format_version": 1})"), ManifestError);
}
