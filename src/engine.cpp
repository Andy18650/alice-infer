// The alice model format loader (docs/model-format.md RC1,
// docs/model-loader.md §8 scope). init_engine is the public entry: it
// reads the .model envelope (magic/framing/signature-state), extracts the
// signed metadata + non-device members (tokenizer, chat_template), and
// loads the device model via gpu::Model::load_from_format — unifying the
// model-format path and the manifest path on one runnable gpu::Model type.
// Signature verification is deferred with the chain of trust: the
// envelope's signature state is DECLARED loudly, not checked.

#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <functional>
#include <span>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "data_loader.hpp"  // LocalFileDataLoader (content-addressed fetch)
#include "gpu/gpu_model.hpp"  // gpu::Model (load_from_format + ~Engine)
#include "sampler.hpp"  // host-side Sampler (moved from gpu::Model)

namespace alice {

using nlohmann::json;

namespace {   // private namespace for private helper stuff

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("model loader: " + what);
}

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }
};

// Descriptor parses via from_json (header); json exceptions carry the key
// name (JSON_DIAGNOSTICS), converted once at the init_engine boundary. Only
// SEMANTIC checks hand-build context here (version, unknown type/codec,
// hash mismatches): things the json library cannot know.

// Reads a .model descriptor file and returns the parsed envelope: text
// region (everything before the first NUL, which must be \n- or \r\n-
// terminated; a binary tail may follow in single-file models, so no
// whole-file read), {"ALICE" magic, then the json parse. The raw text is
// local — only the parsed envelope crosses out (the string is freed once
// the json is constructed). Descriptor budget: real ones measure <100 KiB
// (the 70B-with-MoE projects to ~32 KiB); the 1 MiB read cap exists to
// catch "wrong file pointed at the loader", not as a designed limit.
json read_envelope(const std::filesystem::path& file) {
  constexpr size_t kMaxDescriptor = 1 << 20;

  std::ifstream in(file, std::ios::binary);
  if (!in) fail("cannot open '" + file.string() + "'");
  std::string text(kMaxDescriptor, '\0');
  in.read(text.data(), kMaxDescriptor);  // partial read at EOF is fine
  const size_t got = size_t(in.gcount());

  // Magic first: a wrong file should fail HERE, not on some
  // incidental terminator detail deeper in.
  if (got < 8 || text.compare(0, 8, "{\"ALICE\"") != 0)
    fail("bad magic: a .model file must begin with {\"ALICE\"");

  const size_t nul = text.find('\0');
  if (nul == std::string::npos || nul >= got)
    fail("'" + file.string() + "' has no NUL within the first 1 MiB — "
         "not a .model file?");
  if (nul == 0 || text[nul - 1] != '\n')
    fail("text region must end with \\n before the NUL gate");
  size_t end = nul - 1;  // drop the \n
  if (end > 0 && text[end - 1] == '\r') --end;  // tolerate \r\n
  text.resize(end);

  json envelope;
  try {
    envelope = json::parse(text);
  } catch (const json::exception& e) {
    fail(std::string("envelope JSON: ") + e.what());
  }
  return envelope;
}

}  // namespace

namespace engine {

// init_engine: the envelope half. Reads the .model file (magic/framing),
// validates + parses the descriptor, declares the signature state, builds
// the content-addressed DataLoader from the unsigned `files` table, fetches
// the non-device members (tokenizer, chat_template) + signed metadata, then
// hands the descriptor + DataLoader to Model::load_from_format for the
// device half.
Engine init_engine_impl(const std::filesystem::path& model_file,
                        const EngineConfig& ec, LoadStats& st) {
  Engine out;

  Timer t;
  const json envelope = read_envelope(model_file);
  {     // checking the envelope
    if (!envelope.is_object()) fail("envelope must be an object");
    const json& alice = envelope.at("ALICE");
    if (!alice.is_object()) fail("envelope.ALICE must be an object");
    if (alice.at("format_version").get<uint64_t>() != 1)
      fail("envelope.ALICE: unsupported format_version (this loader reads 1)");
    if (envelope.contains("signature") && !envelope.at("signature").is_null()) {
      std::string key = "?", alg = "?";
      const json& s = envelope.at("signature");
      if (s.is_object()) {
        if (s.contains("key")) key = s.at("key").get<std::string>();
        if (s.contains("alg")) alg = s.at("alg").get<std::string>();
      }
      std::fprintf(stderr,
                  "[model loader] descriptor carries a signature (alg=%s key=%s) — "
                  "verification not implemented (chain-of-trust round); proceeding\n",
                  alg.c_str(), key.c_str());
    } else {
      std::fprintf(stderr,
                  "[model loader] unsigned descriptor — proceeding (dev artifacts only)\n");
    }
  }

  const json& model = envelope.at("model");
  if (!model.is_object()) fail("envelope.model must be an object");
  Descriptor descriptor = model.get<Descriptor>();
  if (descriptor.version != 1)
    fail("descriptor: unsupported version " + std::to_string(descriptor.version) +
         " (this loader reads version 1)");
  st.parse_ms = t.ms();
  st.specs = descriptor.specs.size();

  // The bytes source: content-addressed DataLoader from the unsigned `files`
  // table (location separated from identity, per the format).
  const std::filesystem::path base = model_file.parent_path();
  const loader::LocalFileDataLoader data_loader(envelope.at("files"),
                                                 base.string());

  // Non-device members + signed metadata, retained for the session/REPL
  // layer (not part of the device forward path). Optional: absent -> empty.
  auto fetch_string = [&](const char* key) -> std::string {
    const auto it = descriptor.members.find(key);
    if (it == descriptor.members.end()) return {};
    std::string s;
    data_loader.fetch(it->second.id, [&](std::span<const std::byte> blob) {
      s.assign(reinterpret_cast<const char*>(blob.data()), blob.size());
    });
    return s;
  };
  out.chat_template_ = fetch_string("chat_template");

  // Tokenizer: construct it here so the engine owns both the vocab and the
  // BOS/EOS metadata (pulled from the descriptor's tokenizer member config).
  uint32_t bos_token_id = 0;
  std::vector<uint32_t> eos_token_ids;
  std::string tokenizer_json;
  if (const auto it = descriptor.members.find("tokenizer");
      it != descriptor.members.end()) {
    data_loader.fetch(it->second.id, [&](std::span<const std::byte> blob) {
      tokenizer_json.assign(reinterpret_cast<const char*>(blob.data()), blob.size());
    });
    const json& tc = it->second.config;
    if (tc.contains("bos_token_id"))
      bos_token_id = tc.at("bos_token_id").get<uint32_t>();
    if (tc.contains("eos_token_id")) {
      const json& e = tc.at("eos_token_id");
      if (e.is_array())
        for (const auto& x : e) eos_token_ids.push_back(x.get<uint32_t>());
      else
        eos_token_ids.push_back(e.get<uint32_t>());
    }
  }
  out.tokenizer_ = frontend::Tokenizer::from_json(tokenizer_json, bos_token_id,
                                                 std::move(eos_token_ids));

  out.name_ = descriptor.name;
  out.producer_ = descriptor.producer;
  out.packager_ = descriptor.packager;
  out.license_ = descriptor.license;
  out.license_url_ = descriptor.license_url;
  out.license_text_ = descriptor.license_text;
  out.max_context_ = descriptor.max_context;

  out.model_ = gpu::Model::load_from_format(descriptor, data_loader, ec, st);
  out.sampler_ = std::make_unique<Sampler>(ec.sampling);
  return out;
}

Engine init_engine(const std::filesystem::path& model_file,
                   const EngineConfig& ec, LoadStats* out_stats) {
  LoadStats dummy;
  try {
    return init_engine_impl(model_file, ec, out_stats ? *out_stats : dummy);
  } catch (const json::exception& e) {
    fail(std::string("descriptor: ") + e.what());
  }
}

// Out-of-line: the unique_ptr<Model> deleter needs Model complete
// (gpu_model.hpp, included above). Defined here so engine.hpp stays
// free of the vendor/internal includes Model drags in.
Engine::~Engine() = default;

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_s(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

void print_stats(const char* phase, size_t tokens, double seconds) {
  if (tokens == 0) {
    std::fprintf(stderr, "[%s: 0 tokens in %.3f s]\n", phase, seconds);
    return;
  }
  std::fprintf(stderr,
               "[%s: %zu tokens in %.3f s — %.2f ms/token, %.1f tok/s]\n",
               phase, tokens, seconds, seconds * 1e3 / double(tokens),
               double(tokens) / seconds);
}

// Length of the longest complete-UTF-8 prefix of s.
size_t utf8_complete_len(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    const auto b = uint8_t(s[i]);
    const size_t len =
        b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : 4;
    if (i + len > s.size()) break;
    i += len;
  }
  return i;
}

}  // namespace

std::vector<int32_t> Engine::encode(const std::string& text) const {
  return tokenizer_.encode(text);
}

std::string Engine::decode(const std::vector<int32_t>& ids) const {
  return tokenizer_.decode(ids);
}

std::vector<float> Engine::forward_logits(std::span<const int32_t> tokens) {
  return model_->forward_logits(tokens);
}

std::vector<float> Engine::forward_logits_all(std::span<const int32_t> tokens) {
  return model_->forward_logits_all(tokens);
}

int32_t Engine::forward(std::span<const int32_t> tokens) {
  const auto logits = model_->forward_logits(tokens);
  return sampler_->sample(logits.data(), static_cast<uint32_t>(logits.size()));
}

void Engine::reset() { model_->reset(); }

Engine::GenerateResult Engine::generate(
    const std::string& prompt, const GenerateOptions& opts,
    std::function<void(const std::string&)> on_token) {
  GenerateResult result;
  model_->reset();

  // Encode prompt and prepend BOS.
  auto ids = tokenizer_.encode(prompt);
  ids.insert(ids.begin(), int32_t(tokenizer_.bos_token_id()));
  if (ids.size() >= opts.max_seq_len)
    throw std::runtime_error("generate: prompt too long for max_seq_len");
  result.prompt_tokens = ids.size();

  const auto is_eos = [&](int32_t id) {
    const auto& eos = tokenizer_.eos_token_ids();
    return std::find(eos.begin(), eos.end(), id) != eos.end();
  };

  // Chunked prefill: run all but the last chunk without D2H, then pull
  // logits for the final chunk and sample on the host.
  const auto prefill_t0 = Clock::now();
  int32_t t = -1;
  for (size_t off = 0; off < ids.size(); off += opts.max_prefill_chunk) {
    const size_t n = std::min<size_t>(opts.max_prefill_chunk, ids.size() - off);
    const bool last = off + n >= ids.size();
    if (last) {
      const auto logits = model_->forward_logits(std::span<const int32_t>(ids.data() + off, n));
      t = sampler_->sample(logits.data(), static_cast<uint32_t>(logits.size()));
    } else {
      model_->forward_run(std::span<const int32_t>(ids.data() + off, n));
    }
  }
  result.prefill_seconds = elapsed_s(prefill_t0);
  print_stats("prefill", ids.size(), result.prefill_seconds);

  // Decode, streaming complete-UTF-8 suffixes.
  const auto decode_t0 = Clock::now();
  size_t shown = 0;
  std::vector<int32_t> gen;
  for (int step = 0; step < opts.max_new_tokens; ++step) {
    if (is_eos(t)) break;
    gen.push_back(t);
    if (on_token) {
      const std::string full = tokenizer_.decode(gen);
      const size_t complete = utf8_complete_len(full);
      if (complete > shown) {
        on_token(full.substr(shown, complete - shown));
        shown = complete;
      }
    }
    const auto logits = model_->forward_logits(std::span<const int32_t>(&t, 1));
    t = sampler_->sample(logits.data(), static_cast<uint32_t>(logits.size()));
  }
  // Flush any remaining complete bytes.
  if (on_token && !gen.empty()) {
    const std::string full = tokenizer_.decode(gen);
    const size_t complete = utf8_complete_len(full);
    if (complete > shown) on_token(full.substr(shown, complete - shown));
  }
  result.text = tokenizer_.decode(gen);
  result.generated_tokens = gen.size();
  result.decode_seconds = elapsed_s(decode_t0);
  print_stats("decode", gen.size(), result.decode_seconds);
  return result;
}

}  // namespace engine

}  // namespace alice
