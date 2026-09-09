// Minimal REPL PoC (design §12: for the PoC the CLI owns the whole loop).
// Reads lines from stdin, each line = a fresh session (reset + prefill +
// greedy decode until EOS or token cap). Commands: /quit, /reset.
//
// Usage: alice_repl <model_dir> <manifest.json> [max_new_tokens=64] [max_seq=8192]

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "alice/manifest.hpp"
#include "alice/model.hpp"
#include "tokenizer.hpp"

namespace {

bool is_eos(int32_t t, const std::vector<uint32_t>& eos) {
  for (const uint32_t e : eos)
    if (uint32_t(t) == e) return true;
  return false;
}

// Length of the longest complete-UTF-8 prefix (tokens can split characters).
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

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <model_dir> <manifest.json> [max_new_tokens=64] [max_seq=8192]\n",
                 argv[0]);
    return 1;
  }
  const std::string model_dir = argv[1];
  const std::string manifest_path = argv[2];
  const int max_new = argc > 3 ? std::atoi(argv[3]) : 64;
  const uint32_t max_seq = argc > 4 ? uint32_t(std::atoi(argv[4])) : 8192;

  try {
    const alice::Manifest manifest = alice::Manifest::parse(manifest_path);
    alice::frontend::Tokenizer tokenizer(model_dir + "/tokenizer.json");

    alice::EngineConfig cfg;
    cfg.max_seq_len = max_seq;
    cfg.max_prefill_chunk = 512;
    cfg.sampling.temperature = 0.0f;  // greedy
    auto model = alice::ModelLoader::load(model_dir, manifest_path, cfg);
    std::fprintf(stderr, "[alice] model loaded (vocab %u). /quit to exit, /reset to clear.\n",
                 model->vocab_size());

    std::string line;
    for (;;) {
      std::fprintf(stderr, "\n>> ");
      std::cout.flush();
      if (!std::getline(std::cin, line)) break;
      if (line == "/quit") break;
      if (line == "/reset") {
        model->reset();
        std::fprintf(stderr, "[state cleared]\n");
        continue;
      }
      if (line.empty()) continue;

      model->reset();  // PoC: every line is a fresh session
      std::vector<int32_t> ids = {int32_t(manifest.config.bos_token_id)};
      for (const int32_t t : tokenizer.encode(line)) ids.push_back(t);
      if (ids.size() >= max_seq) {
        std::fprintf(stderr, "[prompt too long for max_seq]\n");
        continue;
      }

      // chunked prefill
      int32_t t = -1;
      for (size_t off = 0; off < ids.size(); off += cfg.max_prefill_chunk) {
        const size_t n = std::min<size_t>(cfg.max_prefill_chunk, ids.size() - off);
        t = model->forward(std::span<const int32_t>(ids.data() + off, n));
      }

      // decode, streaming only complete-UTF-8 suffixes
      size_t shown = 0;
      std::vector<int32_t> gen;
      for (int step = 0; step < max_new; ++step) {
        if (is_eos(t, manifest.config.eos_token_ids)) break;
        gen.push_back(t);
        const std::string full = tokenizer.decode(gen);
        const size_t complete = utf8_complete_len(full);
        if (complete > shown) {
          std::fwrite(full.data() + shown, 1, complete - shown, stdout);
          std::fflush(stdout);
          shown = complete;
        }
        const int32_t cur = t;
        t = model->forward(std::span<const int32_t>(&cur, 1));
      }
      std::printf("\n");
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
