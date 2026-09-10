// Minimal REPL PoC (design §12: for the PoC the CLI owns the whole loop).
// Reads lines from stdin, each line = a fresh session (reset + prefill +
// greedy decode until EOS or token cap). Commands: /quit.
//
// Usage: alice_repl <model_dir> <manifest.json> [max_new_tokens=64] [max_seq=8192]
//        alice_repl <model_dir> <manifest.json> <max_new_tokens> <max_seq> <prompt>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>  // std::min, std::ranges::find

#include "alice/manifest.hpp"
#include "alice/model.hpp"
#include "tokenizer.hpp"

namespace {

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

// Streaming decode helper: prints the bytes of `full` past *shown that form
// complete UTF-8 characters (a token boundary can split a character), then
// advances *shown. Byte-exact (embedded NULs possible) and flushed per call.
void print_new_utf8_suffix(const std::string& full, size_t* shown) {
  const size_t complete = utf8_complete_len(full);
  if (complete <= *shown) return;
  std::fwrite(full.data() + *shown, 1, complete - *shown, stdout);
  std::fflush(stdout);
  *shown = complete;
}

using Clock = std::chrono::steady_clock;
double elapsed_s(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Phase timing on stderr: total, count, average per token, tokens/second.
void print_stats(const char* phase, size_t tokens, double seconds) {
  if (tokens == 0) {
    std::fprintf(stderr, "[%s: 0 tokens in %.3f s]\n", phase, seconds);
    return;
  }
  std::fprintf(stderr, "[%s: %zu tokens in %.3f s — %.2f ms/token, %.1f tok/s]\n",
               phase, tokens, seconds, seconds * 1e3 / double(tokens),
               double(tokens) / seconds);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <model_dir> <manifest.json> [max_new_tokens=64] [max_seq=8192]\n"
                 "       %s <model_dir> <manifest.json> <max_new_tokens> <max_seq> <prompt> [hardware]\n"
                 "       (with a prompt argument: run once, non-interactive;\n"
                 "        hardware: auto (default) | gpu_rocblas | cpu_avx2 | cpu_generic)\n",
                 argv[0], argv[0]);
    return 1;
  }
  const std::string model_dir = argv[1];
  const std::string manifest_path = argv[2];
  const int max_new = argc > 3 ? std::atoi(argv[3]) : 64;
  const uint32_t max_seq = argc > 4 ? uint32_t(std::atoi(argv[4])) : 8192;
  const std::string hw_arg = argc > 6 ? argv[6] : "auto";
  const alice::HardwareClass hw =
      hw_arg == "cpu_generic" ? alice::HardwareClass::CPU_GENERIC
      : hw_arg == "cpu_avx2"  ? alice::HardwareClass::CPU_AVX2
      : hw_arg == "gpu_rocblas" ? alice::HardwareClass::GPU_ROCBLAS
                                : alice::HardwareClass::AUTO;

  try {
    const alice::Manifest manifest = alice::Manifest::parse(manifest_path);
    alice::frontend::Tokenizer tokenizer(model_dir + "/tokenizer.json");

    alice::EngineConfig cfg;
    cfg.max_seq_len = max_seq;
    cfg.max_prefill_chunk = 512;
    cfg.hardware = hw;
    cfg.sampling.temperature = 0.0f;  // greedy
    auto model = alice::ModelLoader::load(model_dir, manifest_path, cfg);
    std::fprintf(stderr, "[alice] model loaded (vocab %u, hardware %s). /quit to exit, session cleared on every round.\n",
                 model->vocab_size(), hw_arg.c_str());

    std::string line;
    for (;;) {
      std::fprintf(stderr, "\n>> ");
      std::cout.flush();
      if (argc > 5) line = argv[5];   // If prompt is supplied then skip user input
      else if (!std::getline(std::cin, line)) break;
      if (line == "/quit") break;
      if (line.empty()) continue;

      model->reset();  // PoC: every line is a fresh session

      // session policy: sequence starts at BOS; encode() is a pure text->ids function
      auto ids = tokenizer.encode(line);
      ids.insert(ids.begin(), int32_t(manifest.config.bos_token_id));
      if (ids.size() >= max_seq) {
        std::fprintf(stderr, "[prompt too long for max_seq]\n");
        continue;
      }

      // chunked prefill
      std::fprintf(stderr, "prefilling...\n");
      const auto prefill_t0 = Clock::now();
      int32_t t = -1;
      for (size_t off = 0; off < ids.size(); off += cfg.max_prefill_chunk) {
        const size_t n = std::min<size_t>(cfg.max_prefill_chunk, ids.size() - off);
        t = model->forward(std::span<const int32_t>(ids.data() + off, n));
      }
      print_stats("prefill", ids.size(), elapsed_s(prefill_t0));

      // decode, streaming only complete-UTF-8 suffixes
      const auto& eos_ids = manifest.config.eos_token_ids;
      const auto is_eos = [&](int32_t id) {  // C++20: ranges::contains is C++23
        return std::ranges::find(eos_ids, id) != eos_ids.end();
      };
      std::fprintf(stderr, "decoding...\n");
      const auto decode_t0 = Clock::now();
      size_t shown = 0;
      std::vector<int32_t> gen;
      for (int step = 0; step < max_new; ++step) {
        if (is_eos(t)) break;
        gen.push_back(t);
        print_new_utf8_suffix(tokenizer.decode(gen), &shown);
        const int32_t cur = t;
        t = model->forward(std::span<const int32_t>(&cur, 1));
      }
      std::printf("\n");
      std::fflush(stdout);  // keep stdout/stderr ordering sane for the stats line
      print_stats("decode", gen.size(), elapsed_s(decode_t0));
      if (argc > 5) break;  // prompt was supplied: no interactive mode
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
