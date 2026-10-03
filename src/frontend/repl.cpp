// Minimal REPL PoC (design §12: for the PoC the CLI owns the whole loop).
// Reads lines from stdin, each line = a fresh session (reset + prefill +
// greedy decode until EOS or token cap). Commands: /quit.
//
// Usage: alice_repl <model.model> [max_new_tokens=64] [max_seq=8192] [tp_degree=1] [prompt]
//        (with a prompt argument: run once, non-interactive)

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

#include "../engine.hpp"

namespace {

void print_new_text(const std::string& text) {
  std::fwrite(text.data(), 1, text.size(), stdout);
  std::fflush(stdout);
}

int run_rounds(alice::engine::Engine& engine, int max_new, uint32_t max_seq,
               const char* prompt, uint32_t chunk) {
  std::string line;
  for (;;) {
    std::fprintf(stderr, "\n>> ");
    std::cout.flush();
    if (prompt)
      line = prompt;
    else if (!std::getline(std::cin, line))
      break;
    if (line == "/quit") break;
    if (line.empty()) continue;

    const auto result = engine.generate(
        line, {max_new, max_seq, chunk},
        [](const std::string& text) { print_new_text(text); });

    std::printf("\n");
    std::fflush(stdout);
    (void)result;  // stats already printed by generate()

    if (prompt) break;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model.model> [max_new_tokens=64] [max_seq=8192] "
                 "[tp_degree=1] [prompt]\n"
                 "       (with a prompt argument: run once, non-interactive)\n",
                 argv[0]);
    return 1;
  }
  const std::string model_file = argv[1];
  const int max_new = argc > 2 ? std::atoi(argv[2]) : 64;
  const uint32_t max_seq = argc > 3 ? uint32_t(std::atoi(argv[3])) : 8192;
  const uint32_t tp_degree = argc > 4 ? uint32_t(std::atoi(argv[4])) : 1;

  try {
    alice::EngineConfig cfg;
    cfg.max_seq_len = max_seq;
    cfg.max_prefill_chunk = 512;
    cfg.sampling.temperature = 0.0f;
    cfg.tp_degree = tp_degree;
    alice::engine::Engine eng = alice::engine::init_engine(model_file, cfg);
    std::fprintf(stderr,
                 "[alice] model loaded (vocab %u, GPU-resident). /quit to exit, "
                 "session cleared on every round.\n",
                 eng.vocab_size());
    return run_rounds(eng, max_new, max_seq, argc > 5 ? argv[5] : nullptr,
                      cfg.max_prefill_chunk);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
