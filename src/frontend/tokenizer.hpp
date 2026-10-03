#pragma once

// Llama-3-style BPE tokenizer (frontend module — design §12: tokenization
// lives outside Model). Loads a HF tokenizer.json: Split pre-tokenizer
// (regex via PCRE2) + ByteLevel BPE model + added special tokens.
//
// Scope notes: encode() does NOT add special tokens (caller adds BOS etc.);
// decode() maps special ids to their literal content.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace alice::engine {
class Engine;  // allows Engine() to default-construct a Tokenizer before from_json
}

namespace alice::frontend {

class Tokenizer {
 public:
  // Throws std::runtime_error on IO/format problems.
  explicit Tokenizer(const std::string& tokenizer_json_path);
  // Construct from an in-memory tokenizer.json string (e.g. the bytes fetched
  // by the model-format loader, which are content-addressed in the .model).
  // bos_token_id and eos_token_ids are model-package metadata; the tokenizer
  // carries them so the engine's generation loop can add BOS and detect EOS
  // without pulling them out of the descriptor separately.
  static Tokenizer from_json(const std::string& tokenizer_json_text,
                             uint32_t bos_token_id = 0,
                             std::vector<uint32_t> eos_token_ids = {});
  ~Tokenizer();
  Tokenizer(Tokenizer&& o) noexcept;
  Tokenizer& operator=(Tokenizer&& o) noexcept;

  std::vector<int32_t> encode(const std::string& text) const;
  std::string decode(const std::vector<int32_t>& ids) const;
  // Single piece as raw bytes (may be a fragment of a UTF-8 sequence);
  // special ids return their literal content like "<|end_of_text|>".
  std::string decode_token(int32_t id) const;

  uint32_t vocab_size() const { return uint32_t(id_to_token_.size()); }

  uint32_t bos_token_id() const { return bos_token_id_; }
  const std::vector<uint32_t>& eos_token_ids() const { return eos_token_ids_; }

 private:
  friend class alice::engine::Engine;  // Engine() default-constructs then assigns via from_json
  Tokenizer() = default;
  void load(const std::string& text);  // parse + build (shared by both ctors)

  std::vector<int32_t> bpe_encode_span(const std::string& bytes) const;

  // GPT-2-style bytes<->alphabet-codepoint maps, built algorithmically.
  uint32_t byte_to_cp_[256]{};
  int32_t cp_to_byte_[324]{};  // -1 = not an alphabet codepoint

  std::unordered_map<std::string, int32_t> token_to_id_;
  std::vector<std::string> id_to_token_;  // base vocab (byte-alphabet UTF-8)
  std::unordered_map<int32_t, std::string> special_id_to_content_;
  std::unordered_map<std::string, int32_t> merges_;  // "A\x01" B -> rank
  struct PretokRegex* pretok_re_ = nullptr;  // PCRE2 (opaque, owned)

  uint32_t bos_token_id_ = 0;
  std::vector<uint32_t> eos_token_ids_;
};

}  // namespace alice::frontend
