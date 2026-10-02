#include "tokenizer.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace alice::frontend {

namespace {

// Pre-tokenizer regex: PCRE2 8-bit, compiled with UTF + UCP (the Llama
// split pattern needs Unicode property classes). Owning wrapper; the
// match_data is per-pattern, reused across next() calls (Tokenizer is
// const-shared across encode/decode).
void utf8_append(uint32_t cp, std::string& out) {
  if (cp < 0x80) {
    out += char(cp);
  } else if (cp < 0x800) {
    out += char(0xC0 | (cp >> 6));
    out += char(0x80 | (cp & 0x3F));
  } else {
    out += char(0xE0 | (cp >> 12));
    out += char(0x80 | ((cp >> 6) & 0x3F));
    out += char(0x80 | (cp & 0x3F));
  }
}

// Decodes the next codepoint; malformed input is replaced by U+FFFD-ish
// behavior: the raw byte is returned as-is (callers only feed well-formed
// alphabet/vocab strings).
uint32_t utf8_next(const std::string& s, size_t* i) {
  const auto b0 = uint8_t(s[*i]);
  if (b0 < 0x80) {
    ++*i;
    return b0;
  }
  if ((b0 & 0xE0) == 0xC0) {
    const uint32_t cp = ((b0 & 0x1F) << 6) | (uint8_t(s[*i + 1]) & 0x3F);
    *i += 2;
    return cp;
  }
  const uint32_t cp = ((b0 & 0x0F) << 12) | ((uint8_t(s[*i + 1]) & 0x3F) << 6) |
                      (uint8_t(s[*i + 2]) & 0x3F);
  *i += 3;
  return cp;
}

}  // namespace

struct PretokRegex {
  pcre2_code* code = nullptr;
  pcre2_match_data* match_data = nullptr;

  explicit PretokRegex(const std::string& pattern) {
    int err = 0;
    PCRE2_SIZE erroff = 0;
    code = pcre2_compile(PCRE2_SPTR(pattern.data()), pattern.size(),
                         PCRE2_UTF | PCRE2_UCP, &err, &erroff, nullptr);
    if (!code) {
      PCRE2_UCHAR8 msg[256];
      (void)pcre2_get_error_message(err, msg, sizeof(msg));
      throw std::runtime_error("Tokenizer: pre-tokenizer regex compile failed (code " +
                               std::to_string(err) + ") at offset " +
                               std::to_string(erroff) + ": " +
                               reinterpret_cast<const char*>(msg));
    }
    match_data = pcre2_match_data_create_from_pattern(code, nullptr);
    if (!match_data) throw std::runtime_error("Tokenizer: match_data_create failed");
  }
  ~PretokRegex() {
    pcre2_match_data_free(match_data);
    pcre2_code_free(code);
  }
  PretokRegex(const PretokRegex&) = delete;
  PretokRegex& operator=(const PretokRegex&) = delete;

  // Finds the next match at or after *offset. On match: sets *begin/*end,
  // advances *offset past the match, returns true. Returns false at NOMATCH.
  bool next(const std::string& subject, size_t* offset, size_t* begin, size_t* end) const {
    const int rc = pcre2_match(code, PCRE2_SPTR(subject.data()), subject.size(), *offset,
                               0, match_data, nullptr);
    if (rc == PCRE2_ERROR_NOMATCH) return false;
    if (rc < 0)
      throw std::runtime_error("Tokenizer: pre-tokenizer match error " + std::to_string(rc));
    const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(match_data);
    *begin = ov[0];
    *end = ov[1];
    // guard against empty matches looping forever (pattern shouldn't produce any)
    *offset = (*end > *begin) ? *end : *end + 1;
    return true;
  }
};

Tokenizer::Tokenizer(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Tokenizer: cannot open " + path);
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  load(text);
}

Tokenizer Tokenizer::from_json(const std::string& json_text) {
  Tokenizer t;
  t.load(json_text);
  return t;
}

void Tokenizer::load(const std::string& text) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(std::string("Tokenizer: ") + e.what());
  }

  // --- byte alphabet (GPT-2 bytes_to_unicode) ---
  for (auto& x : cp_to_byte_) x = -1;
  std::vector<bool> printable(256, false);
  for (uint32_t b = 0x21; b <= 0x7E; ++b) printable[b] = true;   // '!'..'~'
  for (uint32_t b = 0xA1; b <= 0xAC; ++b) printable[b] = true;   // '¡'..'¬'
  for (uint32_t b = 0xAE; b <= 0xFF; ++b) printable[b] = true;   // '®'..'ÿ'
  uint32_t n = 0;
  for (uint32_t b = 0; b < 256; ++b) {
    const uint32_t cp = printable[b] ? b : 256 + n++;
    byte_to_cp_[b] = cp;
    cp_to_byte_[cp] = int32_t(b);
  }

  // --- model: vocab + merges ---
  const auto& model = j.at("model");
  if (model.at("type").get<std::string>() != "BPE")
    throw std::runtime_error("Tokenizer: model type is not BPE");
  for (auto& [tok, id] : model.at("vocab").items()) {
    const int32_t i = id.get<int32_t>();
    token_to_id_[tok] = i;
    if (size_t(i) >= id_to_token_.size()) id_to_token_.resize(size_t(i) + 1);
    id_to_token_[i] = tok;
  }
  int32_t rank = 0;
  for (const auto& m : model.at("merges")) {
    const std::string s = m.get<std::string>();
    const auto sp = s.find(' ');
    if (sp == std::string::npos)
      throw std::runtime_error("Tokenizer: malformed merge '" + s + "'");
    // ' ' cannot appear inside a token (space byte maps to Ġ), so one split
    merges_[s.substr(0, sp) + '\x01' + s.substr(sp + 1)] = rank++;
  }

  for (const auto& at : j.at("added_tokens"))
    special_id_to_content_[at.at("id").get<int32_t>()] = at.at("content").get<std::string>();

  // --- pre-tokenizer regex ---
  // Expect Sequence[Split{pattern{Regex}}, ByteLevel...]; dig out the regex.
  std::string pattern;
  for (const auto& pt : j.at("pre_tokenizer").at("pretokenizers"))
    if (pt.value("type", "") == "Split") pattern = pt.at("pattern").at("Regex").get<std::string>();
  if (pattern.empty()) throw std::runtime_error("Tokenizer: no Split pre-tokenizer found");
  pretok_re_ = std::make_unique<PretokRegex>(pattern);
}

Tokenizer::~Tokenizer() = default;

std::vector<int32_t> Tokenizer::bpe_encode_span(const std::string& bytes) const {
  // bytes -> alphabet symbols (single-codepoint UTF-8 strings)
  std::vector<std::string> syms;
  syms.reserve(bytes.size());
  for (const unsigned char b : bytes) {
    std::string s;
    utf8_append(byte_to_cp_[b], s);
    syms.push_back(std::move(s));
  }
  // merge by rank, leftmost lowest first
  for (;;) {
    int32_t best_rank = INT32_MAX;
    size_t best_i = 0;
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
      const auto it = merges_.find(syms[i] + '\x01' + syms[i + 1]);
      if (it != merges_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_rank == INT32_MAX) break;
    syms[best_i] += syms[best_i + 1];
    syms.erase(syms.begin() + best_i + 1);
  }
  std::vector<int32_t> ids;
  ids.reserve(syms.size());
  for (const auto& s : syms) {
    const auto it = token_to_id_.find(s);
    if (it == token_to_id_.end())
      throw std::runtime_error("Tokenizer: BPE produced a symbol missing from vocab");
    ids.push_back(it->second);
  }
  return ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
  std::vector<int32_t> ids;
  size_t offset = 0, begin = 0, end = 0;
  while (pretok_re_->next(text, &offset, &begin, &end)) {
    auto span_ids = bpe_encode_span(text.substr(begin, end - begin));
    ids.insert(ids.end(), span_ids.begin(), span_ids.end());
  }
  return ids;
}

std::string Tokenizer::decode_token(int32_t id) const {
  if (const auto it = special_id_to_content_.find(id); it != special_id_to_content_.end())
    return it->second;
  if (id < 0 || size_t(id) >= id_to_token_.size())
    throw std::out_of_range("Tokenizer::decode_token");
  return id_to_token_[id];
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string alphabet;
  for (const int32_t id : ids) {
    if (special_id_to_content_.count(id)) continue;  // specials contribute no bytes
    alphabet += decode_token(id);
  }
  // alphabet codepoints -> raw bytes
  std::string out;
  size_t i = 0;
  while (i < alphabet.size()) {
    const uint32_t cp = utf8_next(alphabet, &i);
    if (cp < 324 && cp_to_byte_[cp] >= 0) {
      out += char(cp_to_byte_[cp]);
    } else {
      utf8_append(cp, out);  // not an alphabet cp (shouldn't happen); pass through
    }
  }
  return out;
}

}  // namespace alice::frontend
