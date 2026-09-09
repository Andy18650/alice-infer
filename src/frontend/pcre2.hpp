#pragma once

// Minimal PCRE2 (8-bit) wrapper, loaded with dlopen at runtime — the dev
// package is unavailable on some machines and the runtime soname is nearly
// always present (frontend-only dependency, mirrors the project's
// self-declared extern "C" policy). Throws std::runtime_error if the library
// or a symbol is missing.

#include <cstddef>
#include <string>

namespace alice::frontend {

class Pcre2Regex {
 public:
  Pcre2Regex() = default;
  // pattern compiled with PCRE2_UTF | PCRE2_UCP. Throws on load/compile error.
  explicit Pcre2Regex(const std::string& pattern);
  ~Pcre2Regex();
  Pcre2Regex(const Pcre2Regex&) = delete;
  Pcre2Regex& operator=(const Pcre2Regex&) = delete;
  Pcre2Regex(Pcre2Regex&& o) noexcept : code_(o.code_), match_data_(o.match_data_) {
    o.code_ = nullptr;
    o.match_data_ = nullptr;
  }
  Pcre2Regex& operator=(Pcre2Regex&& o) noexcept {
    if (this != &o) {
      this->~Pcre2Regex();
      code_ = o.code_;
      match_data_ = o.match_data_;
      o.code_ = nullptr;
      o.match_data_ = nullptr;
    }
    return *this;
  }

  // Finds the next match at or after *offset. On match: sets *begin/*end,
  // advances *offset past the match, returns true. Returns false at NOMATCH.
  bool next(const std::string& subject, size_t* offset, size_t* begin,
            size_t* end) const;

 private:
  void* code_ = nullptr;
  void* match_data_ = nullptr;
};

}  // namespace alice::frontend
