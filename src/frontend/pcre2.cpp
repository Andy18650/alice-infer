#include "pcre2.hpp"

#include <dlfcn.h>

#include <cstdint>
#include <stdexcept>

namespace alice::frontend {

namespace {

// Self-declared PCRE2 8-bit API (subset), resolved from libpcre2-8.so.0.
using Pcre2Compile = void* (*)(const uint8_t*, size_t, uint32_t, int*, size_t*, void*);
using Pcre2MatchDataCreate = void* (*)(const void*, void*);
using Pcre2Match = int (*)(const void*, const uint8_t*, size_t, size_t, uint32_t, void*,
                           void*);
using Pcre2GetOvector = size_t* (*)(const void*);
using Pcre2Free = void (*)(void*);

constexpr uint32_t kOptUtf = 0x00080000;  // PCRE2_UTF
constexpr uint32_t kOptUcp = 0x00020000;  // PCRE2_UCP
constexpr int kErrorNoMatch = -1;         // PCRE2_ERROR_NOMATCH

struct Api {
  Pcre2Compile compile = nullptr;
  Pcre2MatchDataCreate md_create = nullptr;
  Pcre2Match match = nullptr;
  Pcre2GetOvector ovector = nullptr;
  Pcre2Free code_free = nullptr;
  Pcre2Free md_free = nullptr;
  void* handle = nullptr;

  static const Api& get() {
    static Api api = [] {
      Api a;
      a.handle = dlopen("libpcre2-8.so.0", RTLD_NOW | RTLD_LOCAL);
      if (!a.handle) throw std::runtime_error("pcre2: cannot load libpcre2-8.so.0");
      auto sym = [&](const char* name) {
        void* p = dlsym(a.handle, name);
        if (!p) throw std::runtime_error(std::string("pcre2: missing symbol ") + name);
        return p;
      };
      a.compile = reinterpret_cast<Pcre2Compile>(sym("pcre2_compile_8"));
      a.md_create = reinterpret_cast<Pcre2MatchDataCreate>(sym("pcre2_match_data_create_from_pattern_8"));
      a.match = reinterpret_cast<Pcre2Match>(sym("pcre2_match_8"));
      a.ovector = reinterpret_cast<Pcre2GetOvector>(sym("pcre2_get_ovector_pointer_8"));
      a.code_free = reinterpret_cast<Pcre2Free>(sym("pcre2_code_free_8"));
      a.md_free = reinterpret_cast<Pcre2Free>(sym("pcre2_match_data_free_8"));
      return a;
    }();
    return api;
  }
};

}  // namespace

Pcre2Regex::Pcre2Regex(const std::string& pattern) {
  const Api& a = Api::get();
  int err = 0;
  size_t erroff = 0;
  code_ = a.compile(reinterpret_cast<const uint8_t*>(pattern.data()), pattern.size(),
                    kOptUtf | kOptUcp, &err, &erroff, nullptr);
  if (!code_)
    throw std::runtime_error("pcre2: compile failed (code " + std::to_string(err) +
                             ") at offset " + std::to_string(erroff) + " in: " + pattern);
  match_data_ = a.md_create(code_, nullptr);
  if (!match_data_) throw std::runtime_error("pcre2: match_data_create failed");
}

Pcre2Regex::~Pcre2Regex() {
  const Api& a = Api::get();
  if (match_data_) a.md_free(match_data_);
  if (code_) a.code_free(code_);
}

bool Pcre2Regex::next(const std::string& subject, size_t* offset, size_t* begin,
                      size_t* end) const {
  const Api& a = Api::get();
  int rc = a.match(code_, reinterpret_cast<const uint8_t*>(subject.data()), subject.size(),
                   *offset, 0, match_data_, nullptr);
  if (rc == kErrorNoMatch) return false;
  if (rc < 0) throw std::runtime_error("pcre2: match error " + std::to_string(rc));
  const size_t* ov = a.ovector(match_data_);
  *begin = ov[0];
  *end = ov[1];
  // guard against empty matches looping forever (pattern shouldn't produce any)
  *offset = (*end > *begin) ? *end : *end + 1;
  return true;
}

}  // namespace alice::frontend
