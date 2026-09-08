#pragma once

// Internal helpers shared by the loaders. Not part of the public API.

#include "alice/tensor.hpp"
#include <string>

namespace alice::detail {

inline const char* dtype_name(DType t) {
  switch (t) {
    case DType::BF16: return "BF16";
    case DType::F32:  return "F32";
  }
  return "?";
}

inline bool dtype_from_string(const std::string& s, DType& out) {
  if (s == "BF16") { out = DType::BF16; return true; }
  if (s == "F32")  { out = DType::F32;  return true; }
  return false;
}

}  // namespace alice::detail
