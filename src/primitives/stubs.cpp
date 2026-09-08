#include "alice/primitives.hpp"

#include "alice/context.hpp"

#include <stdexcept>
#include <string>

// Operator implementations land behind these signatures one at a time
// (naive reference first, then optimized/backends). Until then every
// primitive is an explicit trap, never a silent wrong answer.
namespace alice::prim {

namespace {
[[noreturn]] void nyi(const char* f) {
  throw std::logic_error(std::string("alice::prim::") + f + ": not implemented");
}
}  // namespace

void matmul(TensorView, TensorView, TensorView) { nyi("matmul"); }
void add(TensorView, TensorView) { nyi("add"); }
void rmsnorm(TensorView, TensorView, TensorView, float) { nyi("rmsnorm"); }
void silu_mul(TensorView, TensorView, TensorView) { nyi("silu_mul"); }
void rope(TensorView, TensorView, const RopeTables&, size_t) { nyi("rope"); }
void attention(TensorView, TensorView, KvSlot&, const AttentionParams&) { nyi("attention"); }

}  // namespace alice::prim
