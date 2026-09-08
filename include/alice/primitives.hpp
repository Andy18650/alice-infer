#pragma once

#include "alice/tensor.hpp"
#include <cstddef>
#include <cstdint>

namespace alice {

struct KvSlot;

// Precomputed RoPE tables, built once at model init and passed in as
// tensors — primitives stay pure functions of (inputs, weights, tables).
struct RopeTables {
  TensorView cos;  // [max_pos, head_dim/2], f32
  TensorView sin;  // [max_pos, head_dim/2], f32
  uint32_t head_dim = 0;
};

struct AttentionParams {
  uint32_t n_heads = 0;
  uint32_t n_kv_heads = 0;  // GQA: n_heads % n_kv_heads == 0
  uint32_t head_dim = 0;
  float scale = 0.0f;  // typically 1/sqrt(head_dim)
};

// Stateless compute primitives. No allocation, no caches, no globals:
// the result depends only on the arguments. Storage dtype comes from the
// views; accumulation is fp32. Shape-driven kernel selection (GEMV at
// M==1 vs BLAS GEMM at prefill) happens *inside* these functions and is
// invisible to callers.
namespace prim {

// out[M,N] = x[M,K] @ w[N,K]^T, with w in the usual [out_features, in_features]
// checkpoint layout.
void matmul(TensorView out, TensorView x, TensorView w);

// x += delta
void add(TensorView x, TensorView delta);

// Root-mean-square normalization with per-channel gain w.
void rmsnorm(TensorView out, TensorView x, TensorView w, float eps);

// out = silu(gate) * up   (SwiGLU half)
void silu_mul(TensorView out, TensorView gate, TensorView up);

// In-place RoPE on q [S, n_heads*head_dim] and k [S, n_kv_heads*head_dim],
// positions [start_pos, start_pos + S).
void rope(TensorView q, TensorView k, const RopeTables& tables, size_t start_pos);

// Causal attention for the S query rows in q against the KV cache,
// appending the current tokens' k/v first. out: [S, n_heads*head_dim].
void attention(TensorView out, TensorView q, KvSlot& kv, const AttentionParams& p);

}  // namespace prim
}  // namespace alice
