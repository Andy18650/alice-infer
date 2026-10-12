#pragma once
#include <hip/hip_runtime.h>
#include "weights.hpp"
#include <cstdint>
#include <vector>

namespace granularity {
// A: row-major [M,K] BF16 activations; B: column-major [K,N] BF16 weights;
// C: row-major [M,N] FP32. Strides are elements, never bytes.
struct Args {
  const uint16_t* a;
  const uint16_t* b;
  float* c;
  int m,n,k,lda,ldb,ldc;
  PackedInfo packed{};
};
struct Kernel {
  const char* name;
  int tm,tn,tk,threads;
  hipFuncAttributes attributes;
  int blocks_per_cu;
  bool (*supported)(const Args&);
  void (*launch)(const Args&,hipStream_t);
  hipFuncAttributes split_attributes;
  int split_blocks_per_cu;
  // One producer launch; z selects disjoint K slices and private FP32 outputs.
  void (*launch_split)(const Args&,int,size_t,size_t,size_t,hipStream_t);
  hipFuncAttributes packed_attributes;
  int packed_blocks_per_cu;
  void (*launch_packed)(const Args&,int,size_t,size_t,hipStream_t);
};
void register_simt(std::vector<Kernel>&);
void register_16x16(std::vector<Kernel>&);
void register_16x16_k128(std::vector<Kernel>&);
void register_16x32_k128(std::vector<Kernel>&);
void register_16x32_k256(std::vector<Kernel>&);
void register_16x32(std::vector<Kernel>&);
void register_16x64(std::vector<Kernel>&);
void register_32x32(std::vector<Kernel>&);
void register_64x64(std::vector<Kernel>&);
void register_64x128(std::vector<Kernel>&);
void register_128x128(std::vector<Kernel>&);
void register_128x256(std::vector<Kernel>&);
void register_256x256(std::vector<Kernel>&);
}
