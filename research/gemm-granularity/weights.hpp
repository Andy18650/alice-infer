#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
namespace granularity {
// Fixed storage tiles. Neither dimension is a launch/split parameter.
constexpr int weight_tile_n=32;
constexpr int weight_tile_k=128;
enum class WeightFormat : int { Column=0, TileN=1, TileK=2 };
struct PackedInfo {
  int n_extent=0,k_extent=0,n_origin=0,k_origin=0;
  WeightFormat format=WeightFormat::Column;
};
__host__ __device__ constexpr size_t packed_elements(int n,int k) {
  return size_t((n+31)/32)*((k+127)/128)*32*128;
}
// Inverse GPU initializer uses a separate traversal; do not use this as the
// independent host packer's implementation.
__host__ __device__ constexpr size_t packed_offset(int n,int k,int full_n,int full_k,WeightFormat f) {
  const int nt=(full_n+31)/32,kt=(full_k+127)/128;
  const size_t tile=f==WeightFormat::TileN?size_t(n/32)*kt+k/128:size_t(k/128)*nt+n/32;
  return tile*4096+size_t(n%32)*128+k%128;
}
}
