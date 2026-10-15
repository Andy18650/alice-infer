#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <vector>
namespace compute_bound {
enum class Storage : int { Affine, BPanel, ABPanel };
struct Args {const uint16_t* a;const uint16_t* b;float* c;int gm,gn,k,order;};
struct Candidate {
  const char* name;
  int tm,tn,tk,warp_tile,threads,pipeline,stages,lds_buffers;
  hipFuncAttributes attributes[3];int capacity[3];
  void (*launch)(Args,Storage,hipStream_t);
  int storage_mask=7,order_mask=3;
  const char* order_name=nullptr;
};
void add_v1_128(std::vector<Candidate>&);
void add_v3_128(std::vector<Candidate>&);
void add_v4_128(std::vector<Candidate>&);
void add_v3_256x128(std::vector<Candidate>&);
void add_v3_256(std::vector<Candidate>&);
void add_v4_256(std::vector<Candidate>&);
void add_v3_128_w16(std::vector<Candidate>&);
void add_v3_256_w16(std::vector<Candidate>&);
void add_mem_128(std::vector<Candidate>&);
void add_classic_128(std::vector<Candidate>&);
void add_classic_256(std::vector<Candidate>&);
void add_v3_128_wk8(std::vector<Candidate>&);
void add_v3_256_wk8(std::vector<Candidate>&);
void add_v3_256x128_wk8(std::vector<Candidate>&);
void add_direct4(std::vector<Candidate>&);
void add_direct8(std::vector<Candidate>&);
}
