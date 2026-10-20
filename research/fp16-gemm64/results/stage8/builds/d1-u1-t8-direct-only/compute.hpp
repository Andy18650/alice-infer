constexpr int kDepth=1, kUnroll=1, kStageK=8, kPhases=1;
template<bool Hot> __device__ __forceinline__ void compute(uint32_t alo,uint32_t blo,int steps,int records,F16& c) {
  uint32_t remaining=steps,offset=0; const uint32_t window=records*2048u;
  if constexpr(Hot==true) {
    asm volatile(
      "v_add_u32 v4, %[alo], %[offset]\n\t"
      "v_add_u32 v5, %[blo], %[offset]\n\t"
      ".Lgemm64_hot_0_%=:\n\t"
      "ds_read_b64 v[0:1], v4\n\t"
      "ds_read_b64 v[2:3], v5\n\t"
      "s_waitcnt lgkmcnt(0)\n\t"
      "v_mfma_f32_32x32x8f16 %[c], v[0:1], v[2:3], %[c]\n\t"
      "s_sub_u32 %[remaining], %[remaining], 1\n\t"
      "s_cmp_eq_u32 %[remaining], 0\n\t"
      "s_cbranch_scc0 .Lgemm64_hot_0_%=\n\t"
      ".Lgemm64_hot_done_%=:\n\t"
      "s_waitcnt lgkmcnt(0)\n\t"
      : [c] "+a"(c),[remaining] "+&s"(remaining),[offset] "+&s"(offset)
      : [alo] "v"(alo),[blo] "v"(blo),[window] "s"(window)
      : "memory","scc","v0","v1","v2","v3","v4","v5");
  }
  if constexpr(Hot==false) {
    asm volatile(
      ".Lgemm64_sweep_0_%=:\n\t"
      "v_add_u32 v4, %[alo], %[offset]\n\t"
      "v_add_u32 v5, %[blo], %[offset]\n\t"
      "ds_read_b64 v[0:1], v4\n\t"
      "ds_read_b64 v[2:3], v5\n\t"
      "s_add_u32 %[offset], %[offset], 2048\n\t"
      "s_cmp_ge_u32 %[offset], %[window]\n\t"
      "s_cselect_b32 %[offset], 0, %[offset]\n\t"
      "s_waitcnt lgkmcnt(0)\n\t"
      "v_mfma_f32_32x32x8f16 %[c], v[0:1], v[2:3], %[c]\n\t"
      "s_sub_u32 %[remaining], %[remaining], 1\n\t"
      "s_cmp_eq_u32 %[remaining], 0\n\t"
      "s_cbranch_scc0 .Lgemm64_sweep_0_%=\n\t"
      ".Lgemm64_sweep_done_%=:\n\t"
      "s_waitcnt lgkmcnt(0)\n\t"
      : [c] "+a"(c),[remaining] "+&s"(remaining),[offset] "+&s"(offset)
      : [alo] "v"(alo),[blo] "v"(blo),[window] "s"(window)
      : "memory","scc","v0","v1","v2","v3","v4","v5");
  }
}
__device__ __forceinline__ void copy_direct(uint32_t lo,uint32_t hi,uint32_t dst) {
  uint32_t saved;
  asm volatile(
    "s_mov_b32 %[saved], m0\n\t"
    "s_mov_b32 m0, %[dst]\n\t"
    "s_nop 0\n\t"
    "v_mov_b32 v6, %[lo]\n\t"
    "v_mov_b32 v7, %[hi]\n\t"
    "global_load_dword v[6:7], off lds\n\t"
    "global_load_dword v[6:7], off offset:256 lds\n\t"
    "s_mov_b32 m0, %[saved]\n\t"
    : [saved] "=&s"(saved) : [lo] "v"(lo),[hi] "v"(hi),[dst] "s"(dst)
    : "v6","v7","vcc","scc","memory");
}
