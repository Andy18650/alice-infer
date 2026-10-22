constexpr int kRing=32;
__device__ __forceinline__ void load_vgpr(uint32_t lo,uint32_t hi,uint32_t mask,uint32_t groups,U4 (&ring)[kRing],
    uint64_t& begin,uint64_t& issued,uint64_t& completed,
    uint64_t& wall_begin,uint64_t& wall_issued,uint64_t& wall_completed) {
  uint32_t remaining=groups,offset=0;
  asm volatile(
    "s_memtime %[begin]\n\t"
    "s_memrealtime %[wall_begin]\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    ".Lvgpr_ring_%=:\n\t"
    "v_add_co_u32 v8, vcc, %[offset], %[lo]\n\t"
    "v_addc_co_u32 v9, vcc, 0, %[hi], vcc\n\t"
    "global_load_dwordx4 %[r0], v[8:9], off\n\t"
    "global_load_dwordx4 %[r1], v[8:9], off offset:1024\n\t"
    "global_load_dwordx4 %[r2], v[8:9], off offset:2048\n\t"
    "global_load_dwordx4 %[r3], v[8:9], off offset:3072\n\t"
    "v_add_co_u32 v8, vcc, 4096, v8\n\t"
    "v_addc_co_u32 v9, vcc, 0, v9, vcc\n\t"
    "global_load_dwordx4 %[r4], v[8:9], off\n\t"
    "global_load_dwordx4 %[r5], v[8:9], off offset:1024\n\t"
    "global_load_dwordx4 %[r6], v[8:9], off offset:2048\n\t"
    "global_load_dwordx4 %[r7], v[8:9], off offset:3072\n\t"
    "v_add_co_u32 v8, vcc, 4096, v8\n\t"
    "v_addc_co_u32 v9, vcc, 0, v9, vcc\n\t"
    "global_load_dwordx4 %[r8], v[8:9], off\n\t"
    "global_load_dwordx4 %[r9], v[8:9], off offset:1024\n\t"
    "global_load_dwordx4 %[r10], v[8:9], off offset:2048\n\t"
    "global_load_dwordx4 %[r11], v[8:9], off offset:3072\n\t"
    "v_add_co_u32 v8, vcc, 4096, v8\n\t"
    "v_addc_co_u32 v9, vcc, 0, v9, vcc\n\t"
    "global_load_dwordx4 %[r12], v[8:9], off\n\t"
    "global_load_dwordx4 %[r13], v[8:9], off offset:1024\n\t"
    "global_load_dwordx4 %[r14], v[8:9], off offset:2048\n\t"
    "global_load_dwordx4 %[r15], v[8:9], off offset:3072\n\t"
    "v_add_co_u32 v8, vcc, 4096, v8\n\t"
    "v_addc_co_u32 v9, vcc, 0, v9, vcc\n\t"
    "global_load_dwordx4 %[r16], v[8:9], off\n\t"
    "global_load_dwordx4 %[r17], v[8:9], off offset:1024\n\t"
    "global_load_dwordx4 %[r18], v[8:9], off offset:2048\n\t"
    "global_load_dwordx4 %[r19], v[8:9], off offset:3072\n\t"
    "v_add_co_u32 v8, vcc, 4096, v8\n\t"
    "v_addc_co_u32 v9, vcc, 0, v9, vcc\n\t"
    "global_load_dwordx4 %[r20], v[8:9], off\n\t"
    "global_load_dwordx4 %[r21], v[8:9], off offset:1024\n\t"
    "global_load_dwordx4 %[r22], v[8:9], off offset:2048\n\t"
    "global_load_dwordx4 %[r23], v[8:9], off offset:3072\n\t"
    "v_add_co_u32 v8, vcc, 4096, v8\n\t"
    "v_addc_co_u32 v9, vcc, 0, v9, vcc\n\t"
    "global_load_dwordx4 %[r24], v[8:9], off\n\t"
    "global_load_dwordx4 %[r25], v[8:9], off offset:1024\n\t"
    "global_load_dwordx4 %[r26], v[8:9], off offset:2048\n\t"
    "global_load_dwordx4 %[r27], v[8:9], off offset:3072\n\t"
    "v_add_co_u32 v8, vcc, 4096, v8\n\t"
    "v_addc_co_u32 v9, vcc, 0, v9, vcc\n\t"
    "global_load_dwordx4 %[r28], v[8:9], off\n\t"
    "global_load_dwordx4 %[r29], v[8:9], off offset:1024\n\t"
    "global_load_dwordx4 %[r30], v[8:9], off offset:2048\n\t"
    "global_load_dwordx4 %[r31], v[8:9], off offset:3072\n\t"
    "s_add_u32 %[offset], %[offset], 32768\n\t"
    "s_and_b32 %[offset], %[offset], %[mask]\n\t"
    "s_sub_u32 %[remaining], %[remaining], 1\n\t"
    "s_cmp_lg_u32 %[remaining], 0\n\t"
    "s_cbranch_scc1 .Lvgpr_ring_%=\n\t"
    "s_memtime %[issued]\n\t"
    "s_memrealtime %[wall_issued]\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    "s_waitcnt vmcnt(0)\n\t"
    "s_memtime %[completed]\n\t"
    "s_memrealtime %[wall_completed]\n\t"
    "s_waitcnt lgkmcnt(0)\n\t"
    : [r0] "=&v"(ring[0]),
      [r1] "=&v"(ring[1]),
      [r2] "=&v"(ring[2]),
      [r3] "=&v"(ring[3]),
      [r4] "=&v"(ring[4]),
      [r5] "=&v"(ring[5]),
      [r6] "=&v"(ring[6]),
      [r7] "=&v"(ring[7]),
      [r8] "=&v"(ring[8]),
      [r9] "=&v"(ring[9]),
      [r10] "=&v"(ring[10]),
      [r11] "=&v"(ring[11]),
      [r12] "=&v"(ring[12]),
      [r13] "=&v"(ring[13]),
      [r14] "=&v"(ring[14]),
      [r15] "=&v"(ring[15]),
      [r16] "=&v"(ring[16]),
      [r17] "=&v"(ring[17]),
      [r18] "=&v"(ring[18]),
      [r19] "=&v"(ring[19]),
      [r20] "=&v"(ring[20]),
      [r21] "=&v"(ring[21]),
      [r22] "=&v"(ring[22]),
      [r23] "=&v"(ring[23]),
      [r24] "=&v"(ring[24]),
      [r25] "=&v"(ring[25]),
      [r26] "=&v"(ring[26]),
      [r27] "=&v"(ring[27]),
      [r28] "=&v"(ring[28]),
      [r29] "=&v"(ring[29]),
      [r30] "=&v"(ring[30]),
      [r31] "=&v"(ring[31]),
      [begin] "=&s"(begin),[issued] "=&s"(issued),[completed] "=&s"(completed),
      [wall_begin] "=&s"(wall_begin),[wall_issued] "=&s"(wall_issued),[wall_completed] "=&s"(wall_completed),
      [remaining] "+&s"(remaining),[offset] "+&s"(offset)
    : [lo] "v"(lo),[hi] "v"(hi),[mask] "s"(mask)
    : "v8","v9","vcc","scc","memory");
}
