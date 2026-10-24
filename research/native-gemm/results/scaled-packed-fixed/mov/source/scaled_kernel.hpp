#pragma once

// All values crossing the opaque256-VGPR asm boundary are explicitly scalar.
// There is no C++ epilogue keeping work-item IDs or output tuples live across it.
__global__ __launch_bounds__(256) void native_gemm(
    const uint32_t* a, const uint32_t* b, float* out, uint64_t* stamps,
    int m, int n, int mp, int np, int tiles, int swizzle, const float* scales) {
    extern __shared__ __attribute__((aligned(16))) uint32_t smem[];
    const int grid_columns = np / 128;
    const int tile_row = blockIdx.x / grid_columns;
    const int tile_column = blockIdx.x % grid_columns;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int row_anchor = tile_row * 128 + (wave / 2) * 64;
    const int col_anchor = tile_column * 128 + (wave % 2) * 64;
    const uintptr_t a_base = reinterpret_cast<uintptr_t>(a) + size_t(tile_row) * 8192 + wave * 2048;
    const uintptr_t b_base = reinterpret_cast<uintptr_t>(b) + size_t(tile_column) * 4096 + wave * 1024;
    const uintptr_t scale_base = reinterpret_cast<uintptr_t>(scales) + tile_column * 4;
    const uintptr_t out_base = reinterpret_cast<uintptr_t>(out)
        + (uint64_t(row_anchor) * n + col_anchor) * 4;
    const uintptr_t stamp_base = reinterpret_cast<uintptr_t>(stamps)
        + (uint64_t(blockIdx.x) * 4 + wave) * 32;
    const uint32_t lds = __builtin_amdgcn_readfirstlane(uint32_t(reinterpret_cast<uintptr_t>(smem)));
    // Linear LDS only in this explicit variant. Caller rejects swizzle!=0.
    (void)swizzle;
    const uint32_t rows_left = row_anchor < m ? uint32_t(m - row_anchor) : 0;
    const uint32_t cols_left = col_anchor < n ? uint32_t(n - col_anchor) : 0;
    scaled_pipeline(
        __builtin_amdgcn_readfirstlane(uint32_t(a_base)),
        __builtin_amdgcn_readfirstlane(uint32_t(a_base >> 32)),
        __builtin_amdgcn_readfirstlane(uint32_t(b_base)),
        __builtin_amdgcn_readfirstlane(uint32_t(b_base >> 32)),
        __builtin_amdgcn_readfirstlane(uint32_t(mp * 64)),
        __builtin_amdgcn_readfirstlane(uint32_t(np * 32)),
        lds + (wave / 2) * 1024, lds + 8192 + (wave % 2) * 1024, lds + wave * 2048,
        __builtin_amdgcn_readfirstlane(uint32_t(tiles)),
        __builtin_amdgcn_readfirstlane(uint32_t(scale_base)),
        __builtin_amdgcn_readfirstlane(uint32_t(scale_base >> 32)),
        __builtin_amdgcn_readfirstlane(uint32_t(grid_columns * 4)),
        out_base, stamp_base, __builtin_amdgcn_readfirstlane(uint32_t(n * 4)),
        __builtin_amdgcn_readfirstlane(rows_left), __builtin_amdgcn_readfirstlane(cols_left));
}
