#include "api.hpp"
#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <stdexcept>
#include <string>

#ifndef TILE_M
#error "build.sh supplies the closed candidate set"
#endif
namespace granularity {
namespace {
using D=ck_tile::bf16_t;
using Row=ck_tile::tensor_layout::gemm::RowMajor;
using Col=ck_tile::tensor_layout::gemm::ColumnMajor;
using Shape=ck_tile::TileGemmShape<ck_tile::sequence<TILE_M,TILE_N,TILE_K>,
    ck_tile::sequence<WARP_M,WARP_N,1>,ck_tile::sequence<WARP_TILE,WARP_TILE,16>>;
using Partitioner=ck_tile::GemmTile1DPartitioner<Shape>;
// Physical K stride is padded to 16 with zeros in both implementations.
// CK masks the remaining tile tails. Keep aligned 8-BF16 vector loads even
// with kPadK: the passed K is always a multiple of 16.
using Traits=ck_tile::TileGemmTraits<true,true,true,Row,Col,Row>;
using Problem=ck_tile::GemmPipelineProblem<D,D,float,Shape,Traits,D,true,8,8>;
using Pipeline=ck_tile::GemmPipelineAGmemBGmemCRegV1<Problem>;
using Epilogue=ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
    D,D,ck_tile::tuple<>,float,float,ck_tile::tuple<>,Row,
    ck_tile::element_wise::PassThrough,Problem::kBlockSize,
    Partitioner::MPerBlock,Partitioner::NPerBlock,WARP_M,WARP_N,
    WARP_TILE,WARP_TILE,16,Problem::TransposeC,ck_tile::memory_operation_enum::set>>;
using Gemm=ck_tile::GemmKernel<Partitioner,Pipeline,Epilogue>;
using KArgs=typename Gemm::UniversalGemmKernel::KernelArgs;
__global__ __launch_bounds__(Problem::kBlockSize,1) void entry(KArgs args) {
  Gemm{}(args);
}
__global__ __launch_bounds__(Problem::kBlockSize,1) void split_entry(KArgs args,size_t a_step,size_t b_step,size_t c_step) {
  using U=Gemm::UniversalGemmKernel;
  const auto slice=__builtin_amdgcn_readfirstlane(blockIdx.z);
  const auto [im,in]=Partitioner{args.M,args.N}.GetOutputTileIndex(__builtin_amdgcn_readfirstlane(blockIdx.x));
  const std::array<const D*,1> a={static_cast<const D*>(args.as_ptr[0])+slice*a_step};
  const std::array<const D*,1> b={static_cast<const D*>(args.bs_ptr[0])+slice*b_step};
  auto* c=static_cast<float*>(args.e_ptr)+slice*c_step;
  // Each z slice is an ordinary GEMM writing a private partial. Explicit id=0
  // avoids CK's default blockIdx.z-based split offset being applied twice.
  const U::SplitKBatchOffset offset(args,0);
  __shared__ char lds[U::GetSmemSize()];
  U::template RunGemm<true>(a,b,args.ds_ptr,c,lds,args,offset,im*TILE_M,in*TILE_N);
}
__device__ auto packed_view(const D* p,PackedInfo info,int n,int k) {
  using namespace ck_tile;
  const int nt=(info.n_extent+31)/32,kt=(info.k_extent+127)/128;
  const int ns=nt*32,ks=kt*128;
  const int n_stride=info.format==WeightFormat::TileN?kt*4096:4096;
  const int k_stride=info.format==WeightFormat::TileN?4096:nt*4096;
  const auto physical=make_naive_tensor_descriptor(
      make_tuple(nt,kt,number<32>{},number<128>{}),
      make_tuple(n_stride,k_stride,number<128>{},number<1>{}),number<8>{},number<1>{});
  const auto matrix=transform_tensor_descriptor(physical,
      make_tuple(make_merge_transform(make_tuple(nt,number<32>{})),
                 make_merge_transform(make_tuple(kt,number<128>{}))),
      make_tuple(sequence<0,2>{},sequence<1,3>{}),make_tuple(sequence<0>{},sequence<1>{}));
  const auto sliced=transform_tensor_descriptor(matrix,
      make_tuple(make_slice_transform(ns,info.n_origin,info.n_origin+n),
                 make_slice_transform(ks,info.k_origin,info.k_origin+k)),
      make_tuple(sequence<0>{},sequence<1>{}),make_tuple(sequence<0>{},sequence<1>{}));
  // Slice transforms discard the known vector alignment. Our supported
  // contract guarantees K origins and lengths are multiples of 8, and the
  // innermost 128-wide storage row is contiguous. Reassert only that safe
  // eight-BF16 group, not contiguity across a storage-tile boundary.
  using Desc=remove_cvref_t<decltype(sliced)>;
  constexpr int kd=Desc::get_top_dimension_hidden_ids().at(number<1>{});
  using V=typename Desc::GuaranteedVectorLengths;
  using S=typename Desc::GuaranteedVectorStrides;
  using Aligned=tensor_descriptor<remove_cvref_t<decltype(sliced.get_transforms())>,
      remove_cvref_t<decltype(Desc::get_lower_dimension_hidden_idss())>,
      remove_cvref_t<decltype(Desc::get_upper_dimension_hidden_idss())>,
      remove_cvref_t<decltype(Desc::get_top_dimension_hidden_ids())>,typename Desc::ElementSpaceSizeType,
      decltype(V::modify(number<kd>{},number<8>{})),decltype(S::modify(number<kd>{},number<1>{}))>;
  const Aligned aligned{sliced,sliced.get_element_space_size()};
  return make_tensor_view<address_space_enum::global>(p,aligned);
}
__global__ __launch_bounds__(Problem::kBlockSize,1) void packed_entry(KArgs args,PackedInfo info,size_t a_step,size_t c_step) {
  using U=Gemm::UniversalGemmKernel;
  const auto slice=__builtin_amdgcn_readfirstlane(blockIdx.z);
  info.k_origin+=slice*a_step;
  const auto [im,in]=Partitioner{args.M,args.N}.GetOutputTileIndex(__builtin_amdgcn_readfirstlane(blockIdx.x));
  const std::array<const D*,1> a={static_cast<const D*>(args.as_ptr[0])+slice*a_step};
  const std::array<const D*,1> b={static_cast<const D*>(args.bs_ptr[0])};
  auto* c=static_cast<float*>(args.e_ptr)+slice*c_step;
  const U::SplitKBatchOffset offset(args,0);
  const auto ordinary=U::template MakeGemmTensorViews<ck_tile::memory_operation_enum::set>(a,b,args.ds_ptr,c,args,offset);
  const auto views=ck_tile::make_tuple(ordinary.at(ck_tile::number<0>{}),
      ck_tile::make_tuple(packed_view(b[0],info,args.N,args.K)),
      ordinary.at(ck_tile::number<2>{}),ordinary.at(ck_tile::number<3>{}));
  const auto padded=U::MakeGemmPadViews(views);
  auto windows=U::MakeGemmTileWindows(padded,im*TILE_M,in*TILE_N);
  __shared__ char lds[U::GetSmemSize()];
  const auto accum=Pipeline{}(windows.at(ck_tile::number<0>{})[ck_tile::number<0>{}],
      windows.at(ck_tile::number<1>{})[ck_tile::number<0>{}],Partitioner::GetLoopNum(args.K),lds);
  auto& out=windows.at(ck_tile::number<3>{});
  const auto& ds=windows.at(ck_tile::number<2>{});
  Epilogue{}(out,accum,ds,lds);
}
KArgs convert(const Args& a) {
  return Gemm::MakeKernelArgs({a.a,a.b,a.c,1,a.m,a.n,a.k,a.lda,a.ldb,a.ldc});
}
bool supported(const Args& a) {
  if(a.packed.format!=WeightFormat::Column && a.packed.k_origin%8) return false;
  return Gemm::IsSupportedArgument(convert(a));
}
void launch(const Args& a,hipStream_t stream) {
  const auto g=Gemm::GridSize(a.m,a.n,1);
  entry<<<g,Gemm::BlockSize(),0,stream>>>(convert(a));
}
void launch_split(const Args& a,int q,size_t a_step,size_t b_step,size_t c_step,hipStream_t stream) {
  auto g=Gemm::GridSize(a.m,a.n,1);g.z=q;
  split_entry<<<g,Gemm::BlockSize(),0,stream>>>(convert(a),a_step,b_step,c_step);
}
void launch_packed(const Args& a,int q,size_t a_step,size_t c_step,hipStream_t stream) {
  auto g=Gemm::GridSize(a.m,a.n,1);g.z=q;
  packed_entry<<<g,Gemm::BlockSize(),0,stream>>>(convert(a),a.packed,a_step,c_step);
}
void check(hipError_t e) { if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
}
void REGISTER(std::vector<Kernel>& kernels) {
  Kernel k{KERNEL_NAME,TILE_M,TILE_N,TILE_K,Problem::kBlockSize,{},0,supported,launch,{},0,launch_split,{},0,launch_packed};
  check(hipFuncGetAttributes(&k.attributes,reinterpret_cast<const void*>(entry)));
  check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.blocks_per_cu,
      reinterpret_cast<const void*>(entry),k.threads,0));
  check(hipFuncGetAttributes(&k.split_attributes,reinterpret_cast<const void*>(split_entry)));
  check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.split_blocks_per_cu,
      reinterpret_cast<const void*>(split_entry),k.threads,0));
  check(hipFuncGetAttributes(&k.packed_attributes,reinterpret_cast<const void*>(packed_entry)));
  check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&k.packed_blocks_per_cu,
      reinterpret_cast<const void*>(packed_entry),k.threads,0));
  kernels.push_back(k);
}
}
