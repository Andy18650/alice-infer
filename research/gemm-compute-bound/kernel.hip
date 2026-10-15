#include "api.hpp"
#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <stdexcept>
#ifdef CLOCK_AUDIT
#include "clock_stamp.hpp"
#endif
namespace compute_bound {
namespace {
#ifdef CLOCK_AUDIT
__device__ ClockStamp* clock_stamps;
#endif
using namespace ck_tile;
using D=bf16_t;
using Row=tensor_layout::gemm::RowMajor;
using Col=tensor_layout::gemm::ColumnMajor;
using Shape=TileGemmShape<sequence<TM,TN,TK>,sequence<WM,WN,1>,sequence<WT,WT,WK>>;
#if PIPE==1
using Traits=TileGemmTraits<false,false,false,Row,Col,Row>;
using Problem=GemmPipelineProblem<D,D,float,Shape,Traits,D,true,8,8>;
using Pipeline=GemmPipelineAGmemBGmemCRegV1<Problem>;
constexpr int stages=1,buffers=1;
#else
using Traits=TileGemmUniversalTraits<false,false,false,PIPE==4,Row,Col,Row>;
using Problem=UniversalGemmPipelineProblem<D,D,float,Shape,Traits,
    GemmPipelineScheduler::Intrawave,true,PIPE==4?TailNumber::Two:TailNumber::Full,D,true,8,8>;
#if PIPE==2
using Pipeline=GemmPipelineAgBgCrMem<Problem>;
#elif PIPE==3
using Pipeline=GemmPipelineAgBgCrCompV3<Problem>;
#else
using Pipeline=GemmPipelineAgBgCrCompV4<Problem>;
#endif
constexpr int stages=Pipeline::PrefetchStages,buffers=PIPE==4?2:1;
#endif
using Partitioner=GemmTile1DPartitioner<Shape>;
using Epilogue=CShuffleEpilogue<CShuffleEpilogueProblem<D,D,tuple<>,float,float,tuple<>,Row,
    element_wise::PassThrough,Problem::kBlockSize,TM,TN,WM,WN,WT,WT,WK,
    Problem::TransposeC,memory_operation_enum::set>>;
// A panel uses [K_panel][M_tile][M_inner][K_inner]; B uses the corresponding
// N tile. Each compute operand tile is physically contiguous, with no slice
// transform or global repack in the kernel. Both vector contracts are exact.
template<int Rows>
__device__ auto panel_view(const D* p,int count,int k) {
  const int kt=k/TK;
  const auto physical=make_naive_tensor_descriptor(
      make_tuple(kt,count,number<Rows>{},number<TK>{}),
      make_tuple(count*Rows*TK,number<Rows*TK>{},number<TK>{},number<1>{}),number<8>{},number<1>{});
  const auto matrix=transform_tensor_descriptor(physical,
      make_tuple(make_merge_transform(make_tuple(count,number<Rows>{})),
                 make_merge_transform(make_tuple(kt,number<TK>{}))),
      make_tuple(sequence<1,2>{},sequence<0,3>{}),make_tuple(sequence<0>{},sequence<1>{}));
  using Desc=remove_cvref_t<decltype(matrix)>;
  constexpr int kid=Desc::get_top_dimension_hidden_ids().at(number<1>{});
  using V=typename Desc::GuaranteedVectorLengths;using S=typename Desc::GuaranteedVectorStrides;
  using Aligned=tensor_descriptor<remove_cvref_t<decltype(matrix.get_transforms())>,
      remove_cvref_t<decltype(Desc::get_lower_dimension_hidden_idss())>,
      remove_cvref_t<decltype(Desc::get_upper_dimension_hidden_idss())>,
      remove_cvref_t<decltype(Desc::get_top_dimension_hidden_ids())>,typename Desc::ElementSpaceSizeType,
      decltype(V::modify(number<kid>{},number<8>{})),decltype(S::modify(number<kid>{},number<1>{}))>;
  return make_tensor_view<address_space_enum::global>(p,Aligned{matrix,matrix.get_element_space_size()});
}
template<Storage Format>
__global__ __launch_bounds__(Problem::kBlockSize,1) void entry(Args a) {
#ifdef CLOCK_AUDIT
  const auto core_begin=clock64(),steady_begin=wall_clock64();
#endif
  int mi,ni;
  if(a.order==0) {mi=blockIdx.x/a.gn;ni=blockIdx.x%a.gn;}
  else {mi=blockIdx.x%a.gm;ni=blockIdx.x/a.gm;}
  const auto av=[&] {
    if constexpr(Format==Storage::ABPanel) return panel_view<TM>(reinterpret_cast<const D*>(a.a),a.gm,a.k);
    else return make_naive_tensor_view<address_space_enum::global>(reinterpret_cast<const D*>(a.a),
        make_tuple(a.gm*TM,a.k),make_tuple(a.k,1),number<8>{},number<1>{});
  }();
  const auto bv=[&] {
    if constexpr(Format!=Storage::Affine) return panel_view<TN>(reinterpret_cast<const D*>(a.b),a.gn,a.k);
    else return make_naive_tensor_view<address_space_enum::global>(reinterpret_cast<const D*>(a.b),
        make_tuple(a.gn*TN,a.k),make_tuple(a.k,1),number<8>{},number<1>{});
  }();
  auto aw=make_tile_window(av,make_tuple(number<TM>{},number<TK>{}),{mi*TM,0});
  auto bw=make_tile_window(bv,make_tuple(number<TN>{},number<TK>{}),{ni*TN,0});
  __shared__ char lds0[Pipeline::GetSmemSize()>Epilogue::GetSmemSize()?Pipeline::GetSmemSize():Epilogue::GetSmemSize()];
#if PIPE==4
  __shared__ char lds1[Pipeline::GetSmemSize()];
  const auto accum=Pipeline{}(aw,bw,a.k/TK,lds0,lds1);
#else
  const auto accum=Pipeline{}(aw,bw,a.k/TK,lds0);
#endif
  const auto cv=make_naive_tensor_view<address_space_enum::global>(a.c,
      make_tuple(a.gm*TM,a.gn*TN),make_tuple(a.gn*TN,1),number<Epilogue::GetVectorSizeC()>{},number<1>{});
  auto cw=make_tile_window(cv,make_tuple(number<TM>{},number<TN>{}),{mi*TM,ni*TN});
  const tuple<> ds;
  Epilogue{}(cw,accum,ds,lds0);
#ifdef CLOCK_AUDIT
  const auto steady_end=wall_clock64(),core_end=clock64();
  if((threadIdx.x%64)==0) clock_stamps[size_t(blockIdx.x)*(Problem::kBlockSize/64)+threadIdx.x/64]=
      {uint64_t(core_begin),uint64_t(steady_begin),uint64_t(core_end),uint64_t(steady_end)};
#endif
}
void launch(Args a,Storage s,hipStream_t stream) {
  if(s==Storage::Affine) entry<Storage::Affine><<<a.gm*a.gn,Problem::kBlockSize,0,stream>>>(a);
  else if(s==Storage::BPanel) entry<Storage::BPanel><<<a.gm*a.gn,Problem::kBlockSize,0,stream>>>(a);
  else entry<Storage::ABPanel><<<a.gm*a.gn,Problem::kBlockSize,0,stream>>>(a);
}
void check(hipError_t e) {if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
}
void REGISTER(std::vector<Candidate>& candidates) {
  Candidate c{KERNEL_NAME,TM,TN,TK,WT,Problem::kBlockSize,PIPE,stages,buffers,{}, {},launch};
  const void* functions[]={reinterpret_cast<const void*>(entry<Storage::Affine>),
      reinterpret_cast<const void*>(entry<Storage::BPanel>),reinterpret_cast<const void*>(entry<Storage::ABPanel>)};
  for(int i=0;i<3;++i) {
    check(hipFuncGetAttributes(&c.attributes[i],functions[i]));
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&c.capacity[i],functions[i],c.threads,0));
  }
  candidates.push_back(c);
}
}
