#include "api.hpp"
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp>
#include <stdexcept>
namespace compute_bound {
namespace {
using D=ck::bhalf_t;
using Row=ck::tensor_layout::gemm::RowMajor;using Col=ck::tensor_layout::gemm::ColumnMajor;
using Pass=ck::tensor_operation::element_wise::PassThrough;
template<ck::index_t... Is>using S=ck::Sequence<Is...>;
using Op=ck::tensor_operation::device::DeviceGemm_Xdl_CShuffleV3<
    Row,Col,Row,D,D,float,float,float,Pass,Pass,Pass,
    ck::tensor_operation::device::GemmSpecialization::Default,
    256,TM,TN,64,8,8,XDL,XDL,4,4,
    S<8,32,1>,S<1,0,2>,S<1,0,2>,2,8,8,0,
    S<8,32,1>,S<1,0,2>,S<1,0,2>,2,8,8,0,
    1,2,S<1,32,1,8>,8,
    ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3,D,D>;
using Grid=Op::GridwiseGemm;
constexpr int min_occupancy=TM*TN/256<=128?2:1;
void launch(Args a,Storage storage,hipStream_t stream) {
    if(storage!=Storage::Affine)throw std::runtime_error("classic control requires affine inputs");
    auto args=Op::MakeArgument(reinterpret_cast<const D*>(a.a),reinterpret_cast<const D*>(a.b),a.c,
        a.gm*TM,a.gn*TN,a.k,a.k,a.k,a.gn*TN,1,Pass{},Pass{},Pass{});
    if(!Op::IsSupportedArgument(args) || !Grid::CalculateHasMainKBlockLoop(a.k))
        throw std::runtime_error("classic control shape unsupported");
    ck::kernel_gemm_xdl_cshuffle_v3<Grid,true,ck::InMemoryDataOperationEnum::Set,min_occupancy>
        <<<a.gm*a.gn,256,0,stream>>>(args);
}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
}
void REGISTER(std::vector<Candidate>& cs) {
    Candidate c{KERNEL_NAME,TM,TN,64,XDL,256,13,2,1,{}, {},launch,1,1,"grouped-m4"};
    const auto* fn=reinterpret_cast<const void*>(ck::kernel_gemm_xdl_cshuffle_v3<Grid,true,ck::InMemoryDataOperationEnum::Set,min_occupancy>);
    check(hipFuncGetAttributes(&c.attributes[0],fn));
    check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&c.capacity[0],fn,256,0));
    cs.push_back(c);
}
}
