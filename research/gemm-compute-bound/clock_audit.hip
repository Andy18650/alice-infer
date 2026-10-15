// Dedicated diagnostic build of the working kernel. Main benchmark binaries
// remain separate; resource/runtime perturbation is explicitly checked.
#define COMPUTE_BOUND_NO_MAIN 1
#define CLOCK_AUDIT 1
#include "kernel.hip"
#include "bench.hip"
__global__ void timed_reference(float* out,ClockStamp* stamps,int loops,bool random_data=false) {
  const auto cb=clock64(),sb=wall_clock64();
  U4 a{},b{};
  #pragma unroll
  for(int j=0;j<4;++j){a[j]=random_data?value(size_t(threadIdx.x%64)*4+j,123):0x3b80;b[j]=random_data?value(size_t(threadIdx.x%64)*4+j,456):0x3b80;}
  V16 c0{},c1{},c2{},c3{};
  #pragma unroll
  for(int j=0;j<16;++j){c1[j]=1;c2[j]=2;c3[j]=3;}
  #pragma clang loop unroll(disable)
  for(int i=0;i<loops;++i){
    c0=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a,b,c0,0,0,0);
    c1=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a,b,c1,0,0,0);
    c2=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a,b,c2,0,0,0);
    c3=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a,b,c3,0,0,0);
  }
  auto c=c0+c1+c2+c3;float result=0;
  #pragma unroll
  for(int j=0;j<16;++j)result+=c[j];out[size_t(blockIdx.x)*blockDim.x+threadIdx.x]=result;
  const auto se=wall_clock64(),ce=clock64();
  if(threadIdx.x%64==0)stamps[size_t(blockIdx.x)*4+threadIdx.x/64]={uint64_t(cb),uint64_t(sb),uint64_t(ce),uint64_t(se)};
}
int main(int argc,char** argv)try {
  const std::string out=argc>1?argv[1]:".";HIP(hipSetDevice(0));Runtime r;
  int steady_khz=0;HIP(hipDeviceGetAttribute(&steady_khz,hipDeviceAttributeWallClockRate,0));
  std::fprintf(stderr,"steady_clock_khz=%d; timestamps are diagnostic, not a frequency override\n",steady_khz);
  Buffer<ClockStamp> stamps{1664*4};Buffer<float> values{416*256};
  auto* pointer=stamps.get();HIP(hipMemcpyToSymbol(HIP_SYMBOL(compute_bound::clock_stamps),&pointer,sizeof(pointer)));
  std::ofstream csv(out+"/wave-clocks.csv");csv<<"phase,round,cta,wave,core_cycles,steady_ticks,core_per_steady,effective_MHz\n";
  auto record=[&](const char* phase,int round,int blocks){
    std::vector<ClockStamp> host(size_t(blocks)*4);HIP(hipMemcpy(host.data(),stamps.get(),host.size()*sizeof(ClockStamp),hipMemcpyDeviceToHost));
    for(size_t i=0;i<host.size();++i){const auto& s=host[i];const auto core=s.core_end-s.core_begin,steady=s.steady_end-s.steady_begin;double ratio=double(core)/steady;csv<<phase<<','<<round<<','<<i/4<<','<<i%4<<','<<core<<','<<steady<<','<<ratio<<','<<ratio*steady_khz/1000<<'\n';}
  };
  {std::ofstream phase(out+"/status.txt");phase<<"reference,NA,0,0\n";}
  const auto begin=std::chrono::steady_clock::now();do{for(int i=0;i<8;++i)timed_reference<<<208,256,0,r.s>>>(values.get(),stamps.get(),4096);HIP(hipStreamSynchronize(r.s));}while(std::chrono::steady_clock::now()-begin<std::chrono::seconds(2));
  for(int round=0;round<9;++round){auto t=time(r,[&]{timed_reference<<<104,256,0,r.s>>>(values.get(),stamps.get(),4096);},16);record("reference",round,104);double flop=104.0*4*4096*4*16384;std::fprintf(stderr,"TIMED_REFERENCE round=%d tflops=%.6f gpu_us=%.6f\n",round,flop/t.gpu/1e6,t.gpu);}
  {std::ofstream phase(out+"/status.txt");phase<<"random-reference,NA,0,0\n";}
  for(int i=0;i<8;++i)timed_reference<<<104,256,0,r.s>>>(values.get(),stamps.get(),4096,true);
  HIP(hipStreamSynchronize(r.s));
  for(int round=0;round<9;++round){auto t=time(r,[&]{timed_reference<<<104,256,0,r.s>>>(values.get(),stamps.get(),4096,true);},16);record("random-reference",round,104);const double f=104.0*4*4096*4*16384;std::fprintf(stderr,"RANDOM_REFERENCE round=%d tflops=%.6f gpu_us=%.6f\n",round,f/t.gpu/1e6,t.gpu);}
  std::vector<Candidate> cs;add_profile(cs);const auto& x=cs.front();
  const int gm=4096/x.tm,gn=6656/x.tn,k=32768; // same full problem across tile/pipeline probes
  Matrix m(x,gm,gn,k,Storage::ABPanel,r,false);Args args{m.a.get(),m.b.get(),m.c.get(),gm,gn,k,0};
  std::fprintf(stderr,"TIMED_GEMM resources regs=%d LDS=%zu private=%zu capacity=%d\n",x.attributes[2].numRegs,x.attributes[2].sharedSizeBytes,x.attributes[2].localSizeBytes,x.capacity[2]);
  m.poison(r);for(int i=0;i<3;++i)x.launch(args,Storage::ABPanel,r.s);m.check(r);
  {std::ofstream phase(out+"/status.txt");phase<<"gemm,NA,0,0\n";}
  for(int round=0;round<9;++round){auto t=time(r,[&]{x.launch(args,Storage::ABPanel,r.s);},16);m.check(r);record("gemm",round,gm*gn);std::fprintf(stderr,"TIMED_GEMM round=%d tflops=%.6f gpu_us=%.6f\n",round,2.0*m.m*m.n*k/t.gpu/1e6,t.gpu);}
  {std::ofstream phase(out+"/status.txt");phase<<"hot-reference,NA,0,0\n";}
  for(int round=0;round<9;++round){
    // A short arithmetic probe follows a full GEMM immediately on the same
    // stream, testing the ceiling in the GEMM-established clock state.
    x.launch(args,Storage::ABPanel,r.s);
    auto t=time(r,[&]{timed_reference<<<104,256,0,r.s>>>(values.get(),stamps.get(),512);},1);
    record("hot-reference",round,104);
    const double flop=104.0*4*512*4*16384;
    std::fprintf(stderr,"HOT_REFERENCE round=%d tflops=%.6f gpu_us=%.6f\n",round,flop/t.gpu/1e6,t.gpu);
  }
  return 0;
}catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
