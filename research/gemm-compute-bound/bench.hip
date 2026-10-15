#include "api.hpp"
#include <rocblas/rocblas.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace compute_bound;
#define HIP(x) do {auto status=(x);if(status!=hipSuccess)throw std::runtime_error(std::string(#x)+": "+hipGetErrorString(status));} while(0)
#define BLAS(x) do {auto status=(x);if(status!=rocblas_status_success)throw std::runtime_error(std::string(#x)+": rocBLAS="+std::to_string(status));} while(0)
__host__ __device__ uint32_t mix(uint32_t v) {v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16);}
__host__ __device__ uint16_t value(size_t i,uint32_t seed) {uint32_t v=mix(uint32_t(i)^mix(uint32_t(i>>32))^seed);return 0x3e80u|(v&127u)|((v>>16)&0x8000u);}
float f32(uint16_t v) {return std::bit_cast<float>(uint32_t(v)<<16);}
__global__ void initialize(uint16_t* p,size_t size,int rows,int k,int tile_k,bool panel,uint32_t seed) {
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size;i+=size_t(gridDim.x)*blockDim.x) {
    size_t logical=i;
    if(panel) {const size_t kp=i/(size_t(rows)*tile_k),local=i%(size_t(rows)*tile_k);logical=(local/tile_k)*k+kp*tile_k+local%tile_k;}
    p[i]=value(logical,seed);
  }
}
__global__ void gather(const float* p,const size_t* positions,float* out,size_t count) {for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x)out[i]=p[positions[i]];}
using U4=uint16_t __attribute__((ext_vector_type(4)));
using V4=float __attribute__((ext_vector_type(4)));
using V16=float __attribute__((ext_vector_type(16)));
template<int Matrix>
__global__ void reference(float* out,int loops) {
  const U4 a={0x3b80,0x3b80,0x3b80,0x3b80};
  if constexpr(Matrix==16) {
    V4 c0={0,0,0,0},c1={1,1,1,1},c2={2,2,2,2},c3={3,3,3,3};
    #pragma clang loop unroll(disable)
    for(int i=0;i<loops;++i) {
      c0=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,a,c0,0,0,0);c1=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,a,c1,0,0,0);
      c2=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,a,c2,0,0,0);c3=__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a,a,c3,0,0,0);
    }
    const auto c=c0+c1+c2+c3;out[size_t(blockIdx.x)*blockDim.x+threadIdx.x]=c[0]+c[1]+c[2]+c[3];
  } else {
    V16 c0{},c1{},c2{},c3{};
    #pragma unroll
    for(int j=0;j<16;++j) {c1[j]=1;c2[j]=2;c3[j]=3;}
    #pragma clang loop unroll(disable)
    for(int i=0;i<loops;++i) {
      c0=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a,a,c0,0,0,0);c1=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a,a,c1,0,0,0);
      c2=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a,a,c2,0,0,0);c3=__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a,a,c3,0,0,0);
    }
    const auto c=c0+c1+c2+c3;float s=0;
    #pragma unroll
    for(int j=0;j<16;++j)s+=c[j];out[size_t(blockIdx.x)*blockDim.x+threadIdx.x]=s;
  }
}
struct Free {template<class T>void operator()(T* p)const {if(p)(void)hipFree(p);}};
template<class T>struct Buffer {std::unique_ptr<T,Free> p;size_t count;explicit Buffer(size_t n):count(n){T* raw=nullptr;HIP(hipMalloc(&raw,n*sizeof(T)));p.reset(raw);}T* get()const{return p.get();}};
struct Runtime {
 hipStream_t s{};hipEvent_t start{},stop{};rocblas_handle blas{};Buffer<char> workspace{64ull<<20};
 Runtime(){HIP(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));HIP(hipEventCreate(&start));HIP(hipEventCreate(&stop));BLAS(rocblas_create_handle(&blas));BLAS(rocblas_set_stream(blas,s));BLAS(rocblas_set_atomics_mode(blas,rocblas_atomics_not_allowed));BLAS(rocblas_set_workspace(blas,workspace.get(),workspace.count));}
 ~Runtime(){(void)rocblas_destroy_handle(blas);(void)hipEventDestroy(start);(void)hipEventDestroy(stop);(void)hipStreamDestroy(s);}
};
struct Timing {double gpu,cpu;};
template<class F>Timing time(Runtime& r,F fn,int repetitions) {
 auto t0=std::chrono::steady_clock::now();HIP(hipEventRecord(r.start,r.s));for(int i=0;i<repetitions;++i)fn();HIP(hipEventRecord(r.stop,r.s));HIP(hipEventSynchronize(r.stop));auto t1=std::chrono::steady_clock::now();float ms=0;HIP(hipEventElapsedTime(&ms,r.start,r.stop));return {ms*1000.0/repetitions,std::chrono::duration<double,std::micro>(t1-t0).count()/repetitions};
}
struct Options {std::string suite="quick",impl="all";int rounds=3,repeats=8,gm=0,gn=0,k=0;};
Options parse(int argc,char** argv){Options o;for(int i=1;i<argc;++i){std::string key=argv[i];if(i+1==argc)throw std::runtime_error("missing argument");std::string v=argv[++i];if(key=="--suite")o.suite=v;else if(key=="--impl")o.impl=v;else if(key=="--rounds")o.rounds=std::stoi(v);else if(key=="--repeats")o.repeats=std::stoi(v);else if(key=="--gm")o.gm=std::stoi(v);else if(key=="--gn")o.gn=std::stoi(v);else if(key=="--k")o.k=std::stoi(v);else throw std::runtime_error("unknown argument "+key);}if(o.rounds<1||o.repeats<1||o.repeats>128)throw std::runtime_error("invalid repetitions");return o;}
struct Matrix {
 int m,n,k;Buffer<uint16_t>a,b;Buffer<float>c,samples;Buffer<size_t>indices;std::vector<size_t>positions;std::vector<double>gold;std::vector<uint32_t>previous;
 Matrix(const Candidate& x,int gm,int gn,int K,Storage storage,Runtime& r,bool full):m(x.tm*gm),n(x.tn*gn),k(K),a(size_t(m)*k),b(size_t(n)*k),c(size_t(m)*n+256),samples(full?size_t(m)*n:128),indices(full?size_t(m)*n:128){
  initialize<<<104,256,0,r.s>>>(a.get(),a.count,m,k,x.tk,storage==Storage::ABPanel,123);
  initialize<<<104,256,0,r.s>>>(b.get(),b.count,n,k,x.tk,storage!=Storage::Affine,456);
  size_t count=full?size_t(m)*n:128;
  for(size_t j=0;j<count;++j){size_t idx=full?j:j==0?0:j==1?size_t(m)*n-1:size_t(mix(uint32_t(j)))*size_t(m)*n/(uint64_t(1)<<32);positions.push_back(idx);int row=idx/n,col=idx%n;double v=0;for(int t=0;t<k;++t)v+=double(f32(value(size_t(row)*k+t,123)))*double(f32(value(size_t(col)*k+t,456)));gold.push_back(v);}
  HIP(hipMemcpyAsync(indices.get(),positions.data(),indices.count*sizeof(size_t),hipMemcpyHostToDevice,r.s));HIP(hipStreamSynchronize(r.s));
 }
 void poison(Runtime&r){HIP(hipMemsetAsync(c.get(),0xff,c.count*4,r.s));HIP(hipStreamSynchronize(r.s));}
 double check(Runtime&r){HIP(hipStreamSynchronize(r.s));std::array<float,256>guard{};HIP(hipMemcpy(guard.data(),c.get()+size_t(m)*n,guard.size()*4,hipMemcpyDeviceToHost));for(float v:guard)if(std::bit_cast<uint32_t>(v)!=0xffffffffu)throw std::runtime_error("C guard overwritten");gather<<<std::min<size_t>((samples.count+255)/256,104),256,0,r.s>>>(c.get(),indices.get(),samples.get(),samples.count);HIP(hipStreamSynchronize(r.s));std::vector<float>values(samples.count);HIP(hipMemcpy(values.data(),samples.get(),values.size()*4,hipMemcpyDeviceToHost));std::vector<uint32_t>current;double max_error=0;for(size_t j=0;j<values.size();++j){const double error=std::abs(double(values[j])-gold[j]);if(!std::isfinite(values[j])||error>2e-4+3e-5*std::abs(gold[j]))throw std::runtime_error("FP64 oracle mismatch at "+std::to_string(positions[j]));max_error=std::max(max_error,error);current.push_back(std::bit_cast<uint32_t>(values[j]));}if(!previous.empty()&&previous!=current)throw std::runtime_error("within-implementation nondeterminism");previous=std::move(current);return max_error;}
};
void arithmetic_reference(Runtime&r,int rounds){Buffer<float>output{416*256};auto begin=std::chrono::steady_clock::now();do{for(int i=0;i<8;++i)reference<32><<<208,256,0,r.s>>>(output.get(),4096);HIP(hipStreamSynchronize(r.s));}while(std::chrono::steady_clock::now()-begin<std::chrono::seconds(2));for(int matrix:{16,32})for(int blocks:{104,208,416})for(int round=0;round<rounds;++round){auto fn=[&]{if(matrix==16)reference<16><<<blocks,256,0,r.s>>>(output.get(),4096);else reference<32><<<blocks,256,0,r.s>>>(output.get(),4096);};auto t=time(r,fn,16);std::vector<float>v(size_t(blocks)*256);HIP(hipMemcpy(v.data(),output.get(),v.size()*4,hipMemcpyDeviceToHost));for(float f:v)if(f!=(matrix==16?40.0f:128.0f))throw std::runtime_error("MFMA reference mismatch");double flops=double(blocks)*4*4096*4*(matrix==16?8192:16384);std::fprintf(stderr,"MFMA_REFERENCE matrix=%d blocks=%d round=%d tflops=%.6f gpu_us=%.6f\n",matrix,blocks,round,flops/t.gpu/1e6,t.gpu);}}
#ifndef COMPUTE_BOUND_NO_MAIN
int main(int argc,char**argv)try{
 auto o=parse(argc,argv);HIP(hipSetDevice(0));hipDeviceProp_t p{};HIP(hipGetDeviceProperties(&p,0));if(std::string(p.gcnArchName).find("gfx90a")==std::string::npos||p.warpSize!=64)throw std::runtime_error("requires gfx90a wave64");std::fprintf(stderr,"device=0 arch=%s CU=%d model=none context=none BF16xBF16->FP32\n",p.gcnArchName,p.multiProcessorCount);
 std::vector<Candidate>cs;add_v1_128(cs);add_v3_128(cs);add_v4_128(cs);add_v3_256x128(cs);add_v3_256(cs);add_v4_256(cs);add_v3_128_w16(cs);add_v3_256_w16(cs);add_mem_128(cs);add_classic_128(cs);add_classic_256(cs);add_v3_128_wk8(cs);add_v3_256_wk8(cs);add_v3_256x128_wk8(cs);add_direct4(cs);add_direct8(cs);Runtime r;
 if(o.suite!="test")arithmetic_reference(r,o.rounds);
 std::vector<std::array<int,3>>shapes;
 if(o.gm&&o.gn&&o.k)shapes.push_back({o.gm,o.gn,o.k});
 else if(o.suite=="test")shapes={{1,1,512},{2,3,1024}};
 else if(o.suite=="quick")for(auto g:std::vector<std::array<int,2>>{{10,10},{8,13},{16,13},{16,26}})for(int k:{4096,8192})shapes.push_back({g[0],g[1],k});
 else if(o.suite=="sweep")for(auto g:std::vector<std::array<int,2>>{{10,10},{8,13},{13,8},{16,13},{13,16},{16,26},{26,16}})for(int k:{4096,8192,16384})shapes.push_back({g[0],g[1],k});
 else throw std::runtime_error("unknown suite");
 std::puts("impl,pipeline,stages,lds_buffers,storage,order,m,n,k,tm,tn,tk,warp_tile,grid_m,grid_n,ctas,threads,registers,lds_bytes,private_bytes,blocks_per_cu,round,mode,repeats,gpu_us,cpu_us,tflops,max_abs_error");
 std::mt19937 rng(3711);std::shuffle(shapes.begin(),shapes.end(),rng);std::shuffle(cs.begin(),cs.end(),rng);
 for(const auto&x:cs){if(o.impl!="all"&&o.impl!=x.name)continue;for(auto shape:shapes){int gm=shape[0],gn=shape[1],k=shape[2];if(gm<1||gn<1||k%x.tk||k/x.tk<4||(k/x.tk)%2)throw std::runtime_error("requires aligned even hot-loop K and positive grid");for(auto storage:{Storage::Affine,Storage::BPanel,Storage::ABPanel}){
  if(!(x.storage_mask&(1<<int(storage))))continue;
  Matrix mat(x,gm,gn,k,storage,r,o.suite=="test"&&gm==1&&gn==1);Args args{mat.a.get(),mat.b.get(),mat.c.get(),gm,gn,k,0};
  for(int order:{0,1}){if(!(x.order_mask&(1<<order)))continue;args.order=order;mat.previous.clear();mat.poison(r);for(int warm=0;warm<3;++warm)x.launch(args,storage,r.s);HIP(hipGetLastError());double error=mat.check(r);
   hipGraph_t graph{};hipGraphExec_t exec{};HIP(hipStreamBeginCapture(r.s,hipStreamCaptureModeGlobal));x.launch(args,storage,r.s);HIP(hipStreamEndCapture(r.s,&graph));HIP(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));mat.poison(r);HIP(hipGraphLaunch(exec,r.s));mat.check(r);
   for(int round=0;round<o.rounds;++round)for(int reps:(o.repeats==1?std::vector<int>{1}:std::vector<int>{1,o.repeats})){for(int mode=0;mode<2;++mode){auto t=time(r,[&]{if(mode)HIP(hipGraphLaunch(exec,r.s));else x.launch(args,storage,r.s);},reps);const auto& at=x.attributes[int(storage)];double flops=2.0*mat.m*mat.n*k;std::printf("%s,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%d,%d,%s,%d,%.6f,%.6f,%.6f,%.9g\n",x.name,x.pipeline,x.stages,x.lds_buffers,storage==Storage::Affine?"affine":storage==Storage::BPanel?"b-panel":"ab-panel",x.order_name?x.order_name:order?"m-fast":"n-fast",mat.m,mat.n,k,x.tm,x.tn,x.tk,x.warp_tile,gm,gn,gm*gn,x.threads,at.numRegs,at.sharedSizeBytes,at.localSizeBytes,x.capacity[int(storage)],round,mode?"graph":"eager",reps,t.gpu,t.cpu,flops/t.gpu/1e6,error);std::fflush(stdout);}}
   mat.check(r);HIP(hipGraphExecDestroy(exec));HIP(hipGraphDestroy(graph));
  }
 }}}
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
#endif
