#include "platform.hpp"

#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <hip/hiprtc.h>

namespace alice::gpu_rocblas {

namespace {

#define HIP_CHECK(x)                                                                     \
  do {                                                                                   \
    const hipError_t e_ = (x);                                                           \
    if (e_ != hipSuccess)                                                                \
      throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(e_) +      \
                               " at " __FILE__ ":" + std::to_string(__LINE__));          \
  } while (0)

void check(hiprtcResult r, hiprtcProgram prog, const char* what) {
  if (r == HIPRTC_SUCCESS) return;
  size_t n = 0;
  hiprtcGetProgramLogSize(prog, &n);
  std::string log(n, '\0');
  if (n > 1) hiprtcGetProgramLog(prog, log.data());
  throw std::runtime_error(std::string("hipRTC ") + what + " failed: " +
                           hiprtcGetErrorString(r) + "\n" + log);
}

}  // namespace

bool available_impl() {
  static const bool ok = [] {
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
  }();
  return ok;
}

Platform& platform() {
  static Platform p = [] {
    if (!available_impl()) throw std::runtime_error("gpu_rocblas: no ROCm GPU present");
    Platform q;
    HIP_CHECK(hipSetDevice(q.device));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, q.device));
    q.arch = prop.gcnArchName;  // "gfx90a:sramecc+:xnack-" -> "gfx90a"
    q.arch = q.arch.substr(0, q.arch.find(':'));
    if (rocblas_create_handle(&q.handle) != rocblas_status_success)
      throw std::runtime_error("gpu_rocblas: rocblas_create_handle failed");
    return q;
  }();
  return p;
}

hipFunction_t rtc_kernel(const char* name, const char* src) {
  static std::unordered_map<std::string, hipFunction_t> cache;
  if (auto it = cache.find(name); it != cache.end()) return it->second;

  Platform& p = platform();
  hiprtcProgram prog;
  check(hiprtcCreateProgram(&prog, src, name, 0, nullptr, nullptr), prog, "create");
  const std::string arch_opt = "--offload-arch=" + p.arch;
  const char* opts[] = {arch_opt.c_str()};
  const hiprtcResult cr = hiprtcCompileProgram(prog, 1, opts);
  check(cr, prog, "compile");

  size_t n = 0;
  check(hiprtcGetCodeSize(prog, &n), prog, "code size");
  std::vector<char> code(n);
  check(hiprtcGetCode(prog, code.data()), prog, "get code");
  hiprtcDestroyProgram(&prog);

  hipModule_t mod;
  hipFunction_t fn;
  HIP_CHECK(hipSetDevice(p.device));
  HIP_CHECK(hipModuleLoadData(&mod, code.data()));
  HIP_CHECK(hipModuleGetFunction(&fn, mod, name));
  cache.emplace(name, fn);
  return fn;
}

}  // namespace alice::gpu_rocblas
