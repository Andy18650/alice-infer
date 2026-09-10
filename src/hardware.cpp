// Hardware availability probes (always compiled; the ROCm-backed check lives
// in blocks/gpu_rocblas/platform.cpp when built with ALICE_HAVE_ROCBLAS).
#include "alice/block.hpp"

#if defined(ALICE_HAVE_ROCBLAS)
namespace alice::gpu_rocblas {
bool available_impl();
}
#endif

namespace alice {

bool gpu_available() {
#if defined(ALICE_HAVE_ROCBLAS)
  return gpu_rocblas::available_impl();
#else
  return false;
#endif
}

}  // namespace alice
