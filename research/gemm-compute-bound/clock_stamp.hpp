#pragma once
#include <cstdint>
namespace compute_bound {
struct ClockStamp {uint64_t core_begin,steady_begin,core_end,steady_end;};
}
