#pragma once

// Block interface (design §8.1). The boundary is the device-resident fp32
// stream: `io` is a device pointer to [S, d] row-major fp32 (d fixed per
// model, known to each block from construction), owned by the Trunk.
// In-place: the block reads its input from io and writes its output into
// the same io (the Trunk owns the residual and performs the add).

#include <cstdint>

namespace alice::block {

class Block {
 public:
  virtual ~Block() = default;
  virtual void forward(float* io, uint32_t S) = 0;
};

// A Block that owns non-weight state (attention KV cache). Append-only;
// position is a private counter.
class StatefulBlock : public Block {
 public:
  virtual void reset_state() = 0;
  // Graph replay bypasses forward(): validate/advance the private host state
  // once before a decode replay. Device state progression is recorded in the
  // graph. No position is exposed through the block boundary.
  virtual void advance_decode_state() = 0;
};

}  // namespace alice::block
