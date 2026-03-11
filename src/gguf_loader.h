#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshBuffer;
class MeshCommandQueue;
}

struct ModelBuffers;

// Parse a BFP8_B tiled GGUF file and upload small weights to device DRAM.
// Large matmul weights (pre-packed BFP8_B) are loaded into host vectors for
// subsequent upload by create_weight_tensors(). Returns true on success.
bool load_gguf_weights(
    const std::string& path,
    ModelBuffers& model,
    tt::tt_metal::distributed::MeshDevice* device,
    tt::tt_metal::distributed::MeshCommandQueue& cq);
