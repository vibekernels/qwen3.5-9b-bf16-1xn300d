// SPDX-License-Identifier: Apache-2.0
// Compact GEMV writer: writes only row 0 of each output tile (face0 + face1 = 64 bytes per tile).
// Eliminates 32x DMA waste from writing full 2KB tiles when only row 0 is valid (decode GEMV).
//
// Tile layout: face0 at L1 offset 0 (16 BF16 = 32 bytes), face1 at L1 offset 512 bytes (16 BF16).
// Output buffer: Mt_padded tiles × 64 bytes each (vs 2048 bytes for full tiles).
//
// Compile-time args: [cb_out, acc_config]
// Runtime args: [dst_addr, Mt_per_core, out_start_tile]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    uint32_t dst_addr       = get_arg_val<uint32_t>(0);
    uint32_t Mt_per_core    = get_arg_val<uint32_t>(1);
    uint32_t out_start_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    // Compact output: 32 BF16 per tile (face0: 16 elements, face1: 16 elements of row 0)
    constexpr uint32_t compact_tile_bytes = 32 * sizeof(uint16_t);  // 64 bytes per tile

    constexpr auto acc_args = TensorAccessorArgs<1>();
    const auto acc = TensorAccessor(acc_args, dst_addr, compact_tile_bytes);

    for (uint32_t mt = 0; mt < Mt_per_core; mt++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_addr = get_read_ptr(cb_out);

        // NOC address for this compact tile (64 bytes at tile index out_start_tile + mt)
        uint64_t dst_noc = acc.get_noc_addr(out_start_tile + mt);

        // Face 0: first 16 BF16 of row 0, at L1 offset 0
        noc_async_write(l1_addr, dst_noc, 16 * sizeof(uint16_t));

        // Face 1: next 16 BF16 of row 0, at L1 offset 256 * sizeof(uint16_t) = 512 bytes
        noc_async_write(l1_addr + 256 * sizeof(uint16_t),
                        dst_noc + 16 * sizeof(uint16_t),
                        16 * sizeof(uint16_t));

        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
