// SPDX-License-Identifier: Apache-2.0
// Writer kernel for GEMV: writes output tiles to DRAM.
// Each core writes its subset of output rows.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr     = get_arg_val<uint32_t>(0);  // Output buffer address
    uint32_t start_row    = get_arg_val<uint32_t>(1);  // First output row (in tiles)
    uint32_t num_rows     = get_arg_val<uint32_t>(2);  // Number of output rows

    constexpr uint32_t cb_out = tt::CBIndex::c_16;  // Output tiles

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst_accessor = TensorAccessor(dst_args, dst_addr, get_tile_size(cb_out));

    for (uint32_t row = 0; row < num_rows; row++) {
        uint32_t out_tile_idx = start_row + row;
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        noc_async_write_tile(out_tile_idx, dst_accessor, l1_read_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
