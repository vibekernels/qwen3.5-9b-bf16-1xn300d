// SPDX-License-Identifier: Apache-2.0
// Reader kernel for GEMV: y = W @ x
// Reads weight tiles from DRAM and the input activation vector (broadcast to all cores).
// Each core computes a subset of output rows.
//
// W is stored in tiled layout in DRAM: [N_tiles, K_tiles] tiles
// x is a single row of tiles: [1, K_tiles] tiles (same for all cores)
//
// This reader fetches tiles for the rows assigned to this core.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    uint32_t w_addr       = get_arg_val<uint32_t>(0);  // Weight buffer address
    uint32_t x_addr       = get_arg_val<uint32_t>(1);  // Input vector buffer address
    uint32_t start_row    = get_arg_val<uint32_t>(2);  // First output row (in tiles) for this core
    uint32_t num_rows     = get_arg_val<uint32_t>(3);  // Number of output rows for this core
    uint32_t Kt           = get_arg_val<uint32_t>(4);  // K dimension in tiles

    constexpr uint32_t cb_w = tt::CBIndex::c_0;   // Weight tiles
    constexpr uint32_t cb_x = tt::CBIndex::c_1;   // Input tiles

    // TensorAccessor for weight and input buffers
    constexpr auto w_args = TensorAccessorArgs<0>();
    const auto w_accessor = TensorAccessor(w_args, w_addr, get_tile_size(cb_w));
    constexpr auto x_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    const auto x_accessor = TensorAccessor(x_args, x_addr, get_tile_size(cb_x));

    // For each output row assigned to this core
    for (uint32_t row = 0; row < num_rows; row++) {
        uint32_t w_row = start_row + row;

        // Stream through K dimension
        for (uint32_t kt = 0; kt < Kt; kt++) {
            // Read weight tile W[w_row, kt]
            uint32_t w_tile_idx = w_row * Kt + kt;
            cb_reserve_back(cb_w, 1);
            uint32_t w_l1_addr = get_write_ptr(cb_w);
            noc_async_read_tile(w_tile_idx, w_accessor, w_l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_w, 1);

            // Read input tile x[0, kt]
            cb_reserve_back(cb_x, 1);
            uint32_t x_l1_addr = get_write_ptr(cb_x);
            noc_async_read_tile(kt, x_accessor, x_l1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_x, 1);
        }
    }
}
