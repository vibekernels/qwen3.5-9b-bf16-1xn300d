// SPDX-License-Identifier: Apache-2.0
// Compute kernel for GEMV: y[row] = sum_k W[row,k] * x[k]
// Each core computes a subset of output rows.
// Uses matmul_tiles for tile-level multiply-accumulate.
//
// For GEMV (M=1), we're computing [1, Kt] @ [Kt, num_rows]^T
// But since matmul_tiles operates on 32x32 tiles, we treat this as:
// - For each output row tile: accumulate matmul across K tiles
// - The output is a single tile (32 rows of the output vector, each 32 wide but only 1 element used)
//
// Actually, for single-token decode, the input x is padded to a 32-row tile.
// So we compute: out_tile = sum_k W_tile[row, k] * x_tile[0, k]
// where the matmul gives us a 32x32 output tile per (row, k) pair, accumulated over k.

#include "api/compute/compute_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_rows = get_compile_time_arg_val(0);  // Number of output row tiles for this core
    uint32_t Kt       = get_compile_time_arg_val(1);  // K dimension in tiles

    constexpr uint32_t cb_w   = tt::CBIndex::c_0;   // Weight tiles
    constexpr uint32_t cb_x   = tt::CBIndex::c_1;   // Input tiles
    constexpr uint32_t cb_out = tt::CBIndex::c_16;   // Output tiles

    mm_init(cb_w, cb_x, cb_out);

    for (uint32_t row = 0; row < num_rows; row++) {
        // Acquire destination registers for accumulation
        tile_regs_acquire();

        for (uint32_t kt = 0; kt < Kt; kt++) {
            // Wait for weight and input tiles from reader
            cb_wait_front(cb_w, 1);
            cb_wait_front(cb_x, 1);

            // Multiply-accumulate: dst += W_tile * x_tile
            matmul_tiles(cb_w, cb_x, 0, 0, 0, false);

            // Release input tiles
            cb_pop_front(cb_w, 1);
            cb_pop_front(cb_x, 1);
        }

        // Done accumulating over K for this row tile
        tile_regs_commit();
        tile_regs_wait();

        // Pack result and push to output circular buffer
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        tile_regs_release();
    }
}
}  // namespace NAMESPACE
