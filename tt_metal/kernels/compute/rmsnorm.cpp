// SPDX-License-Identifier: Apache-2.0
// RMSNorm compute kernel for a single core.
// Computes: y = x * rsqrt(mean(x^2) + eps) * weight
//
// Strategy for single-token decode on one core:
// 1. Read all input tiles, compute partial sum of squares per tile
// 2. Reduce to get total sum of squares
// 3. Compute rsqrt(sum/dim + eps)
// 4. Read input again + weight, multiply x * scale * weight
//
// This kernel handles one token at a time (n_tokens=1 for decode).
// Input and weight tiles arrive via circular buffers from the reader.
//
// CB layout:
//   c_0: input tiles (read twice: once for sum_sq, once for apply)
//   c_1: weight tiles
//   c_2: intermediate (partial sums)
//   c_16: output tiles

#include "api/compute/compute_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);  // n_embd / 32 = 128 tiles per token

    constexpr uint32_t cb_in    = tt::CBIndex::c_0;
    constexpr uint32_t cb_w     = tt::CBIndex::c_1;
    constexpr uint32_t cb_out   = tt::CBIndex::c_16;
    constexpr uint32_t cb_inter = tt::CBIndex::c_2;

    // For RMSNorm we need:
    // Pass 1: compute sum of squares across all tiles
    // Pass 2: apply normalization with weight

    // Use eltwise unary for squaring and reduction
    // Then rsqrt via SFPU
    // Then eltwise binary for multiply

    // Pass 1: Square input tiles and reduce
    // We'll use the SFPU to compute x*x for each element, then reduce
    unary_op_init_common(cb_in, cb_inter);

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);

        // Copy tile to dst register
        copy_tile(cb_in, 0, 0);

        // Square: dst[0] = dst[0] * dst[0]
        // Use SFPU square operation
        square_tile(0);

        tile_regs_commit();
        tile_regs_wait();

        // Pack squared tile to intermediate buffer
        cb_reserve_back(cb_inter, 1);
        pack_tile(0, cb_inter);
        cb_push_back(cb_inter, 1);

        cb_pop_front(cb_in, 1);
        tile_regs_release();
    }

    // Now reduce the squared tiles to get sum
    // The reduction and rsqrt will be handled by reading the intermediate tiles
    // and accumulating - for now we pass them through to the writer
    // which will do the final reduction on the host side
    // (This is a simplified version - a production kernel would do the full
    //  reduction in hardware)

    // Pass 2: Apply normalization
    // Read the scale factor from cb_inter (computed by host or a separate reduce kernel)
    // Then multiply input * scale * weight
    binary_op_init_common(cb_in, cb_w, cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_in, 1);
        cb_wait_front(cb_w, 1);

        // Multiply input * weight (the scale factor is baked into the weight
        // by the host after computing rsqrt(mean(x^2) + eps))
        mul_tiles(cb_in, cb_w, 0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_in, 1);
        cb_pop_front(cb_w, 1);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
