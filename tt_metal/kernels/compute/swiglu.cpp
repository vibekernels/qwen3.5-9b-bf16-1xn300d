// SPDX-License-Identifier: Apache-2.0
// SwiGLU compute kernel: output = SiLU(gate) * up
// SiLU(x) = x * sigmoid(x)
//
// Input: gate tiles in c_0, up tiles in c_1
// Output: SiLU(gate) * up in c_16
//
// Uses SFPU silu operation on gate, then FPU multiply with up.

#include "api/compute/compute_api.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_gate = tt::CBIndex::c_0;
    constexpr uint32_t cb_up   = tt::CBIndex::c_1;
    constexpr uint32_t cb_out  = tt::CBIndex::c_16;
    constexpr uint32_t cb_silu = tt::CBIndex::c_2;  // intermediate: SiLU(gate)

    // First pass: compute SiLU(gate) into intermediate buffer
    unary_op_init_common(cb_gate, cb_silu);

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_gate, 1);

        copy_tile(cb_gate, 0, 0);
        silu_tile(0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_silu, 1);
        pack_tile(0, cb_silu);
        cb_push_back(cb_silu, 1);

        cb_pop_front(cb_gate, 1);
        tile_regs_release();
    }

    // Second pass: multiply SiLU(gate) * up
    binary_op_init_common(cb_silu, cb_up, cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_regs_acquire();
        cb_wait_front(cb_silu, 1);
        cb_wait_front(cb_up, 1);

        mul_tiles(cb_silu, cb_up, 0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_silu, 1);
        cb_pop_front(cb_up, 1);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
