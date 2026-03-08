// SPDX-License-Identifier: Apache-2.0
// Reader for single-core FPU-based RMSNorm.
// Reads hidden tiles, norm weight tiles, and generates scaler/epsilon tiles.
//
// Compile-time args: [Kt, acc_hidden_config, acc_norm_w_config]
// Runtime args: [hidden_addr, norm_w_addr, n_elements]

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}

void kernel_main() {
    uint32_t hidden_addr  = get_arg_val<uint32_t>(0);
    uint32_t norm_w_addr  = get_arg_val<uint32_t>(1);
    uint32_t n_elements   = get_arg_val<uint32_t>(2);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);

    constexpr uint32_t cb_hidden  = tt::CBIndex::c_0;
    constexpr uint32_t cb_norm_w  = tt::CBIndex::c_2;
    constexpr uint32_t cb_scaler  = tt::CBIndex::c_5;
    constexpr uint32_t cb_eps     = tt::CBIndex::c_6;

    uint32_t tile_size = get_tile_size(cb_hidden);

    constexpr auto acc_hidden_args = TensorAccessorArgs<1>();
    const auto acc_hidden = TensorAccessor(acc_hidden_args, hidden_addr, tile_size);
    constexpr auto acc_norm_w_args = TensorAccessorArgs<acc_hidden_args.next_compile_time_args_offset()>();
    const auto acc_norm_w = TensorAccessor(acc_norm_w_args, norm_w_addr, tile_size);

    // Generate scaler tile (1/N)
    {
        float inv_n = 1.0f / (float)n_elements;
        uint16_t inv_n_bf16 = f32_to_bf16_bits(inv_n);
        uint32_t packed = ((uint32_t)inv_n_bf16 << 16) | inv_n_bf16;

        cb_reserve_back(cb_scaler, 1);
        uint32_t addr = get_write_ptr(cb_scaler);
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);

        for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++) p[i] = 0;
        for (uint32_t i = 0; i < 8; i++) p[i] = packed;
        for (uint32_t i = 0; i < 8; i++) p[128 + i] = packed;
        for (uint32_t i = 0; i < 8; i++) p[256 + i] = packed;
        for (uint32_t i = 0; i < 8; i++) p[384 + i] = packed;
        cb_push_back(cb_scaler, 1);
    }

    // Generate epsilon tile
    {
        float eps = 1e-6f;
        uint16_t eps_bf16 = f32_to_bf16_bits(eps);

        cb_reserve_back(cb_eps, 1);
        uint32_t addr = get_write_ptr(cb_eps);
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);

        for (uint32_t i = 0; i < tile_size / sizeof(uint32_t); i++) p[i] = 0;
        volatile tt_l1_ptr uint16_t* u16 =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr);
        u16[0] = eps_bf16;
        cb_push_back(cb_eps, 1);
    }

    // Read all hidden tiles
    cb_reserve_back(cb_hidden, Kt);
    uint32_t base = get_write_ptr(cb_hidden);
    for (uint32_t kt = 0; kt < Kt; kt++)
        noc_async_read_tile(kt, acc_hidden, base + kt * tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_hidden, Kt);

    // Read all norm weight tiles
    cb_reserve_back(cb_norm_w, Kt);
    base = get_write_ptr(cb_norm_w);
    for (uint32_t kt = 0; kt < Kt; kt++)
        noc_async_read_tile(kt, acc_norm_w, base + kt * tile_size);
    noc_async_read_barrier();
    cb_push_back(cb_norm_w, Kt);
}
