// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/stream_config.hpp"

namespace ck_tile {

struct fmha_fwd_v3_args
{
    enum class data_type_enum
    {
        fp16,
        bf16
    };

    data_type_enum data_type;
    // bool is_varlen;

    ck_tile::index_t batch;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;

    const void* q_ptr;
    ck_tile::index_t stride_q;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t batch_stride_q;

    const void* k_ptr;
    ck_tile::index_t stride_k;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t batch_stride_k;

    const void* v_ptr;
    ck_tile::index_t stride_v;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t batch_stride_v;

    void* o_ptr;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_o;
};

float fmha_fwd_v3(const ck_tile::fmha_fwd_v3_args& args, const ck_tile::stream_config& config);

} // namespace ck_tile
