// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3.ipp"

namespace ck_tile {

using kernel_traits = fmha_fwd_v3_kernel_traits<
                fmha_fwd_v3_args::data_type_enum::bf16, false, true
            >;

template <>
float fmha_fwd_v3_kernel_dispatch<kernel_traits>(
    const fmha_fwd_v3_args& args, const stream_config& config)
{
    return fmha_fwd_v3_kernel_launch<kernel_traits::kernel>(args, config);
}

}