// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"

namespace ck_tile {

template <>
float fmha_fwd_v3_dispatch<type_tag<fmha_fwd_v3_args::data_type_enum::bf16, true>>(
    const fmha_fwd_v3_args& args, const stream_config& config)
{
    return launch<get_kernel_t<FmhaFwdBf16, true, true>>(args, config);
}

}