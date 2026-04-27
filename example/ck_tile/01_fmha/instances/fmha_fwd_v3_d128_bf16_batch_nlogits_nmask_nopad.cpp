// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd_v3.hpp"
#include "fmha_fwd_v3_impl.hpp"

#if CK_FMHA_V3_ENABLE_NOPAD
using kernel_traits_nopad = ck_tile::fmha_fwd_v3_kernel_traits_nopad<FmhaFwdBf16, false, false, false>;

INST_FMHA_FWD_V3_DISPATCH(kernel_traits_nopad)
#endif
