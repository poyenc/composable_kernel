// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"

#include "fmha_fwd.hpp"
#include "fmha_fwd_v3.hpp"
#include "mask.hpp"

namespace ck_tile {

template <typename DataType, bool PadSeqlen, bool IsMasking>
struct get_kernel
{
    using fmha_dtype = DataType;
    //                               M0   N0  K0   N1   K1
    using fmha_block_tile = sequence<256, 32, 128, 128, 32, 128>;

    using fmha_warp_gemm_shape = sequence<32, 32, 16>;

    using fmha_block_warps = sequence<8, 1, 1>;

    using fmha_shape = TileFmhaShape<fmha_block_tile,
                                     fmha_block_warps,
                                     fmha_warp_gemm_shape,
                                     fmha_block_warps,
                                     fmha_warp_gemm_shape,
                                     true // IsVLayoutRowMajor
                                     >;

    using fmha_traits = TileFmhaFwdV3Traits<PadSeqlen, // kPadSeqLenQ
                                            PadSeqlen, // kPadSeqLenK
                                            false,     // kPadHeadDimQ
                                            false,     // kPadHeadDimV
                                            false,     // kStoreLSE
                                            -1         // kBlockPerCu
                                            >;

    using fmha_mask = SimplifiedGenericAttentionMask<IsMasking>;

    using fmha_problem =
        BlockFmhaFwdV3PipelineProblem<typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::SaccDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::SMPLComputeDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::PDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                                      fmha_shape,
                                      false, // kIsGroupMode
                                      fmha_mask,
                                      fmha_traits>;

    using fmha_pipeline = BlockFmhaFwdV3Pipeline<fmha_problem>;

    using fmha_epilogue = Default2DEpilogue<
        Default2DEpilogueProblem<typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                                 true, // kPadM
                                 true  // kPadM
                                 >>;

    using type = FmhaFwdV3Kernel<fmha_pipeline, fmha_epilogue>;
};

template <typename DataType, bool PadSeqlen, bool IsMasking>
using get_kernel_t = typename get_kernel<DataType, PadSeqlen, IsMasking>::type;

template <typename Kernel>
float launch(const fmha_fwd_v3_args& args, const stream_config& config)
{
    auto kargs = Kernel::MakeKargs(args.q_ptr,
                                   args.k_ptr,
                                   args.v_ptr,
                                   nullptr, // lse_ptr
                                   args.o_ptr,
                                   args.seqlen_q,
                                   args.seqlen_k,
                                   args.hdim_q,
                                   args.hdim_v,
                                   args.nhead_q,
                                   args.nhead_q / args.nhead_k,
                                   args.scale_s,
                                   args.stride_q,
                                   args.stride_k,
                                   args.stride_v,
                                   args.stride_o,
                                   args.nhead_stride_q,
                                   args.nhead_stride_k,
                                   args.nhead_stride_v,
                                   0, // nhead_stride_lse
                                   args.nhead_stride_o,
                                   args.batch_stride_q,
                                   args.batch_stride_k,
                                   args.batch_stride_v,
                                   0, // batch_stride_lse
                                   args.batch_stride_o,
                                   args.window_size_left,
                                   args.window_size_right,
                                   args.mask_type);

    dim3 grids            = Kernel::GridSize(args.batch, args.nhead_q, args.seqlen_q, args.hdim_v);
    constexpr dim3 blocks = Kernel::BlockSize();
    constexpr index_t kBlockPerCu = Kernel::kBlockPerCu;

    return launch_kernel(config,
                         make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

template <fmha_fwd_v3_args::data_type_enum DataType, bool IsMasking>
struct type_tag
{
};

template <typename Tag>
float fmha_fwd_v3_dispatch(const fmha_fwd_v3_args& args, const stream_config& config);

} // namespace ck_tile