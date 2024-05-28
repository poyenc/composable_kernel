// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

// auto generated by generate.py
#include "fmha_fwd.hpp"

using fmha_dtype_0 = ck_tile::fp16_t;

using fmha_block_tile_0  = ck_tile::sequence<128, 128, 32, 256, 32, 256>;
using fmha_block_warps_0 = ck_tile::sequence<4, 1, 1>;
using fmha_warp_tile_0   = ck_tile::sequence<32, 32, 16>;

using fmha_shape_0 = ck_tile::TileFmhaShape<fmha_block_tile_0,
                                            fmha_block_warps_0,
                                            fmha_warp_tile_0,
                                            fmha_block_warps_0,
                                            fmha_warp_tile_0,
                                            true>;

using fmha_trait_0 =
    ck_tile::TileFmhaTraits<false, false, false, false, false, false, false, -1, MAX_NUM_SPLITS>;

using fmha_mask_0 = ck_tile::SimplifiedGenericAttentionMask<false>;

using fmha_pipeline_problem_0 =
    ck_tile::BlockFmhaPipelineProblem<typename FmhaFwdTypeConfig<fmha_dtype_0>::QDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype_0>::KDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype_0>::VDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype_0>::SaccDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype_0>::SMPLComputeDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype_0>::BiasDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype_0>::LSEDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype_0>::PDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype_0>::OaccDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype_0>::ODataType,
                                      fmha_shape_0,
                                      false,
                                      fmha_mask_0,
                                      fmha_trait_0>;

using fmha_fwd_splitkv_pipeline_0 =
    ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS<fmha_pipeline_problem_0>;
using fmha_fwd_splitkv_combine_pipeline_0 =
    ck_tile::BlockFmhaFwdSplitKVCombinePipeline<fmha_pipeline_problem_0>;

using fmha_epilogue_0 = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<ck_tile::fp16_t>::OaccDataType,
                                      typename FmhaFwdTypeConfig<ck_tile::fp16_t>::ODataType,
                                      false,
                                      false>>;

using fmha_fwd_splitkv_kernel_0 =
    ck_tile::FmhaFwdSplitKVKernel<ck_tile::FmhaFwdSplitKVTilePartitioner<fmha_shape_0>,
                                  fmha_fwd_splitkv_pipeline_0,
                                  fmha_epilogue_0>;
using fmha_splitkv_combine_kernel_0 = ck_tile::FmhaFwdSplitKVCombineKernel<
    ck_tile::FmhaFwdSplitKVCombineTilePartitioner<fmha_shape_0>,
    fmha_fwd_splitkv_combine_pipeline_0,
    fmha_epilogue_0>;

using trait_0 = fmha_fwd_traits_<256,
                                 ck_tile::fp16_t,
                                 false,
                                 128,
                                 128,
                                 32,
                                 256,
                                 32,
                                 256,
                                 true,
                                 ck_tile::BlockFmhaPipelineEnum::QRKSVS,
                                 fmha_mask_0,
                                 false,
                                 false,
                                 false,
                                 false,
                                 false,
                                 false,
                                 false>;

#include <iostream>

template <>
float fmha_fwd_splitkv_<trait_0>(const ck_tile::stream_config& s, fmha_fwd_args a)
{
    using k_ = fmha_fwd_splitkv_kernel_0;
    if(s.log_level_ > 0)
        std::cout << ", " << k_::GetName() << std::flush;

    float time_a = [&] {
        auto [kargs, grids]                    = fmha_fwd_splitkv_create_kargs_and_grids<k_>(a);
        constexpr dim3 blocks                  = k_::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
        return ck_tile::launch_kernel<blocks.x, kBlockPerCu>(s, k_{}, grids, blocks, 0, kargs);
    }();

    float time_b = [&] {
        using combine_k_      = fmha_splitkv_combine_kernel_0;
        auto [kargs, grids]   = fmha_fwd_splitkv_combine_create_kargs_and_grids<combine_k_>(a);
        constexpr dim3 blocks = k_::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
        return ck_tile::launch_kernel<blocks.x, kBlockPerCu>(
            s, combine_k_{}, grids, blocks, 0, kargs);
    }();

    return time_a + time_b;
}
