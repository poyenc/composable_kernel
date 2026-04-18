// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

#define ENABLE_ASM_MARKER 1
#if ENABLE_ASM_MARKER
#define ASM_MARKER(marker)               \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #marker); \
    __builtin_amdgcn_sched_barrier(0);
#else
#define ASM_MARKER(marker)
#endif

namespace ck_tile {

template <typename BlockGemm>
static constexpr ck_tile::index_t block_gemm_mfma_count_v =
    BlockGemm::MIterPerWarp * BlockGemm::NIterPerWarp * BlockGemm::KIterPerWarp *
    (BlockGemm::WarpGemm::kK / BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK);

template <typename PipelineProblem, typename Policy = BlockFmhaV3PipelineDefaultPolicy>
struct CoreLoopSchedulingParams
{
    using QKBlockGemm =
        ck_tile::remove_cvref_t<decltype(Policy::template GetQKBlockGemm<PipelineProblem>())>;
    using PVBlockGemm =
        ck_tile::remove_cvref_t<decltype(Policy::template GetPVBlockGemm<PipelineProblem>())>;

    static constexpr ck_tile::index_t kMfmaPerWarpGemm0 = block_gemm_mfma_count_v<QKBlockGemm>;
    static constexpr ck_tile::index_t kMfmaPerWarpGemm1 = block_gemm_mfma_count_v<PVBlockGemm>;

    static constexpr bool kIsMasking = PipelineProblem::FmhaMask::IsMasking;
};

template <typename PipelineProblem>
struct CoreLoopSchedulerDefaultBase
{
    using Params = CoreLoopSchedulingParams<PipelineProblem>;

    CK_TILE_DEVICE static constexpr void schedule_gemm0_compute()
    {
        // First MFMA: front-load 4 TRANS to fill pipeline
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::TRANS, 4, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 2, 0);
        // Middle MFMAs: steady-state 2 TRANS each
        static_for<0, Params::kMfmaPerWarpGemm0 - 2, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::TRANS, 2, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 2, 0);
        });
        // Last MFMA: no remaining TRANS
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 2, 0);
    }

    CK_TILE_DEVICE static constexpr void schedule_gemm1_compute()
    {
        static_for<0, Params::kMfmaPerWarpGemm1, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 5, 0);
        });
    }

    CK_TILE_DEVICE static constexpr void schedule_load_phase()
    {
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 2, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::SALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::SALU, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::SALU, 2, 0);
    }

    template <ck_tile::index_t WaveGroup, ck_tile::index_t Phase>
    CK_TILE_DEVICE static constexpr void schedule(ck_tile::number<WaveGroup>,
                                                  ck_tile::number<Phase>)
    {
        constexpr ck_tile::index_t effective = (WaveGroup == 0) ? Phase : (Phase + 3) % 4;
        if constexpr(effective == 0)
            schedule_gemm0_compute();
        else if constexpr(effective == 2)
            schedule_gemm1_compute();
        else
            schedule_load_phase();
    }
};

template <typename PipelineProblem, typename QDataType, typename KDataType, typename VDataType>
struct CoreLoopSchedulerImpl;

template <typename PipelineProblem>
struct CoreLoopSchedulerImpl<PipelineProblem, ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>
    : CoreLoopSchedulerDefaultBase<PipelineProblem>
{
};

template <typename PipelineProblem>
struct CoreLoopSchedulerImpl<PipelineProblem, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>
    : CoreLoopSchedulerDefaultBase<PipelineProblem>
{
};

template <typename PipelineProblem>
struct CoreLoopSchedulerImpl<PipelineProblem, ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::fp8_t>
    : CoreLoopSchedulerDefaultBase<PipelineProblem>
{
    using Base   = CoreLoopSchedulerDefaultBase<PipelineProblem>;
    using Params = typename Base::Params;

    CK_TILE_DEVICE static constexpr void schedule_gemm0_compute()
    {
        // K iter 0: 32 TRANS (v_exp_f32) + 29 VALU (v_add reduction + v_sub + permlane)
        // 4 TRANS x 8 = 32, 4 VALU x 8 = 32 (slightly over 29, pulls from K iter 1)
        static_for<0, Params::kMfmaPerWarpGemm0 / 2, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::TRANS, 4, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 4, 0);
        });
        // K iter 1: ~89 VALU (v_mul scale + v_cvt_pk_fp8 + o_acc rescale)
        // Increased from 6 to 8 to absorb v_cvt_pk_fp8_f32 tail into MFMA window
        static_for<0, Params::kMfmaPerWarpGemm0 / 2, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 6, 0);
        });
    }

    CK_TILE_DEVICE static constexpr void schedule_gemm1_compute()
    {
#if !CK_TILE_DISABLE_PACKED_FP32
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 4, 0);
#endif
        // First half: v_perm + v_max3 + permlane chain + v_fma (~57 VALU)
        static_for<0, Params::kMfmaPerWarpGemm1 / 2, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 6, 0);
        });
        // Second half: v_fma chain + v_mul O rescale (~33 VALU)
        // pk_mul (16 ops in asm volatile) invisible to scheduler
        static_for<0, Params::kMfmaPerWarpGemm1 / 2, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::VALU, 6, 0);
        });
    }

    // Must override schedule() — static methods have no virtual dispatch
    template <ck_tile::index_t WaveGroup, ck_tile::index_t Phase>
    CK_TILE_DEVICE static constexpr void schedule(ck_tile::number<WaveGroup>,
                                                  ck_tile::number<Phase>)
    {
        constexpr ck_tile::index_t effective = (WaveGroup == 0) ? Phase : (Phase + 3) % 4;

        if constexpr(effective == 0)
            schedule_gemm0_compute();
        else if constexpr(effective == 2)
            schedule_gemm1_compute();
        else
            Base::schedule_load_phase();
    }
};

template <typename PipelineProblem>
struct CoreLoopScheduler : CoreLoopSchedulerImpl<PipelineProblem,
                                                 typename PipelineProblem::QDataType,
                                                 typename PipelineProblem::KDataType,
                                                 typename PipelineProblem::VDataType>
{
};

namespace detail {
CK_TILE_DEVICE float fma_impl_vsv(float a, float b, float c) { return a * b + c; }

CK_TILE_DEVICE float add_impl_vv(float lhs, float rhs)
{
    float result;
    asm volatile("v_add_f32_e32 %[result], %[lhs], %[rhs]"
                 : [result] "=v"(result)
                 : [lhs] "v"(lhs), [rhs] "v"(rhs));
    return result;
}

CK_TILE_DEVICE float mul_impl_vv(float lhs, float rhs)
{
    float result;
    asm volatile("v_mul_f32_e32 %[result], %[lhs], %[rhs]"
                 : [result] "=v"(result)
                 : [lhs] "v"(lhs), [rhs] "v"(rhs));
    return result;
}

CK_TILE_DEVICE fp16x2_t cvt_pk_fp16_f32(float a, float b)
{
    fp16x2_t result;
    asm volatile("v_cvt_pk_f16_f32 %[result], %[a], %[b]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
}

CK_TILE_DEVICE bf16x2_t cvt_pk_bf16_f32(float a, float b)
{
    bf16x2_t result;
    asm volatile("v_cvt_pk_bf16_f32 %[result], %[a], %[b]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
}

CK_TILE_DEVICE uint32_t cvt_pk_fp8_f32(float a, float b)
{
    uint32_t result;
    asm volatile("v_cvt_pk_fp8_f32 %[result], %[a], %[b]"
                 : [result] "=v"(result)
                 : [a] "v"(a), [b] "v"(b));
    return result;
}

CK_TILE_DEVICE fp32x2_t pk_mul_f32(fp32x2_t lhs, fp32x2_t rhs)
{
    asm volatile("v_pk_mul_f32 %[result], %[result], %[rhs]" : [result] "+v"(lhs) : [rhs] "v"(rhs));
    return lhs;
}
} // namespace detail

/// NOTICE: This pipeline is a work in progress and is awaiting upcoming compiler fixes and
/// instruction scheduling optimizations.
template <typename Problem_, typename Policy_ = BlockFmhaV3PipelineDefaultPolicy>
struct BlockFmhaFwdV3Pipeline
{
    using Problem             = ck_tile::remove_cvref_t<Problem_>;
    using Policy              = ck_tile::remove_cvref_t<Policy_>;
    using QDataType           = ck_tile::remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = ck_tile::remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = ck_tile::remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = ck_tile::remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = ck_tile::remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using LSEDataType         = ck_tile::remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType           = ck_tile::remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType        = ck_tile::remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType           = ck_tile::remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant    = ck_tile::remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask            = ck_tile::remove_cvref_t<typename Problem::FmhaMask>;
    static_assert(is_generic_attention_mask_v<FmhaMask>);

    static_assert(std::is_same_v<SaccDataType, SMPLComputeDataType>,
                  "we will the same dist tensor 'sp_compute' for both gemm0 & softmax");

    using BlockFmhaShape = ck_tile::remove_cvref_t<typename Problem::BlockFmhaShape>;

    using VLayout = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>);

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    static constexpr ck_tile::index_t kM0           = BlockFmhaShape::kM0;
    static constexpr ck_tile::index_t kN0           = BlockFmhaShape::kN0;
    static constexpr ck_tile::index_t kK0           = BlockFmhaShape::kK0;
    static constexpr ck_tile::index_t kN1           = BlockFmhaShape::kN1;
    static constexpr ck_tile::index_t kK1           = BlockFmhaShape::kK1;
    static constexpr ck_tile::index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr ck_tile::index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;

    static_assert(kQKHeaddim == 128 && kSubQKHeaddim == 128, "only supports hdim=hdim_v=128");

    static constexpr bool kIsGroupMode      = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kHasDropout       = Problem::kHasDropout;
    static constexpr auto QScaleEnum        = Problem::QScaleEnum;
    static constexpr bool kSkipMinSeqlenQ   = Problem::kSkipMinSeqlenQ;
    static_assert((BiasEnum == BlockAttentionBiasEnum::NO_BIAS && !kHasDropout && !kSkipMinSeqlenQ),
                  "enable unsupported features");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr ck_tile::index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr ck_tile::index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr ck_tile::index_t kAlignmentV =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();

    static constexpr ck_tile::index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

    static constexpr ck_tile::index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            return 2;
        }
    }();

    template <typename DataType, typename Descriptor>
    CK_TILE_DEVICE static constexpr auto make_lds_tile_window(DataType* __restrict__ base,
                                                              const Descriptor& desc)
    {
        using namespace ck_tile;

        auto tensor_view = make_tensor_view<address_space_enum::lds>(base, desc);
        return make_tile_window(tensor_view, desc.get_lengths(), {0, 0});
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_DEVICE auto
    operator()(multi_index<2> partition_index,
               const QDramBlockWindowTmp& __restrict__ q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& __restrict__ k_dram_block_window_tmp, // N0*K0 tile
               [[maybe_unused]] const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& __restrict__ v_dram_block_window_tmp, // N1*K1 tile
               [[maybe_unused]] const VElementFunction& v_element_func,
               LSEDramBlockWindowTmp& __restrict__ lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               [[maybe_unused]] const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               KDataType* __restrict__ smem_k0,
               KDataType* __restrict__ smem_k1,
               VDataType* __restrict__ smem_v0,
               VDataType* __restrict__ smem_v1,
               void* __restrict__ smem_ptr) const
    {
        using namespace ck_tile;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        const index_t warp_id       = partition_index[0];
        const index_t warp_group_id = warp_id / 4;
        const index_t stagger       = (warp_group_id != 0) ? 1 : 0;
        const index_t lane_id       = partition_index[1];

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp,
                                              Policy::template MakeQRegTileDistribution<Problem>(),
                                              partition_index);

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        auto k_lds_window_store = generate_tuple(
            [&](auto write_idx) {
                auto k_buf = (write_idx == 0 ? smem_k0 : smem_k1);
                return make_lds_tile_window(
                    k_buf, Policy::template MakeKLdsStoreBlockDescriptor<Problem>());
            },
            number<2>{});

        auto v_lds_window_store = generate_tuple(
            [&](auto write_idx) {
                auto v_buf = (write_idx == 0 ? smem_v0 : smem_v1);
                return make_lds_tile_window(
                    v_buf, Policy::template MakeVLdsStoreBlockDescriptor<Problem>());
            },
            number<2>{});

        constexpr auto all_zeros_partition_index = make_multi_index(0, 0);
        statically_indexed_array<decltype(make_tile_window(
                                     make_lds_tile_window<KDataType>(
                                         nullptr,
                                         Policy::template MakeKLdsLoadBlockDescriptor<Problem>()),
                                     Policy::template MakeKRegTileDistribution<Problem>(),
                                     all_zeros_partition_index)),
                                 2>
            k_lds_window_load;

        statically_indexed_array<decltype(make_tile_window(
                                     make_lds_tile_window<VDataType>(
                                         nullptr,
                                         Policy::template MakeVLdsLoadBlockDescriptor<Problem>()),
                                     Policy::template MakeVRegTileDistribution<Problem>(),
                                     all_zeros_partition_index)),
                                 2>
            v_lds_window_load;

        decltype(make_static_distributed_tensor<QDataType>(
            Policy::template MakeQRegTileDistribution<Problem>())) q_tile;

        union kv_tile_type
        {
            CK_TILE_DEVICE kv_tile_type() {}

            decltype(load_tile(k_lds_window_load(number<0>{}))) k_tile;

            decltype(load_tile_transpose(v_lds_window_load(number<0>{}))) v_tile;
        } kv_tile;

        union sp_compute_type
        {
            CK_TILE_DEVICE sp_compute_type() {}

            decltype(gemm_0.MakeCBlockTile()) sp_compute;
            decltype(make_static_distributed_tensor<PDataType>(
                Policy::template MakePRegTileDistribution<Problem>())) p;
        };
        statically_indexed_array<sp_compute_type, 2> sp;

        decltype(gemm_1.MakeCBlockTile()) o_acc;
        constexpr float RESCALE_THRESHOLD = 8.0f;

        decltype(block_tile_reduce<SMPLComputeDataType>(
            sp(number<0>{}).sp_compute, sequence<1>{}, f_max, SMPLComputeDataType{0})) m;
        decltype(m) l;

        // initialize k_lds_window and v_lds_window with all_zeros_partition_index
        // The actual per-thread offset is computed below and passed to load_tile_with_offset

        static_for<0, 2, 1>{}([&](auto idx) {
            k_lds_window_load(idx) =
                make_tile_window(make_lds_tile_window(
                                     [&] {
                                         if constexpr(idx == 0)
                                             return smem_k0;
                                         else
                                             return smem_k1;
                                     }(),
                                     Policy::template MakeKLdsLoadBlockDescriptor<Problem>()),
                                 Policy::template MakeKRegTileDistribution<Problem>(),
                                 all_zeros_partition_index);
        });

        static_for<0, 2, 1>{}([&](auto idx) {
            v_lds_window_load(idx) =
                make_tile_window(make_lds_tile_window(
                                     [&] {
                                         if constexpr(idx == 0)
                                             return smem_v0;
                                         else
                                             return smem_v1;
                                     }(),
                                     Policy::template MakeVLdsLoadBlockDescriptor<Problem>()),
                                 Policy::template MakeVRegTileDistribution<Problem>(),
                                 all_zeros_partition_index);
        });

        // Compute per-thread LDS load offset using hardcoded formulas (empirically derived)
        // Using hardcoded formulas instead of make_tensor_adaptor_coordinate() reduces VGPR
        // pressure because the ds_read base address becomes a simple lane_id computation
        // instead of requiring runtime coordinate tracking across the loop.
        const index_t k_lds_load_offset = [&] {
            if constexpr(std::is_same_v<KDataType, fp8_t>)
            {
                // fp8: use make_tensor_adaptor_coordinate for new tile distribution
                // TODO: derive hardcoded formula to reduce VGPR pressure
                constexpr auto k_tile_dstr = Policy::template MakeKRegTileDistribution<Problem>();
                constexpr auto k_lds_desc = Policy::template MakeKLdsLoadBlockDescriptor<Problem>();
                constexpr index_t NDimY   = decltype(k_tile_dstr)::NDimY;
                auto top_index            = container_concat(partition_index, multi_index<NDimY>{});
                const auto adaptor_coord  = make_tensor_adaptor_coordinate(
                    k_tile_dstr.get_ps_ys_to_xs_adaptor(), top_index);
                const auto bottom_idx = adaptor_coord.get_bottom_index();
                const auto lds_coord  = make_tensor_coordinate(k_lds_desc, bottom_idx);
                return lds_coord.get_offset();
            }
            else
            {
                // bf16/fp16: use hardcoded formula (empirically derived)
                index_t start_row   = lane_id % 32;
                index_t start_col   = lane_id / 32 * 8;
                index_t warp_offset = (start_row / 8) * (4 * 4) / 2;
                return (start_row * 64) + start_col + warp_offset;
            }
        }();

        const index_t v_lds_load_offset = [&] {
            if constexpr(std::is_same_v<VDataType, fp8_t>)
            {
                // fp8: use make_tensor_adaptor_coordinate for new tile distribution
                // TODO: derive hardcoded formula to reduce VGPR pressure
                constexpr auto v_tile_dstr = Policy::template MakeVRegTileDistribution<Problem>();
                constexpr auto v_lds_desc = Policy::template MakeVLdsLoadBlockDescriptor<Problem>();
                constexpr index_t NDimY   = decltype(v_tile_dstr)::NDimY;
                auto top_index            = container_concat(partition_index, multi_index<NDimY>{});
                const auto adaptor_coord  = make_tensor_adaptor_coordinate(
                    v_tile_dstr.get_ps_ys_to_xs_adaptor(), top_index);
                const auto bottom_idx = adaptor_coord.get_bottom_index();
                const auto lds_coord  = make_tensor_coordinate(v_lds_desc, bottom_idx);
                return lds_coord.get_offset();
            }
            else
            {
                // bf16/fp16: use hardcoded formula (empirically derived)
                index_t group_idx     = lane_id / 16;
                index_t local_lane_id = lane_id % 16;
                index_t start_row     = (group_idx / 2) * 4 + local_lane_id / 4;
                index_t start_col     = (group_idx % 2) * 16 + (local_lane_id % 4) * 4;
                return (start_row * 64) + start_col;
            }
        }();

        {
            auto origin_q      = load_tile(q_dram_window);
            auto transformed_q = tile_elementwise_in(q_element_func, origin_q);

            q_tile = transformed_q;
        }

        clear_tile(o_acc);
        set_tile(m, bit_cast<float>(0xff7fffff)); // a bit larger than -infinity
        clear_tile(l);

        const auto q_origin = q_dram_window.get_window_origin();
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0);
        index_t kv_token_start    = seqlen_k_start;

        auto k_dram_window = make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                                              k_dram_block_window_tmp.get_window_lengths(),
                                              {seqlen_k_start, 0},
                                              Policy::template MakeKDramTileDistribution<Problem>(),
                                              partition_index);

        auto v_dram_window = make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                                              v_dram_block_window_tmp.get_window_lengths(),
                                              {seqlen_k_start, 0}, // TODO: hdim split?
                                              Policy::template MakeVDramTileDistribution<Problem>(),
                                              partition_index);

        // prefetch K tile
        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;
        static_assert(1 == k0_loops);
        static_assert(1 == k1_loops);
        static_assert(kN0 == kK1);

        constexpr index_t NumWarpGroups = Problem::kBlockSize / Policy::NumThreadPerWarpGroup;
        static_assert(NumWarpGroups == 2);

        constexpr int K_mem_su_ld_insts = k_dram_window.get_num_of_access();
        constexpr int V_mem_su_ld_insts = v_dram_window.get_num_of_access();

        auto K_mem_load = [&](auto k_lds_write_idx) {
            async_load_tile(k_lds_window_store(k_lds_write_idx),
                            k_dram_window,
                            number<-1>{},
                            bool_constant<false>{});

            /// FIXME: use the future-predicting method to move the window
            // move K tile windows
            move_tile_window(k_dram_window, {kN0, 0});
        };

        auto K_lds_load = [&](auto k_lds_read_idx) {
            kv_tile.k_tile =
                load_tile_with_offset(k_lds_window_load(k_lds_read_idx), k_lds_load_offset);
        };

        auto V_mem_load = [&](auto v_lds_write_idx) {
            async_load_tile(v_lds_window_store(v_lds_write_idx),
                            v_dram_window,
                            number<-1>{},
                            bool_constant<false>{});

            /// FIXME: use the future-predicting method to move the window
            move_tile_window(v_dram_window, {kK1, 0});
        };

        auto V_lds_load = [&](auto v_lds_read_idx) {
            kv_tile.v_tile = load_tile_transpose_with_offset(v_lds_window_load(v_lds_read_idx),
                                                             v_lds_load_offset);
        };

        decltype(m) m_old;
        SMPLComputeDataType o_acc_scale; // rescale o_acc in fmha_rescale_o()
        auto fmha_logits_trans = [&](auto sp_reg_idx) {
            if constexpr(kHasLogitsSoftCap)
            {
                auto apply_logits_transform = [&variant, &variant_params, &block_indices](
                                                  auto& logits) {
                    logits = variant.LogitsTransform(variant_params,
                                                     variant.QueryTransform(variant_params, logits),
                                                     block_indices.batch_idx,
                                                     block_indices.qo_head_idx,
                                                     block_indices.kv_head_idx);
                };

                tile_elementwise_inout(apply_logits_transform, sp(sp_reg_idx).sp_compute);
            }
        };

        auto fmha_alu0 = [&](auto sp_reg_idx) {
            m_old = m; // m{j-1}
            static_assert(m.thread_buf_.size() == 1,
                          "assuming that each thread holds 1 rowmax value");
            auto m_latest = block_tile_reduce<SMPLComputeDataType>(
                sp(sp_reg_idx).sp_compute, sequence<1>{}, f_max, m.thread_buf_[0]);
#if defined(__gfx950__)
            // assuming that we are using 32x32 mfma
            int32x2_t swapped_regs =
                __builtin_amdgcn_permlane32_swap(bit_cast<int32_t>(m_latest.thread_buf_[0]),
                                                 bit_cast<int32_t>(m_latest.thread_buf_[0]),
                                                 false,
                                                 false);
            /// TODO: eliminate 2 redudant v_max_f32 instructions generated by the compiler
            m_latest.thread_buf_[0] = f_max(bit_cast<SMPLComputeDataType>(swapped_regs.x),
                                            bit_cast<SMPLComputeDataType>(swapped_regs.y));
#else
            block_tile_reduce_sync(m_latest, f_max, bool_constant<false>{});
#endif
            m = m_latest;

            // threshold check before subtraction (opus_attn style)
            // ensures P = exp2(sp_compute - m) is consistent with chosen m
            {
                float max_diff = m.thread_buf_[0] - m_old.thread_buf_[0];
                bool below     = (max_diff <= RESCALE_THRESHOLD);
                bool all_below = (__builtin_amdgcn_ballot_w64(below) ==
                                  __builtin_amdgcn_read_exec());
                if(__builtin_expect(all_below, 1))
                {
                    m.thread_buf_[0] = m_old.thread_buf_[0];
                }
            }

            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    if constexpr(kHasLogitsSoftCap)
                    {
                        sp(sp_reg_idx).sp_compute(i_j_idx) =
                            sp(sp_reg_idx).sp_compute(i_j_idx) - m(i_j_idx);
                    }
                    else
                    {
                        sp(sp_reg_idx).sp_compute(i_j_idx) = detail::fma_impl_vsv(
                            sp(sp_reg_idx).sp_compute(i_j_idx), scale_s, -scale_s * m(i_j_idx));
                    }
                });
            });
            /// TODO: move some fmha_alu1() code here if necessary
        };

        auto fmha_alu1 = [&](auto sp_reg_idx) {
            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    sp(sp_reg_idx).sp_compute(i_j_idx) =
                        ck_tile::exp2(sp(sp_reg_idx).sp_compute(i_j_idx));
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                sp(sp_reg_idx).sp_compute,
                sequence<1>{},
                f_sum,
                SMPLComputeDataType{0}); // rowsum(Pcompute{j})
            static_assert(rowsum_p.thread_buf_.size() == 1,
                          "assuming that each thread holds 1 rowsum value");
#if defined(__gfx950__)
            // assuming that we are using 32x32 mfma
            int32x2_t swapped_regs =
                __builtin_amdgcn_permlane32_swap(bit_cast<int32_t>(rowsum_p.thread_buf_[0]),
                                                 bit_cast<int32_t>(rowsum_p.thread_buf_[0]),
                                                 false,
                                                 false);
            rowsum_p.thread_buf_[0] = f_sum(bit_cast<SMPLComputeDataType>(swapped_regs.x),
                                            bit_cast<SMPLComputeDataType>(swapped_regs.y));
#else
            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
#endif

            // l{j}
            /// Note: The compiler keeps moving the following instructions elsewhere because 'l'
            /// is first consumed later. To anchor them here, we rewrite the final addition in
            /// inline assembly to create a dependency, forcing the dependent instructions to
            /// be emitted at this point.
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                l(i_idx) = detail::add_impl_vv(l[i_idx], rowsum_p[i_idx]);
            });

            /// Note: The compiler keeps sinking the conversion instructions because the
            /// result 'p' is only consumed later. To anchor them here, we rewrite
            /// the cast_tile() call as inline assembly, forcing the conversions to be
            /// emitted at this point.
            static_assert(sp(sp_reg_idx).p.thread_buf_.size() % 2 == 0);
            static_for<0, sp(sp_reg_idx).p.thread_buf_.size(), 2>{}([&](auto idx) {
                float x = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx]);
                float y = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1]);
                if constexpr(std::is_same_v<PDataType, fp16_t>)
                {
                    auto casted                           = detail::cvt_pk_fp16_f32(x, y);
                    sp(sp_reg_idx).p.thread_buf_[idx]     = casted.x;
                    sp(sp_reg_idx).p.thread_buf_[idx + 1] = casted.y;
                }
                else if constexpr(std::is_same_v<PDataType, bf16_t>)
                {
                    auto casted                           = detail::cvt_pk_bf16_f32(x, y);
                    sp(sp_reg_idx).p.thread_buf_[idx]     = casted.x;
                    sp(sp_reg_idx).p.thread_buf_[idx + 1] = casted.y;
                }
                else if constexpr(std::is_same_v<PDataType, fp8_t>)
                {
                    uint32_t packed = detail::cvt_pk_fp8_f32(x, y);
                    sp(sp_reg_idx).p.thread_buf_[idx] =
                        bit_cast<fp8_t>(static_cast<uint8_t>(packed & 0xFF));
                    sp(sp_reg_idx).p.thread_buf_[idx + 1] =
                        bit_cast<fp8_t>(static_cast<uint8_t>((packed >> 8) & 0xFF));
                }
            });

            /// Note: Place fmha_alu1() at the end of the phase. The surrounding inline assembly
            /// can interfere with the behavior of sched_group_barrier(), so ending the phase here
            /// avoids unintended reordering.
        };

        auto gemm = [&](auto sp_reg_idx, auto gemm_idx) {
            if constexpr(gemm_idx == 0)
            {
                clear_tile(sp(sp_reg_idx).sp_compute); // initialize C
                gemm_0(sp(sp_reg_idx).sp_compute,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       get_slice_tile(kv_tile.k_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kN0, k0_loops * kK0>{}));
            }
            else
            {
                gemm_1(o_acc,
                       get_slice_tile(sp(sp_reg_idx).p,
                                      sequence<0, (k1_loops - 1) * kK1>{},
                                      sequence<kM0, k1_loops * kK1>{}),
                       get_slice_tile(kv_tile.v_tile,
                                      sequence<0, (k1_loops - 1) * kK1>{},
                                      sequence<kN1, k1_loops * kK1>{}));
            }
        };

        auto cl_calc = [&](auto sp_reg_idx, auto gemm_idx) {
            if constexpr(gemm_idx == 0)
            {
                clear_tile(sp(sp_reg_idx).sp_compute); // initialize C
                gemm_0(sp(sp_reg_idx).sp_compute,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       get_slice_tile(kv_tile.k_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kN0, k0_loops * kK0>{}));
            }
            else
            {
                gemm_1(o_acc,
                       get_slice_tile(sp(sp_reg_idx).p,
                                      sequence<0, (k1_loops - 1) * kK1>{},
                                      sequence<kM0, k1_loops * kK1>{}),
                       get_slice_tile(kv_tile.v_tile,
                                      sequence<0, (k1_loops - 1) * kK1>{},
                                      sequence<kN1, k1_loops * kK1>{}));
                fmha_alu0(number<1>{} - sp_reg_idx);
            }
        };

        auto fmha_rescale_o = [&] {
            // fmha_alu0 already applied threshold+ballot to choose m.
            // If m was reverted to m_old, no rescale needed.
            if(m.thread_buf_[0] == m_old.thread_buf_[0])
            {
                return;
            }

            o_acc_scale = [&] {
                if constexpr(kHasLogitsSoftCap)
                {
                    return ck_tile::exp2(m_old.thread_buf_[0] - m.thread_buf_[0]);
                }
                else
                {
                    return ck_tile::exp2(scale_s *
                                         (m_old.thread_buf_[0] - m.thread_buf_[0]));
                }
            }();

            l.thread_buf_[0] *= o_acc_scale;

            static_for<0, o_acc.thread_buf_.size(), 1>{}(
                [&](auto idx) { o_acc.thread_buf_[idx] *= o_acc_scale; });
        };

        auto fmha_mask = [&](auto sp_reg_idx) {
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(
                    q_origin.at(number<0>{}), kv_token_start, number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    // Per-warp fast path: each warp handles kMPerWarp contiguous rows.
                    // Skip set_tile_if for warps fully inside the causal triangle.
                    constexpr index_t kMPerWarp = kM0 * get_warp_size() / kBlockSize;
                    bool warp_needs_mask =
                        mask.IsEdgeTile(q_origin.at(number<0>{}) + warp_id * kMPerWarp,
                                        kv_token_start,
                                        number<kMPerWarp>{},
                                        number<kN0>{});
                    if(warp_needs_mask)
                    {
                        set_tile_if(
                            sp(sp_reg_idx).sp_compute,
                            -numeric<SMPLComputeDataType>::infinity(),
                            [&](auto tile_idx) {
                                const auto row =
                                    q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                const auto col = kv_token_start + tile_idx.at(number<1>{});
                                return !variant.LogitsMask(variant_params,
                                                           block_indices.batch_idx,
                                                           row,
                                                           col,
                                                           block_indices.qo_head_idx,
                                                           block_indices.kv_head_idx);
                            },
                            partition_index);
                    }
                }
            }
        };

        auto cl_load = [&](auto load_type, auto mem_wr_idx, auto lds_rd_idx) {
            if constexpr(load_type == 0)
            {
                V_mem_load(mem_wr_idx);
                K_lds_load(lds_rd_idx);
            }
            else
            {
                K_mem_load(mem_wr_idx);
                V_lds_load(lds_rd_idx);
            }
        };

        auto core_loop = [&]() {
            auto gemm0 = number<0>{};
            auto gemm1 = number<1>{};

            auto memV = number<0>{};
            auto memK = number<1>{};

            using Scheduler = CoreLoopScheduler<Problem>;

            // --- Cluster 0: GEMM0(sp[1]) + fmha_alu1(sp[0]) + logits_trans(sp[1]) ---
            {
                __builtin_amdgcn_sched_barrier(0);
                ASM_MARKER("cluster0 (pi=0)");
                s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                cl_calc(number<1>{}, gemm0);
                fmha_alu1(number<0>{});
                fmha_logits_trans(number<1>{});

                Scheduler::schedule(number<0>{}, number<0>{});
                __builtin_amdgcn_sched_barrier(0);
            }
            // --- Cluster 1: K_mem_load(1) + V_lds_load(0) + fmha_mask(sp[1]) ---
            {
                ASM_MARKER("cluster1 (pi=0)");
                s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                cl_load(memK, number<1>{}, number<0>{});
                Scheduler::schedule(number<0>{}, number<1>{});
                fmha_mask(number<1>{});

                __builtin_amdgcn_sched_barrier(0);
            }
            // --- Cluster 2: GEMM1(sp[0]) + fmha_rescale_o ---
            {
                ASM_MARKER("cluster2 (pi=0)");
                s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_nop 1");
                __builtin_amdgcn_sched_barrier(0);
                cl_calc(number<0>{}, gemm1);
                fmha_rescale_o();
                Scheduler::schedule(number<0>{}, number<2>{});

                __builtin_amdgcn_sched_barrier(0);
            }
            // --- Cluster 3: V_mem_load(0) + K_lds_load(0) + loop check ---
            {
                ASM_MARKER("cluster3 (pi=0)");
                s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                cl_load(memV, number<0>{}, number<0>{});

                Scheduler::schedule(number<0>{}, number<3>{});
                kv_token_start += kN0;
                if(num_total_loop <= ++i_total_loops)
                {
                    return false;
                }
            }

            // --- Cluster 4: GEMM0(sp[0]) + fmha_alu1(sp[1]) + logits_trans(sp[0]) ---
            {
                __builtin_amdgcn_sched_barrier(0);
                ASM_MARKER("cluster4 (pi=1)");
                s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                cl_calc(number<0>{}, gemm0);
                fmha_alu1(number<1>{});
                fmha_logits_trans(number<0>{});

                Scheduler::schedule(number<0>{}, number<0>{});
                __builtin_amdgcn_sched_barrier(0);
            }
            // --- Cluster 5: K_mem_load(0) + V_lds_load(1) + fmha_mask(sp[0]) ---
            {
                ASM_MARKER("cluster5 (pi=1)");
                s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                cl_load(memK, number<0>{}, number<1>{});
                Scheduler::schedule(number<0>{}, number<1>{});
                fmha_mask(number<0>{});

                __builtin_amdgcn_sched_barrier(0);
            }
            // --- Cluster 6: GEMM1(sp[1]) + fmha_rescale_o ---
            {
                ASM_MARKER("cluster6 (pi=1)");
                s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                asm volatile("s_nop 1");
                __builtin_amdgcn_sched_barrier(0);
                cl_calc(number<1>{}, gemm1);
                fmha_rescale_o();
                Scheduler::schedule(number<0>{}, number<2>{});

                __builtin_amdgcn_sched_barrier(0);
            }
            // --- Cluster 7: V_mem_load(1) + K_lds_load(1) + loop check ---
            {
                ASM_MARKER("cluster7 (pi=1)");
                s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
                cl_load(memV, number<1>{}, number<1>{});

                Scheduler::schedule(number<0>{}, number<3>{});
                kv_token_start += kN0;
                if(num_total_loop <= ++i_total_loops)
                {
                    return false;
                }
            }

            return true;
        };

        auto fmha_post_process = [&](auto d) {
            auto ps_pi        = number<1>{} - d;
            auto V_lds_rd_idx = ps_pi;

            if(1 < num_total_loop)
            {
                s_waitcnt<K_mem_su_ld_insts>();
            }
            else
            {
                s_waitcnt<0>();
            }
            __builtin_amdgcn_s_barrier();

            V_lds_load(V_lds_rd_idx);
            fmha_alu1(ps_pi);

            s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();

            auto xdl_SP_p23_reg_idx = ps_pi;
            gemm(xdl_SP_p23_reg_idx, /*gemm_idx=*/number<1>{});
        };

        if(num_total_loop > 0)
        {
            // pre-stage
            {
                ASM_MARKER("before pre-stage");
                // (1) load K0 to LDS & VGPR
                K_mem_load(number<0>{}); // mem_K0

                s_waitcnt<0>();
                __builtin_amdgcn_s_barrier();

                K_lds_load(number<0>{}); // lds_K0

                s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                __builtin_amdgcn_s_barrier();

                // (2) prefetch K1 and V0 to LDS in parallel with GEMM0
                if(1 < num_total_loop)
                {
                    K_mem_load(number<1>{}); // mem_K1
                }
                V_mem_load(number<0>{}); // mem_V0

                // (3) mfma (Q*K0) + softmax
                gemm(number<0>{}, /*gemm_idx=*/number<0>{});
                fmha_logits_trans(number<0>{});
                fmha_mask(number<0>{});
                /// TODO: find better way to map fmha_alu(0,96) call
                fmha_alu0(number<0>{});
                fmha_rescale_o();

                kv_token_start += kN0;
                ++i_total_loops;
                if(num_total_loop <= i_total_loops)
                {
                    goto label_main_loops_exit;
                }

                if(2 < num_total_loop)
                {
                    K_mem_load(number<0>{}); // mem_K2

                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_s_barrier();
                }

                // Prologue stagger: WG1 calls one extra barrier to offset
                // by 2 clusters relative to WG0. After this, WG0 and WG1
                // execute the same core_loop code but are phase-shifted.
                if(stagger)
                {
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts,
                              waitcnt_arg::kMaxExpCnt,
                              0>();
                    __builtin_amdgcn_s_barrier();
                }

                ASM_MARKER("end pre-stage");
            }

            if(1 < num_total_loop)
            {
                V_mem_load(number<1>{}); // V1
                K_lds_load(number<1>{}); // K1

                s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts,
                          waitcnt_arg::kMaxExpCnt,
                          0>();
                __builtin_amdgcn_s_barrier();
                while(core_loop())
                    ;

                // Epilogue realign: WG0 exits the loop first (it was
                // 2 clusters ahead). This barrier lets WG0 wait for
                // WG1 to finish its last iteration.
                if(!stagger)
                {
                    __builtin_amdgcn_s_barrier();
                }
            }
        label_main_loops_exit:
            if(num_total_loop % 2)
            {
                fmha_post_process(number<1>{});
            }
            if(!(num_total_loop % 2))
            {
                fmha_post_process(number<0>{});
            }
        }

        // store lse
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_spans = decltype(lse)::get_distributed_spans();
            sweep_tile_span(lse_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                lse(i_idx)           = m[i_idx] / C_LOG2E + log(l[i_idx]);
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_DEVICE auto
    operator()(multi_index<2> partition_index,
               const QDramBlockWindowTmp& __restrict__ q_dram_block_window_tmp, // M0*K0 tile
               const KDramBlockWindowTmp& __restrict__ k_dram_block_window_tmp, // N0*K0 tile
               const VDramBlockWindowTmp& __restrict__ v_dram_block_window_tmp, // N1*K1 tile
               LSEDramBlockWindowTmp& __restrict__ lse_dram_block_window_tmp,   // M0*1 tile
               FmhaMask mask,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               KDataType* __restrict__ smem_k0,
               KDataType* __restrict__ smem_k1,
               VDataType* __restrict__ smem_v0,
               VDataType* __restrict__ smem_v1,
               void* __restrict__ smem_ptr) const
    {
        using namespace ck_tile;

        return operator()(partition_index,
                          q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          lse_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_k0,
                          smem_k1,
                          smem_v0,
                          smem_v1,
                          smem_ptr);
    }
};

} // namespace ck_tile
