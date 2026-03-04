// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

/// V3 pipeline adapted for batch prefill with scatter-gather KV loads (paged KV cache).
///
/// This pipeline inherits the V3 4-phase double warp group architecture
/// (CoreLoopScheduler, double-buffered LDS, phase barriers) and replaces
/// contiguous K/V DRAM loads with scatter-gather loads using page table lookups.
///
/// Key differences from BlockFmhaFwdV3Pipeline:
///   - K/V DRAM windows are tile_scatter_gather instead of tile_window
///   - Per-iteration page offset recomputation (load_physical_pages + kv_offset_array_transform)
///   - Additional operator() parameters: page_idx, stride_k/v, page_stride_k/v
///   - Problem type requires kPageBlockSize, kVectorSize, kKVMemoryLayout
template <typename Problem_, typename Policy_ = BlockFmhaV3PipelineDefaultPolicy>
struct BlockFmhaBatchPrefillV3Pipeline
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

    // Paged KV cache parameters
    static constexpr ck_tile::index_t kPageBlockSize = Problem::kPageBlockSize;
    static constexpr ck_tile::index_t kVectorSize    = Problem::kVectorSize;
    static constexpr auto kKVMemoryLayout            = Problem::kKVMemoryLayout;
    static_assert(kKVMemoryLayout == BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT,
                  "V3 batch prefill only supports LINEAR_LAYOUT (VECTORIZED requires sub-dword "
                  "async loads which violate buffer addressing constraints)");

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
    static_assert(QScaleEnum != BlockAttentionQuantScaleEnum::KV_BLOCKSCALE,
                  "V3 batch prefill does not support KV_BLOCKSCALE quantization");
    static_assert(!kPadHeadDimQ && !kPadHeadDimV,
                  "V3 batch prefill requires hdim=128 which is always aligned, no padding needed");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    //
    // Unlike the contiguous V3 fwd pipeline which uses alignment=1 for padded dims,
    // scatter-gather relies on the tensor descriptor's GuaranteedLastDimensionVectorLength
    // to determine ScalarPerVector for buffer loads. Setting alignment=1 would result in
    // per-element (2-byte bf16) loads, violating the 4-byte dword minimum for async buffer loads.
    // Since hdim is always 128 (enforced by static_assert above), the full alignment is safe.
    static constexpr ck_tile::index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr ck_tile::index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr ck_tile::index_t kAlignmentV = Policy::template GetAlignmentV<Problem>();

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

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // V3 kernel allocates smem_k[2] and smem_v[2] separately (double buffered),
        // plus epilogue smem. Return max(K_single, V_single) * 2 for the pipeline portion.
        return 2 * Policy::template GetSmemSizeK<Problem>() +
               2 * Policy::template GetSmemSizeV<Problem>();
    }

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
               void* __restrict__ smem_ptr,
               // Paged KV cache parameters
               const index_t* page_idx,
               index_t stride_k,
               index_t stride_v,
               index_t page_stride_k,
               index_t page_stride_v,
               index_t max_page_table_idx = 0x7FFFFFFF) const
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
        constexpr index_t fmha_alu_D_reg_cnt =
            6; // Threshold for determining how many fmha_alu_D_upd() unpacked
               // instructions to relocate to fmha_alu1().
        static_assert(fmha_alu_D_reg_cnt % 2 == 0 &&
                      fmha_alu_D_reg_cnt <= o_acc.thread_buf_.size());

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
        const index_t k_lds_load_offset = [&] {
            if constexpr(std::is_same_v<KDataType, fp8_t>)
            {
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
                index_t start_row   = lane_id % 32;
                index_t start_col   = lane_id / 32 * 8;
                index_t warp_offset = (start_row / 8) * (4 * 4) / 2;
                return (start_row * 64) + start_col + warp_offset;
            }
        }();

        const index_t v_lds_load_offset = [&] {
            if constexpr(std::is_same_v<VDataType, fp8_t>)
            {
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

        // =====================================================================
        // Scatter-gather K DRAM window setup (replaces contiguous make_tile_window)
        // =====================================================================
        auto k_dist               = Policy::template MakeKDramTileDistribution<Problem>();
        auto k_coord              = k_dist.calculate_index();
        using KDstrEncode         = typename decltype(k_dist)::DstrEncode;
        constexpr index_t NRepeat = KDstrEncode::hs_lengthss_[number<0>{}][number<0>{}];

        index_t current_seq_k = seqlen_k_start;
        statically_indexed_array<index_t, NRepeat> k_physical_pages{};
        statically_indexed_array<index_t, NRepeat> k_offsets;

        load_physical_pages<statically_indexed_array<index_t, NRepeat>,
                            decltype(k_coord),
                            0,
                            kPageBlockSize,
                            0,
                            NRepeat,
                            kN0 / NRepeat,
                            kKVMemoryLayout,
                            true,
                            kN0>(
            page_idx, k_coord, current_seq_k, k_physical_pages, max_page_table_idx);

        kv_offset_array_transform<statically_indexed_array<index_t, NRepeat>,
                                  decltype(k_coord),
                                  0,
                                  kPageBlockSize,
                                  0,
                                  NRepeat,
                                  kN0 / NRepeat,
                                  kKVMemoryLayout,
                                  true,
                                  kN0,
                                  kVectorSize>(
            k_physical_pages, stride_k, page_stride_k, k_coord, k_offsets, current_seq_k);

        auto k_dram_window =
            make_tile_scatter_gather(k_dram_block_window_tmp.get_bottom_tensor_view(),
                                     k_dram_block_window_tmp.get_window_lengths(),
                                     {seqlen_k_start, 0},
                                     k_dist,
                                     k_offsets);

        // =====================================================================
        // Scatter-gather V DRAM window setup
        // =====================================================================
        auto v_dist       = Policy::template MakeVDramTileDistribution<Problem>();
        auto v_coord      = v_dist.calculate_index();
        using VDstrEncode = typename decltype(v_dist)::DstrEncode;

        // V tensor K-dimension decomposition for page index computation.
        // In V3's distribution, the sequence (K) dimension is Hs index 0 (hs_lengthss_[0]),
        // and head_dim (N) is Hs index 1. This differs from the batch prefill pipeline where
        // sequence is Hs index 1. The Ps+Hs → rhs mapping: rhs[0]=Ps, rhs[1]=Hs[0], rhs[2]=Hs[1].
        //
        // V3 distribution for dim 0 (seq): {KPerThread, NumWarps, KThreadPerWarp} (bf16)
        //                              or: {NumIssues, LaneGroups, NumWarps}       (fp8)
        // The FIRST element is the per-thread Y iteration count; the rest are Ps (partitions).
        constexpr index_t V_KIterInner = VDstrEncode::hs_lengthss_[number<0>{}][number<0>{}];

        constexpr index_t V_KIterOuter = 1;

        constexpr index_t V_KLanes = VDstrEncode::hs_lengthss_[number<0>{}][number<0>{}];

        constexpr index_t V_PageIdxRepeat = V_KIterInner * V_KIterOuter;

        constexpr auto VPageIndexYDims = []() {
            // rhs[1] = first Hs dim (dim 0) = sequence dimension in V3's distribution.
            // Minor index 0 is the per-thread iteration (Y dim); the rest are Ps partitions.
            constexpr index_t Y_K1 = VDstrEncode::detail::rhs_major_minor_to_ys_[1][number<0>{}];
            return sequence<Y_K1>{};
        }();

        static_assert(decltype(VPageIndexYDims)::at(0) < VDstrEncode::NDimY,
                      "V page-index Y dim must be valid");

        statically_indexed_array<index_t, V_PageIdxRepeat> v_offsets;
        statically_indexed_array<index_t, V_PageIdxRepeat> v_physical_pages{};

        // Prefetch V physical pages helper
        // kCoordAxis=0 because V3 distribution has sequence as first Hs dim (axis 0)
        auto prefetch_v_physical_pages = [&](auto k_loop_start) {
            constexpr index_t kLoopStart = decltype(k_loop_start)::value;
            load_physical_pages<statically_indexed_array<index_t, V_KIterInner>,
                                decltype(v_coord),
                                0,
                                kPageBlockSize,
                                kLoopStart,
                                V_KIterInner,
                                1,
                                kKVMemoryLayout,
                                false,
                                kN0>(
                page_idx, v_coord, current_seq_k, v_physical_pages, max_page_table_idx);
        };

        // Update V offsets using pre-loaded physical pages
        auto update_v_offsets = [&](auto k_loop_start) {
            constexpr index_t kLoopStart = decltype(k_loop_start)::value;
            kv_offset_array_transform<statically_indexed_array<index_t, V_KIterInner>,
                                      decltype(v_coord),
                                      0,
                                      kPageBlockSize,
                                      kLoopStart,
                                      V_KIterInner,
                                      1,
                                      kKVMemoryLayout,
                                      false,
                                      kN0,
                                      kVectorSize>(
                v_physical_pages, stride_v, page_stride_v, v_coord, v_offsets, current_seq_k);
        };

        // Initial V offset computation
        prefetch_v_physical_pages(number<0>{});
        update_v_offsets(number<0>{});

        auto v_dram_window =
            make_tile_scatter_gather(v_dram_block_window_tmp.get_bottom_tensor_view(),
                                     v_dram_block_window_tmp.get_window_lengths(),
                                     {seqlen_k_start, 0},
                                     v_dist,
                                     v_offsets,
                                     number<0>{}, // HsGatherDim: sequence is first Hs dim in V3
                                     number<1>{}, // NumCoord
                                     VPageIndexYDims);

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

        index_t current_k_seq = seqlen_k_start;
        index_t current_v_seq = seqlen_k_start;

        // =====================================================================
        // K/V mem load lambdas (load-only, no page offset update)
        // =====================================================================
        auto K_mem_load = [&](auto k_lds_write_idx) {
            async_load_tile(k_lds_window_store(k_lds_write_idx),
                            k_dram_window,
                            number<-1>{},
                            bool_constant<false>{});
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
        };

        // Page offset advance lambdas — split into issue/consume pairs so
        // the global_load_dword (page table lookup) can be issued BEFORE the
        // buffer_loads from cl_load, placing it oldest in the vmcnt FIFO.
        // At consume time, s_waitcnt(N) drains only the oldest global_load
        // while keeping the N buffer_loads in flight.
        //
        // Page table index clamping happens inside load_physical_pages() via
        // max_page_table_idx, so the counters can advance freely past seqlen_k_end.
        // Past-end lookups return a valid (but stale) page; the loaded data is
        // discarded by the loop exit.

        // Issue: fire global_load_dword for next iteration's page table (oldest in vmcnt FIFO)
        auto K_page_issue = [&]() {
            current_k_seq += kN0;
            load_physical_pages<statically_indexed_array<index_t, NRepeat>,
                                decltype(k_coord),
                                0,
                                kPageBlockSize,
                                0,
                                NRepeat,
                                kN0 / NRepeat,
                                kKVMemoryLayout,
                                true,
                                kN0>(
                page_idx, k_coord, current_k_seq, k_physical_pages, max_page_table_idx);
        };

        // Consume: use result to compute offsets (needs vmcnt to drain global_load_dword)
        auto K_page_consume = [&]() {
            // Wait for global_load_dword (oldest) to complete.
            // K_mem_su_ld_insts buffer_loads from cl_load(memK) remain in flight.
            s_waitcnt<K_mem_su_ld_insts>();
            kv_offset_array_transform<statically_indexed_array<index_t, NRepeat>,
                                      decltype(k_coord),
                                      0,
                                      kPageBlockSize,
                                      0,
                                      NRepeat,
                                      kN0 / NRepeat,
                                      kKVMemoryLayout,
                                      true,
                                      kN0,
                                      kVectorSize>(
                k_physical_pages, stride_k, page_stride_k, k_coord, k_offsets, current_k_seq);
            k_dram_window.update_page_idx(k_offsets);
        };

        auto V_page_issue = [&]() {
            current_v_seq += kN0;
            current_seq_k = current_v_seq; // sync for prefetch_v_physical_pages
            prefetch_v_physical_pages(number<0>{});
        };

        auto V_page_consume = [&]() {
            // Wait for global_load_dword (oldest) to complete.
            // V_mem_su_ld_insts buffer_loads from cl_load(memV) remain in flight.
            s_waitcnt<V_mem_su_ld_insts>();
            update_v_offsets(number<0>{});
            v_dram_window.update_page_idx(v_offsets);
        };

        auto V_lds_load = [&](auto v_lds_read_idx) {
            kv_tile.v_tile = load_tile_transpose_with_offset(v_lds_window_load(v_lds_read_idx),
                                                             v_lds_load_offset);
        };

        decltype(m) m_old;
        SMPLComputeDataType o_acc_scale; // rescale o_acc in fmha_alu1() & fmha_alu_D_upd()
        statically_indexed_array<decltype(sp(number<0>{}).sp_compute), 2> sp_delta;

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
            int32x2_t swapped_regs =
                __builtin_amdgcn_permlane32_swap(bit_cast<int32_t>(m_latest.thread_buf_[0]),
                                                 bit_cast<int32_t>(m_latest.thread_buf_[0]),
                                                 false,
                                                 false);
            m_latest.thread_buf_[0] = f_max(bit_cast<SMPLComputeDataType>(swapped_regs.x),
                                            bit_cast<SMPLComputeDataType>(swapped_regs.y));
#else
            block_tile_reduce_sync(m_latest, f_max, bool_constant<false>{});
#endif
            m = m_latest;

            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    if constexpr(kHasLogitsSoftCap)
                    {
                        sp_delta(sp_reg_idx)(i_j_idx) =
                            sp(sp_reg_idx).sp_compute(i_j_idx) - m(i_j_idx);
                    }
                    else
                    {
                        sp_delta(sp_reg_idx)(i_j_idx) = detail::fma_impl_vsv(
                            sp(sp_reg_idx).sp_compute(i_j_idx), scale_s, -scale_s * m(i_j_idx));
                    }
                });
            });
        };

        auto fmha_alu1 = [&](auto sp_reg_idx) {
            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    sp(sp_reg_idx).sp_compute(i_j_idx) =
                        ck_tile::exp2(sp_delta(sp_reg_idx)(i_j_idx));
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
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp       = [&] {
                    if constexpr(kHasLogitsSoftCap)
                    {
                        return ck_tile::exp2(m_old[i_idx] - m[i_idx]);
                    }
                    else
                    {
                        return ck_tile::exp2(scale_s * (m_old[i_idx] - m[i_idx]));
                    }
                }();
                l(i_idx) = detail::add_impl_vv(tmp * l[i_idx], rowsum_p[i_idx]);
            });

            // update partial o_acc [0, fmha_alu_D_reg_cnt)
            static_for<0, fmha_alu_D_reg_cnt, 1>{}([&](auto idx) {
                o_acc.thread_buf_[idx] = detail::mul_impl_vv(o_acc.thread_buf_[idx], o_acc_scale);
            });

            // P conversion with inline asm anchoring
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

        constexpr index_t num_unpack_insts =
            (kHasLogitsSoftCap ? 48 : (std::is_same_v<KDataType, fp8_t> ? 36 : 26));
        fp32x2_t pk_o_acc_scale;
        auto fmha_alu_D_upd_unpack = [&] {
            o_acc_scale = [&] {
                if constexpr(kHasLogitsSoftCap)
                {
                    return ck_tile::exp2(m_old.thread_buf_[0] - m.thread_buf_[0]);
                }
                else
                {
                    return ck_tile::exp2(scale_s * (m_old.thread_buf_[0] - m.thread_buf_[0]));
                }
            }();

            static_assert(num_unpack_insts % 2 == 0 &&
                          (fmha_alu_D_reg_cnt + num_unpack_insts) <= o_acc.thread_buf_.size());
            static_for<fmha_alu_D_reg_cnt, fmha_alu_D_reg_cnt + num_unpack_insts, 1>{}(
                [&](auto idx) { o_acc.thread_buf_[idx] *= o_acc_scale; });
            pk_o_acc_scale.x = o_acc_scale;
            pk_o_acc_scale.y = o_acc_scale;
        };

        auto fmha_alu_D_upd_pack = [&] {
            constexpr index_t issued_unpack_insts = fmha_alu_D_reg_cnt + num_unpack_insts;
            static_for<issued_unpack_insts, o_acc.thread_buf_.size(), 2>{}([&](auto idx) {
                fp32x2_t input;
                input.x = o_acc.thread_buf_[idx];
                input.y = o_acc.thread_buf_[idx + 1];

                auto output = detail::pk_mul_f32(input, pk_o_acc_scale);

                o_acc.thread_buf_[idx]     = output.x;
                o_acc.thread_buf_[idx + 1] = output.y;
            });
        };

        auto fmha_alu_D_upd = [&] {
            fmha_alu_D_upd_unpack();
            fmha_alu_D_upd_pack();
        };

        auto fmha_mask = [&](auto sp_reg_idx) {
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(
                    q_origin.at(number<0>{}), kv_token_start, number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        sp(sp_reg_idx).sp_compute,
                        -numeric<SMPLComputeDataType>::infinity(),
                        [&](auto tile_idx) {
                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
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

        auto core_loop = [&](auto cl_p) {
            auto gemm0 = number<0>{};
            auto gemm1 = number<1>{};

            auto memV = number<0>{};
            auto memK = number<1>{};

            using Scheduler = CoreLoopScheduler<Problem>;

            auto iteration = [&](auto pi) {
                auto xdl_SP_p01_reg_idx = number<1>{} - pi;
                auto xdl_SP_p23_reg_idx = pi;

                auto K_w0_lds_wr_idx = number<1>{} - pi;
                auto V_w0_lds_wr_idx = pi;
                auto K_w0_lds_rd_idx = pi;
                auto V_w0_lds_rd_idx = pi;

                auto K_w4_lds_wr_idx = number<1>{} - pi;
                auto V_w4_lds_wr_idx = number<1>{} - pi;
                auto K_w4_lds_rd_idx = number<1>{} - pi;
                auto V_w4_lds_rd_idx = pi;

                bool result = true;

                if constexpr(cl_p == 0)
                {
                    __builtin_amdgcn_sched_barrier(0);
                    // phase0
                    if constexpr(pi == 0)
                    {
                        ASM_MARKER("phase0 Wave0-3 (pi=0)");
                    }
                    else
                    {
                        ASM_MARKER("phase0 Wave0-3 (pi=1)");
                    }
                    s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                    __builtin_amdgcn_sched_barrier(0);
#if ADD_SBARRIER_FOR_PHASE0
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
#endif
                    if constexpr(pi == 1)
                    {
                        if constexpr(!std::is_same_v<KDataType, fp8_t>)
                        {
                            asm volatile("s_nop 1");
                            __builtin_amdgcn_sched_barrier(0);
                        }
                    }
                    cl_calc(xdl_SP_p01_reg_idx, gemm0);
                    fmha_alu1(xdl_SP_p23_reg_idx);
                    fmha_logits_trans(xdl_SP_p01_reg_idx);

                    Scheduler::schedule(cl_p, number<0>{});
                    __builtin_amdgcn_sched_barrier(0);
                    // phase1
                    ASM_MARKER("phase1 Wave0-3");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    K_page_issue();                                  // global_load_dword FIRST
                    __builtin_amdgcn_sched_barrier(0);               // prevent reorder
                    cl_load(memK, K_w0_lds_wr_idx, V_w0_lds_rd_idx); // buffer_loads SECOND
                    Scheduler::schedule(cl_p, number<1>{});
                    fmha_mask(xdl_SP_p01_reg_idx);
                    __builtin_amdgcn_sched_barrier(0); // prevent reorder
                    K_page_consume();                  // vmcnt(K_mem_su_ld_insts)

                    __builtin_amdgcn_sched_barrier(0);
                    // phase2
                    ASM_MARKER("phase2 Wave0-3");
                    s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_nop 1");
                    __builtin_amdgcn_sched_barrier(0);
                    cl_calc(xdl_SP_p23_reg_idx, gemm1);
                    fmha_alu_D_upd_unpack();
                    Scheduler::schedule(cl_p, number<2>{});
                    __builtin_amdgcn_sched_barrier(0);
                    fmha_alu_D_upd_pack();

                    __builtin_amdgcn_sched_barrier(0);
                    // phase3
                    ASM_MARKER("phase3 Wave0-3");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    V_page_issue();                                  // global_load_dword FIRST
                    __builtin_amdgcn_sched_barrier(0);               // prevent reorder
                    cl_load(memV, V_w0_lds_wr_idx, K_w0_lds_rd_idx); // buffer_loads SECOND
                    Scheduler::schedule(cl_p, number<3>{});
                    // Page offset update at loop increment
                    kv_token_start += kN0;
                    if(num_total_loop <= ++i_total_loops)
                    {
                        result = false;
                    }
                    __builtin_amdgcn_sched_barrier(0); // prevent reorder
                    V_page_consume();                  // vmcnt(V_mem_su_ld_insts)
                }
                else
                {
#if ADD_SBARRIER_FOR_PHASE0
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
#endif
                    __builtin_amdgcn_sched_barrier(0);
                    // phase0
                    if constexpr(pi == 0)
                    {
                        ASM_MARKER("phase0 Wave4-7 (pi=0)");
                    }
                    else
                    {
                        ASM_MARKER("phase0 Wave4-7 (pi=1)");
                    }
                    V_page_issue();                                  // global_load_dword FIRST
                    __builtin_amdgcn_sched_barrier(0);               // prevent reorder
                    cl_load(memV, V_w4_lds_wr_idx, K_w4_lds_rd_idx); // buffer_loads SECOND
                    Scheduler::schedule(cl_p, number<0>{});
                    __builtin_amdgcn_sched_barrier(0); // prevent reorder
                    V_page_consume();                  // vmcnt(V_mem_su_ld_insts)
                    __builtin_amdgcn_sched_barrier(0);
                    // phase1
                    ASM_MARKER("phase1 Wave4-7");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts, waitcnt_arg::kMaxExpCnt, 0>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);

                    if constexpr(!std::is_same_v<KDataType, fp8_t>)
                    {
                        asm volatile("s_nop 1");
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    cl_calc(xdl_SP_p01_reg_idx, gemm0);
                    fmha_alu1(xdl_SP_p23_reg_idx);
                    fmha_logits_trans(xdl_SP_p01_reg_idx);

                    Scheduler::schedule(cl_p, number<1>{});
                    __builtin_amdgcn_sched_barrier(0);
                    // phase2
                    ASM_MARKER("phase2 Wave4-7");
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    K_page_issue();                                  // global_load_dword FIRST
                    __builtin_amdgcn_sched_barrier(0);               // prevent reorder
                    cl_load(memK, K_w4_lds_wr_idx, V_w4_lds_rd_idx); // buffer_loads SECOND
                    Scheduler::schedule(cl_p, number<2>{});
                    fmha_mask(xdl_SP_p01_reg_idx);
                    __builtin_amdgcn_sched_barrier(0); // prevent reorder
                    K_page_consume();                  // vmcnt(K_mem_su_ld_insts)

                    // Page offset update at loop increment
                    kv_token_start += kN0;
                    if(num_total_loop <= ++i_total_loops)
                    {
                        result = false;
                    }

                    __builtin_amdgcn_sched_barrier(0);
                    // phase3
                    ASM_MARKER("phase3 Wave4-7");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts, waitcnt_arg::kMaxExpCnt, 0>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    if constexpr(!std::is_same_v<KDataType, fp8_t>)
                    {
                        asm volatile("s_nop 1");
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    cl_calc(xdl_SP_p23_reg_idx, gemm1);
                    fmha_alu_D_upd_unpack();
                    Scheduler::schedule(cl_p, number<3>{});
                    __builtin_amdgcn_sched_barrier(0);
                    fmha_alu_D_upd_pack();
                }
                return result;
            };
            return iteration(number<0>{}) && iteration(number<1>{});
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
                K_page_issue();          // global_load for K1 offset (FIRST)
                K_mem_load(number<0>{}); // buffer_load K0 (SECOND)
                K_page_consume();        // s_waitcnt vmcnt(K_mem_su_ld_insts), transform

                s_waitcnt<0>();
                __builtin_amdgcn_s_barrier();

                K_lds_load(number<0>{}); // lds_K0

                s_waitcnt<waitcnt_arg::kMaxVmCnt, waitcnt_arg::kMaxExpCnt, 0>();
                __builtin_amdgcn_s_barrier();

                // (2) prefetch K1 and V0 to LDS in parallel with GEMM0
                V_page_issue();          // global_load for V1 offset
                V_mem_load(number<0>{}); // buffer_load V0
                V_page_consume();        // s_waitcnt vmcnt(V_mem_su_ld_insts), transform
                if(1 < num_total_loop)
                {
                    K_page_issue();          // global_load for K2 offset
                    K_mem_load(number<1>{}); // buffer_load K1
                    K_page_consume();        // s_waitcnt vmcnt(K_mem_su_ld_insts), transform
                }

                // (3) mfma (Q*K0) + softmax
                gemm(number<0>{}, /*gemm_idx=*/number<0>{});
                fmha_logits_trans(number<0>{});
                fmha_mask(number<0>{});
                fmha_alu0(number<0>{});
                fmha_alu_D_upd();

                kv_token_start += kN0;
                ++i_total_loops;
                if(num_total_loop <= i_total_loops)
                {
                    goto label_main_loops_exit;
                }

                if(2 < num_total_loop)
                {
                    // K2 at seq=start+2*kN0 (k offsets already point here)
                    K_page_issue();          // global_load for K3 offset
                    K_mem_load(number<0>{}); // buffer_load K2
                    K_page_consume();        // s_waitcnt vmcnt(K_mem_su_ld_insts), transform

                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_s_barrier();
                }

                ASM_MARKER("end pre-stage");
            }

            if(1 < num_total_loop)
            {
                // V offsets point to start+kN0 (advanced after V0 load)
                if(warp_group_id == 0)
                {
                    V_page_issue();          // global_load for V2 offset
                    V_mem_load(number<1>{}); // buffer_load V1
                    V_page_consume();        // s_waitcnt vmcnt(V_mem_su_ld_insts), transform
                    K_lds_load(number<1>{}); // K1

                    __builtin_amdgcn_s_setprio(0);
                    __builtin_amdgcn_s_barrier();
                    while(core_loop(number<0>{}))
                        ;
                }
                if(warp_group_id != 0)
                {
                    __builtin_amdgcn_s_setprio(1);
                    __builtin_amdgcn_s_barrier();
                    while(core_loop(number<1>{}))
                        ;
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
               void* __restrict__ smem_ptr,
               // Paged KV cache parameters
               const index_t* page_idx,
               index_t stride_k,
               index_t stride_v,
               index_t page_stride_k,
               index_t page_stride_v,
               index_t max_page_table_idx = 0x7FFFFFFF) const
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
                          smem_ptr,
                          page_idx,
                          stride_k,
                          stride_v,
                          page_stride_k,
                          page_stride_v,
                          max_page_table_idx);
    }
};

} // namespace ck_tile
