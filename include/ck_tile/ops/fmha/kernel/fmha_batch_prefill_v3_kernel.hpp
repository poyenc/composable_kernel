// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_kvcache_layout_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

namespace ck_tile {

/// V3 kernel adapted for batch prefill with paged KV cache.
///
/// Combines FmhaFwdV3Kernel's execution structure (separate smem_k[2]/smem_v[2],
/// partition_index, V3 epilogue) with FmhaBatchPrefillWithPagedKVCacheKernel's
/// paged KV DRAM views (page table resolution, LINEAR/VECTORIZED layout transforms).
///
/// Key differences from FmhaFwdV3Kernel:
///   - K/V DRAM views are paged (transform_tensor_view with page table)
///   - Kargs include page table fields (SGLang/vLLM)
///   - Pipeline receives additional page_idx, stride_k/v, page_stride_k/v params
///   - V DRAM view uses V3 convention: (kK1, kN1) = (sequence, head_dim)
///
/// V3 does NOT support: bias, dropout, randval. MakeKargs omits these.
template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaBatchPrefillV3Kernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using PDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::PDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    static constexpr bool kIsGroupMode      = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = FmhaPipeline::kHasLogitsSoftCap;
    static constexpr bool kStoreLSE         = FmhaPipeline::kStoreLSE;
    static constexpr auto QScaleEnum        = FmhaPipeline::Problem::QScaleEnum;

    // Paged KV cache parameters from pipeline Problem
    static constexpr auto kKVMemoryLayout = FmhaPipeline::kKVMemoryLayout;
    static_assert(kKVMemoryLayout == BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT,
                  "V3 batch prefill only supports LINEAR_LAYOUT");
    static_assert(QScaleEnum != BlockAttentionQuantScaleEnum::KV_BLOCKSCALE,
                  "V3 batch prefill does not support KV_BLOCKSCALE quantization");
    static constexpr auto kKVLookupTable    = FmhaPipeline::Problem::kKVLookupTable;
    static constexpr index_t kPageBlockSize = FmhaPipeline::kPageBlockSize;
    static constexpr index_t kVectorSize    = FmhaPipeline::kVectorSize;

    using AttentionVariant = ck_tile::remove_cvref_t<typename FmhaPipeline::AttentionVariant>;
    using FmhaMask         = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    template <ck_tile::index_t I>
    struct FmhaFwdEmptyKargs
    {
    };

    // Page table kargs — same as FmhaBatchPrefillWithPagedKVCacheKernel
    struct SglangPageTableKargs
    {
        const int32_t* kv_indptr;
        const int32_t* kv_page_indices;
        const int32_t* kv_last_page_lens;
    };

    struct VllmPageTableKargs
    {
        const int32_t* block_table_ptr;
        ck_tile::index_t batch_stride_block_table;
        const int32_t* seqlen_k_ptr;
    };

    using PageBlockTableKargs =
        std::conditional_t<kKVLookupTable ==
                               BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D,
                           SglangPageTableKargs,
                           VllmPageTableKargs>;

    // Kargs
    struct FmhaFwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        void* o_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        ck_tile::index_t nhead_ratio_qk;

        int32_t num_total_pages;
        ck_tile::index_t page_block_size;
        PageBlockTableKargs page_table;

        float scale_s;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_o;
    };

    struct FmhaFwdLogitsSoftCapKargs
    {
        FmhaFwdLogitsSoftCapKargs() = default;

        void init_logits_soft_cap(float logits_soft_cap_)
        {
            if(0 < logits_soft_cap_)
            {
                logits_soft_cap     = logits_soft_cap_;
                logits_soft_cap_rcp = 1.f / logits_soft_cap;
            }
            else
            {
                logits_soft_cap     = 0.f;
                logits_soft_cap_rcp = 0.f;
            }
        }

        float logits_soft_cap;
        float logits_soft_cap_rcp;
    };

    struct FmhaFwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct FmhaFwdCommonLSEKargs
    {
        void* lse_ptr                     = nullptr;
        ck_tile::index_t nhead_stride_lse = 0;
        ck_tile::index_t batch_stride_lse = 0;
    };

    struct FmhaFwdCommonQScaleKargs
    {
        const void* q_descale_ptr = nullptr;
        const void* k_descale_ptr = nullptr;
        const void* v_descale_ptr = nullptr;
    };

    struct FmhaFwdBatchModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<0>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<QScaleEnum == ck_tile::BlockAttentionQuantScaleEnum::PERTENSOR,
                             FmhaFwdCommonQScaleKargs,
                             FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<3>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;
    };

    struct FmhaFwdGroupModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<0>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<1>>,
          std::conditional_t<QScaleEnum == ck_tile::BlockAttentionQuantScaleEnum::PERTENSOR,
                             FmhaFwdCommonQScaleKargs,
                             FmhaFwdEmptyKargs<2>>,
          std::conditional_t<kHasLogitsSoftCap, FmhaFwdLogitsSoftCapKargs, FmhaFwdEmptyKargs<3>>
    {
        const int32_t* seqstart_q_ptr;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
    };

    using Kargs = std::conditional_t<kIsGroupMode, FmhaFwdGroupModeKargs, FmhaFwdBatchModeKargs>;

    struct BlockIndices
    {
        ck_tile::index_t batch_idx;
        ck_tile::index_t qo_head_idx;
        ck_tile::index_t kv_head_idx;
    };

    // Batch mode MakeKargs
    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* q_descale_ptr,
              const void* k_descale_ptr,
              const void* v_descale_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              int32_t num_total_pages,
              ck_tile::index_t page_block_size,
              const PageBlockTableKargs& page_table,
              float scale_s,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_q,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t batch_stride_lse,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     seqlen_q,
                     -1,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     num_total_pages,
                     page_block_size,
                     page_table,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for qscale
                    {},               // placeholder for logits_soft_cap
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
            kargs.batch_stride_lse = batch_stride_lse;
        }
        if constexpr(QScaleEnum == ck_tile::BlockAttentionQuantScaleEnum::PERTENSOR)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }

        return kargs;
    }

    // Group mode MakeKargs
    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              const void* q_descale_ptr,
              const void* k_descale_ptr,
              const void* v_descale_ptr,
              void* lse_ptr,
              void* o_ptr,
              const void* seqstart_q_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              int32_t num_total_pages,
              ck_tile::index_t page_block_size,
              const PageBlockTableKargs& page_table,
              float scale_s,
              float logits_soft_cap,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_k,
              ck_tile::index_t batch_stride_v,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     num_total_pages,
                     page_block_size,
                     page_table,
#if CK_TILE_FMHA_FWD_FAST_EXP2
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
#else
                     scale_s,
#endif
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for mask
                    {},               // placeholder for lse
                    {},               // placeholder for qscale
                    {},               // placeholder for logits_soft_cap
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    batch_stride_k,
                    batch_stride_v};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
        }
        if constexpr(QScaleEnum == ck_tile::BlockAttentionQuantScaleEnum::PERTENSOR)
        {
            kargs.q_descale_ptr = q_descale_ptr;
            kargs.k_descale_ptr = k_descale_ptr;
            kargs.v_descale_ptr = v_descale_ptr;
        }
        if constexpr(kHasLogitsSoftCap)
        {
            kargs.init_logits_soft_cap(logits_soft_cap);
        }

        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_)
    {
        if constexpr(kIsGroupMode)
        {
            return dim3(nhead_,
                        batch_size_,
                        ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1));
        }
        else
        {
            return dim3(ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1),
                        nhead_,
                        batch_size_);
        }
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs)
    {
        if constexpr(kIsGroupMode)
        {
            const index_t num_tile_n1 =
                ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1);

            const index_t i_block = blockIdx.z;
            const index_t i_nhead = blockIdx.x;
            const index_t i_batch = blockIdx.y;

            const auto f = [](index_t dividend, index_t divisor) {
                index_t quotient = dividend / divisor;
                index_t modulus  = dividend - quotient * divisor;
                return ck_tile::make_tuple(quotient, modulus);
            };

            const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);
            if constexpr(kHasMask)
            {
                return ck_tile::make_tuple(gridDim.z - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
            }
            else
            {
                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
            }
        }
        else
        {
            const index_t num_tile_n1 =
                ck_tile::integer_divide_ceil(kargs.hdim_v, FmhaPipeline::kN1);

            const index_t i_block = blockIdx.x;
            const index_t i_nhead = blockIdx.y;
            const index_t i_batch = blockIdx.z;

            const auto f = [](index_t dividend, index_t divisor) {
                index_t quotient = dividend / divisor;
                index_t modulus  = dividend - quotient * divisor;
                return ck_tile::make_tuple(quotient, modulus);
            };

            const auto [i_tile_m, i_tile_n] = f(i_block, num_tile_n1);

            if constexpr(kHasMask)
            {
                return ck_tile::make_tuple(gridDim.x - 1 - i_tile_m, i_tile_n, i_nhead, i_batch);
            }
            else
            {
                return ck_tile::make_tuple(i_tile_m, i_tile_n, i_nhead, i_batch);
            }
        }
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using namespace ck_tile;

        // divide problem
        const auto [i_tile_m, i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);

        const index_t i_m0 = amd_wave_read_first_lane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = amd_wave_read_first_lane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q   = 0;
        long_index_t batch_offset_lse = 0;
        long_index_t batch_offset_o   = 0;

        // Resolve seqlen_k from page table
        const index_t seqlen_k = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
            {
                const int32_t page_start      = kargs.page_table.kv_indptr[i_batch];
                const int32_t page_end        = kargs.page_table.kv_indptr[i_batch + 1];
                const int32_t num_page_blocks = page_end - page_start;
                const int32_t last_page_len   = [&]() {
                    if constexpr(kPageBlockSize == 1)
                        return static_cast<int32_t>(kPageBlockSize);
                    else
                        return kargs.page_table.kv_last_page_lens[i_batch];
                }();
                return num_page_blocks > 0
                           ? static_cast<index_t>((num_page_blocks - 1) * kargs.page_block_size +
                                                  last_page_len)
                           : 0;
            }
            else // VLLM_BLOCK_TABLE_2D
            {
                if(kargs.page_table.seqlen_k_ptr != nullptr)
                    return static_cast<index_t>(kargs.page_table.seqlen_k_ptr[i_batch]);
                else
                    return kargs.seqlen_k;
            }
        }();

        // Resolve page_idx pointer for this batch
        const int32_t* page_idx = [&]() {
            if constexpr(kKVLookupTable ==
                         BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D)
            {
                return kargs.page_table.kv_page_indices + kargs.page_table.kv_indptr[i_batch];
            }
            else // VLLM_BLOCK_TABLE_2D
            {
                return kargs.page_table.block_table_ptr +
                       static_cast<long_index_t>(i_batch) *
                           kargs.page_table.batch_stride_block_table;
            }
        }();

        if constexpr(kIsGroupMode)
        {
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;

            if constexpr(kStoreLSE)
            {
                batch_offset_lse = query_start;
            }
            batch_offset_o = query_start * kargs.stride_o;

            kargs.seqlen_q = kargs.seqstart_q_ptr[i_batch + 1] - query_start;

            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }

            kargs.seqlen_k = seqlen_k;
        }
        else
        {
            batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            if constexpr(kStoreLSE)
            {
                batch_offset_lse = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
            }
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;

            kargs.seqlen_k = seqlen_k;
        }

        // Q pointer: per-batch + per-head offset
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        // K/V pointers: per-head offset only (paged layout, no per-batch offset)
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // Q DRAM view
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                number<FmhaPipeline::kAlignmentQ>{},
                number<1>{});

            return pad_tensor_view(
                q_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
                sequence<kPadSeqLenQ, kPadHeadDimQ>{});
        }();
        static_assert(FmhaPipeline::kN0 == 64 || FmhaPipeline::kN0 == 128,
                      "only kN0 == 64 or 128 is supported");
        static_assert(FmhaPipeline::kK1 == 64 || FmhaPipeline::kK1 == 128,
                      "only kK1 == 64 or 128 is supported");

        // K DRAM view (paged layout, LINEAR only)
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.num_total_pages, kargs.page_block_size, kargs.hdim_q),
                make_tuple(kargs.batch_stride_k, kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            auto k_dram_2d = transform_tensor_view(
                k_dram_naive,
                make_tuple(
                    make_merge_transform(make_tuple(kargs.num_total_pages, kargs.page_block_size)),
                    make_pass_through_transform(kargs.hdim_q)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return pad_tensor_view(
                k_dram_2d,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();

        // V DRAM view (paged layout, LINEAR only, V3 convention: (kK1, kN1) = (sequence, head_dim))
        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.num_total_pages, kargs.page_block_size, kargs.hdim_v),
                make_tuple(kargs.batch_stride_v, kargs.stride_v, 1),
                number<FmhaPipeline::kAlignmentV>{},
                number<1>{});

            auto v_dram_2d = transform_tensor_view(
                v_dram_naive,
                make_tuple(
                    make_merge_transform(make_tuple(kargs.num_total_pages, kargs.page_block_size)),
                    make_pass_through_transform(kargs.hdim_v)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return pad_tensor_view(
                v_dram_2d,
                make_tuple(number<FmhaPipeline::kK1>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenK, kPadHeadDimV>{});
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(number<FmhaPipeline::kK1>{}, number<FmhaPipeline::kN1>{}),
                             {0, i_n1});

        // lse
        auto lse_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto lse_dram_window_lengths = make_tuple(number<FmhaPipeline::kM0>{});
            if constexpr(kStoreLSE)
            {
                LSEDataType* lse_ptr =
                    reinterpret_cast<LSEDataType*>(kargs.lse_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_lse + batch_offset_lse;

                const auto lse_dram = [&]() {
                    const auto lse_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        lse_ptr,
                        make_tuple(kargs.seqlen_q),
                        make_tuple(1),
                        number<1>{},
                        number<1>{});

                    return pad_tensor_view(
                        lse_dram_naive, lse_dram_window_lengths, sequence<kPadSeqLenQ>{});
                }();

                return make_tile_window(lse_dram, lse_dram_window_lengths, {i_m0});
            }
            else
            {
                return make_null_tile_window(lse_dram_window_lengths);
            }
        }();

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        // V3-style separate LDS allocations for K and V double buffers
        __shared__ char
            smem_k[2]
                  [FmhaPipeline::Policy::template GetSmemSizeK<typename FmhaPipeline::Problem>()];
        __shared__ char
            smem_v[2]
                  [FmhaPipeline::Policy::template GetSmemSizeV<typename FmhaPipeline::Problem>()];
        constexpr auto smem_epilogue_size = max(1, EpiloguePipeline::GetSmemSize());
        __shared__ char smem_epilogue_buf[smem_epilogue_size];

        auto* smem_k0  = reinterpret_cast<KDataType*>(smem_k[0]);
        auto* smem_k1  = reinterpret_cast<KDataType*>(smem_k[1]);
        auto* smem_v0  = reinterpret_cast<VDataType*>(smem_v[0]);
        auto* smem_v1  = reinterpret_cast<VDataType*>(smem_v[1]);
        void* smem_ptr = smem_epilogue_buf;

        const auto partition_index = multi_index<2>{get_warp_id(), get_lane_id()};

        AttentionVariant variant;

        const float scale_s = [&] {
            if constexpr(QScaleEnum == ck_tile::BlockAttentionQuantScaleEnum::PERTENSOR)
            {
                float q_descale = *(reinterpret_cast<const float*>(kargs.q_descale_ptr));
                float k_descale = *(reinterpret_cast<const float*>(kargs.k_descale_ptr));
                return kargs.scale_s * q_descale * k_descale;
            }
            else
            {
                return kargs.scale_s;
            }
        }();

        const auto variant_params = [&] {
            if constexpr(kHasLogitsSoftCap)
            {
                return ck_tile::LogitsSoftCapParams<FmhaMask, CK_TILE_FMHA_FWD_FAST_EXP2>{
                    mask, scale_s, kargs.logits_soft_cap, kargs.logits_soft_cap_rcp};
            }
            else
            {
                return ck_tile::StandardAttentionParams<FmhaMask>{mask, scale_s};
            }
        }();

        BlockIndices block_indices{i_batch, i_nhead, i_nhead / kargs.nhead_ratio_qk};

        // Strides for the pipeline's scatter-gather offset computation (LINEAR only)
        const index_t stride_k_for_pipeline = kargs.stride_k;
        const index_t stride_v_for_pipeline = kargs.stride_v;

        // Max valid index into page_idx[] array for this batch entry.
        // For page_size=1: max_page_table_idx = seqlen_k - 1 (one entry per token)
        // For page_size>1: max_page_table_idx = (seqlen_k - 1) / page_block_size
        // Used by load_physical_pages() to clamp past-end lookups to valid entries.
        const index_t max_page_table_idx =
            kargs.seqlen_k > 0 ? (kargs.seqlen_k - 1) / kPageBlockSize : 0;

        auto o_acc_tile = [&] {
            if constexpr(QScaleEnum == ck_tile::BlockAttentionQuantScaleEnum::PERTENSOR)
            {
                float v_descale = *(reinterpret_cast<const float*>(kargs.v_descale_ptr));
                float scale_p   = ck_tile::type_convert<float>(ck_tile::numeric<PDataType>::max());
                float scale_o   = v_descale / scale_p;

                auto o_acc_element_func = [&]() {
                    if constexpr(std::is_same_v<ODataType, ck_tile::fp8_t>)
                        return make_composes(
                            ck_tile::saturates<ck_tile::fp8_t>{},
                            ck_tile::scales<ck_tile::remove_cvref_t<decltype(scale_o)>>{scale_o});
                    else
                        return ck_tile::scales<ck_tile::remove_cvref_t<decltype(scale_o)>>{scale_o};
                }();

                return FmhaPipeline{}(partition_index,
                                      q_dram_window,
                                      identity{},
                                      k_dram_window,
                                      identity{},
                                      v_dram_window,
                                      identity{},
                                      lse_dram_window,
                                      identity{},
                                      identity{},
                                      scales<ck_tile::remove_cvref_t<decltype(scale_p)>>{scale_p},
                                      o_acc_element_func,
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
                                      stride_k_for_pipeline,
                                      stride_v_for_pipeline,
                                      kargs.batch_stride_k,
                                      kargs.batch_stride_v,
                                      max_page_table_idx);
            }
            else
            {
                return FmhaPipeline{}(partition_index,
                                      q_dram_window,
                                      k_dram_window,
                                      v_dram_window,
                                      lse_dram_window,
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
                                      stride_k_for_pipeline,
                                      stride_v_for_pipeline,
                                      kargs.batch_stride_k,
                                      kargs.batch_stride_v,
                                      max_page_table_idx);
            }
        }();

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<FmhaPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, partition_index);
    }
};
} // namespace ck_tile
