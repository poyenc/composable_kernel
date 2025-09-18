// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"

#include <type_traits>
#include <utility>

#if !defined(CK_TILE_REMAP_TOKEN_USING_DESC)
#define CK_TILE_REMAP_TOKEN_USING_DESC 1
#endif

namespace ck_tile {

template <typename T>
constexpr T swap_bit_positions(T x, unsigned p, unsigned q)
{
    static_assert(std::is_unsigned<T>::value, "T must be unsigned");

    T bitp = (x >> p) & T(1);
    T bitq = (x >> q) & T(1);

    T mask = ((T(1) << p) | (T(1) << q));
    x &= ~mask;

    x |= (bitp << q) | (bitq << p);
    return x;
}

template <typename T>
constexpr T swap_bit1_bit2(T x)
{
    return swap_bit_positions<T>(x, 1u, 2u);
}

template <typename T>
constexpr T permute_bits_5(T x)
{
    static_assert(std::is_unsigned<T>::value, "T must be unsigned");

    T b0 = (x >> 0) & 1u;
    T b1 = (x >> 1) & 1u;
    T b2 = (x >> 2) & 1u;
    T b3 = (x >> 3) & 1u;
    T b4 = (x >> 4) & 1u;

    T y = (b2 << 4) | (b1 << 3) | (b0 << 2) | (b4 << 1) | b3;

    return (x & ~T(31)) | y;
}

template <typename T>
constexpr T inverse_permute_bits_5(T x)
{
    static_assert(std::is_unsigned<T>::value, "T must be unsigned");

    T y  = x & 31u;
    T c4 = (y >> 4) & 1u;
    T c3 = (y >> 3) & 1u;
    T c2 = (y >> 2) & 1u;
    T c1 = (y >> 1) & 1u;
    T c0 = (y >> 0) & 1u;

    T orig = (c1 << 4) | (c0 << 3) | (c4 << 2) | (c3 << 1) | c2;
    return (x & ~T(31)) | orig;
}

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdV3Kernel
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
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    static constexpr bool kIsGroupMode = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = FmhaPipeline::kPadHeadDimV;
    static constexpr bool kStoreLSE    = FmhaPipeline::kStoreLSE;

    using FmhaMask                 = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct FmhaFwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
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
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_ratio_qk;
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

    struct FmhaFwdMaskKargs
    {
        // ck_tile::index_t window_size_left, window_size_right;
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
        ck_tile::index_t remap_opt;
    };

    struct FmhaFwdCommonLSEKargs
    {
        void* lse_ptr                     = nullptr;
        ck_tile::index_t nhead_stride_lse = 0;
        ck_tile::index_t batch_stride_lse = 0;
    };

    struct FmhaFwdBatchModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<0>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<1>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_o;
    };

    struct FmhaFwdGroupModeKargs
        : FmhaFwdCommonKargs,
          std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<0>>,
          std::conditional_t<kStoreLSE, FmhaFwdCommonLSEKargs, FmhaFwdEmptyKargs<1>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_k_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode, FmhaFwdGroupModeKargs, FmhaFwdBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
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
              ck_tile::index_t mask_type,
              ck_tile::index_t remap_opt)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     num_head_q,
                     nhead_ratio_qk,
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
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
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    batch_stride_o};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
            kargs.remap_opt         = remap_opt;
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
            kargs.batch_stride_lse = batch_stride_lse;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              void* lse_ptr,
              void* o_ptr,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_k_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t nhead_ratio_qk,
              float scale_s,
              ck_tile::index_t stride_q,
              ck_tile::index_t stride_k,
              ck_tile::index_t stride_v,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_q,
              ck_tile::index_t nhead_stride_k,
              ck_tile::index_t nhead_stride_v,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t window_size_left,
              ck_tile::index_t window_size_right,
              ck_tile::index_t mask_type,
              ck_tile::index_t remap_opt)
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
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
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
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr)};

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
            kargs.remap_opt         = remap_opt;
        }
        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
        }

        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                                ck_tile::index_t nhead_,
                                                ck_tile::index_t seqlen_q_,
                                                ck_tile::index_t hdim_v_)
    {
        // TODO: this may need tuning
        if constexpr(kHasMask)
        {
            return dim3(nhead_,
                        ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1),
                        batch_size_);
        }
        else
        {
            return dim3(nhead_,
                        ck_tile::integer_divide_ceil(seqlen_q_, FmhaPipeline::kM0) *
                            ck_tile::integer_divide_ceil(hdim_v_, FmhaPipeline::kN1),
                        batch_size_);
        }
    }

    CK_TILE_DEVICE static constexpr auto
    RemapTileIndices(int32_t tg_idx, int32_t tg_idy, int32_t remap_option)
    {
        if(remap_option < 1)
        {
            return make_tuple(static_cast<int32_t>(gridDim.x - tg_idx - 1), tg_idy);
        }

        int32_t remapped_tg_idx = tg_idx;
        int32_t remapped_tg_idy = tg_idy;

        if(remap_option == 2)
        { // special remapping
            int32_t tmp0 = (remapped_tg_idy & 0x7) * gridDim.x + remapped_tg_idx;
            int32_t tmp1 = tmp0 & 0x7;

            remapped_tg_idx = tmp0 >> 3;
            remapped_tg_idy = (remapped_tg_idy & 0xfffffff8) + tmp1;
        }
        else
        { // normal remapping
            int32_t cus_per_xdim_per_xcc = gridDim.x >> 3;
            int32_t tgs_cu_id            = remapped_tg_idx >> 3;

            if(tgs_cu_id < cus_per_xdim_per_xcc)
            {
                int32_t tgs_xcc_id = remapped_tg_idx & 0x7;
                int32_t new_tg_idx = tgs_xcc_id * cus_per_xdim_per_xcc + tgs_cu_id;

                remapped_tg_idx = new_tg_idx;
            }
        }

        return make_tuple(remapped_tg_idx, remapped_tg_idy);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs&)
    {
        using namespace ck_tile;

        // const index_t num_tile_n1 = ck_tile::integer_divide_ceil(kargs.hdim_v,
        // FmhaPipeline::kN1);

        // assume that num_tile_n1 is always 1
        if constexpr(kHasMask)
        {
            const index_t i_nhead = blockIdx.x;
            const index_t i_block = blockIdx.y;
            const index_t i_batch = blockIdx.z;

            return ck_tile::make_tuple(gridDim.y - 1 - i_block, 0, i_nhead, i_batch);
        }
        else
        {
            const index_t i_nhead = blockIdx.x;
            const index_t i_block = blockIdx.y;
            const index_t i_batch = blockIdx.z;

            return ck_tile::make_tuple(i_block, 0, i_nhead, i_batch);
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

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * FmhaPipeline::kM0);
        const index_t i_n1 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN1);

        long_index_t batch_offset_q   = 0;
        long_index_t batch_offset_k   = 0;
        long_index_t batch_offset_v   = 0;
        long_index_t batch_offset_lse = 0;
        long_index_t batch_offset_o   = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            batch_offset_q = query_start * kargs.stride_q;
            batch_offset_k = key_start * kargs.stride_k;
            batch_offset_v = key_start * kargs.stride_v;

            if constexpr(kStoreLSE)
            {
                batch_offset_lse = query_start;
            }
            batch_offset_o = query_start * kargs.stride_o;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }

            if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }
        }
        else
        {
            batch_offset_q = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            if constexpr(kStoreLSE)
            {
                batch_offset_lse = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
            }
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
        }

        // We add key offset to the base address for each thread here
        index_t thread_key_token_offset = 0;
        {
            constexpr index_t kVectorSize =
                FmhaPipeline::Policy::template GetAlignmentK<typename FmhaPipeline::Problem>();
            if constexpr(FmhaPipeline::kN0 == 32)
            {
                static_assert(FmhaPipeline::kN0 * FmhaPipeline::kK0 / kBlockSize == kVectorSize);
            }
            else if constexpr(FmhaPipeline::kN0 == 64)
            {
                static_assert(FmhaPipeline::kN0 * FmhaPipeline::kK0 / kBlockSize ==
                              2 * kVectorSize);
            }
            else
            {
                static_assert(false, "unsupported kN0");
            }

            constexpr index_t KThreadPerBlock = FmhaPipeline::kK0 / kVectorSize;

            // Make sure that a thread must read data from the same key token
            // and the calculation of old_key_offset should match the MakeKDramTileDistribution()
            index_t old_key_offset = get_thread_local_1d_id() / KThreadPerBlock;
            index_t new_key_offset =
                bit_cast<index_t>(permute_bits_5(bit_cast<uint32_t>(old_key_offset)));

#if !CK_TILE_REMAP_TOKEN_USING_DESC
            thread_key_token_offset = new_key_offset - old_key_offset;
            batch_offset_k += thread_key_token_offset * kargs.stride_k;
#endif
        }
        // We add value offset to the base address for each thread here
        index_t thread_value_token_offset = 0;
        {
            constexpr index_t kVectorSize =
                FmhaPipeline::Policy::template GetAlignmentV<typename FmhaPipeline::Problem>();
            if constexpr(FmhaPipeline::kK1 == 32)
            {
                static_assert(FmhaPipeline::kN1 * FmhaPipeline::kK1 / kBlockSize == kVectorSize);
            }
            else if constexpr(FmhaPipeline::kK1 == 64)
            {
                static_assert(FmhaPipeline::kN1 * FmhaPipeline::kK1 / kBlockSize ==
                              2 * kVectorSize);
            }
            else
            {
                static_assert(false, "unsupported kK1");
            }
            constexpr index_t NThreadPerBlock = FmhaPipeline::kN1 / kVectorSize;

            // Make sure that a thread must read data from the same value token
            // and the calculation of old_value_offset should match the MakeVDramTileDistribution()
            index_t old_value_offset = get_thread_local_1d_id() / NThreadPerBlock;
            index_t new_value_offset =
                bit_cast<index_t>(permute_bits_5(bit_cast<uint32_t>(old_value_offset)));

#if !CK_TILE_REMAP_TOKEN_USING_DESC
            thread_value_token_offset = new_value_offset - old_value_offset;
            batch_offset_v += thread_value_token_offset * kargs.stride_k;
#endif
        }

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // Q/K/V DRAM and DRAM window
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
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_k - thread_key_token_offset, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            auto k_dram_transformed =
                transform_tensor_view(k_dram_naive,
                                      make_tuple(make_functor_transform(
                                                     [](auto idx) {
#if CK_TILE_REMAP_TOKEN_USING_DESC
                                                         return bit_cast<index_t>(permute_bits_5(
                                                             bit_cast<uint32_t>(idx)));
#else
                                                         return idx;
#endif
                                                     },
                                                     kargs.seqlen_k - thread_key_token_offset),
                                                 make_pass_through_transform(kargs.hdim_q)),
                                      make_tuple(sequence<0>{}, sequence<1>{}),
                                      make_tuple(sequence<0>{}, sequence<1>{}));

            return pad_tensor_view(
                k_dram_transformed,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_k - thread_value_token_offset, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                number<FmhaPipeline::kAlignmentV>{},
                number<1>{});

            auto v_dram_transformed =
                transform_tensor_view(v_dram_naive,
                                      make_tuple(make_functor_transform(
                                                     [](auto idx) {
#if CK_TILE_REMAP_TOKEN_USING_DESC
                                                         return bit_cast<index_t>(permute_bits_5(
                                                             bit_cast<uint32_t>(idx)));
#else
                                                         return idx;
#endif
                                                     },
                                                     kargs.seqlen_k - thread_value_token_offset),
                                                 make_pass_through_transform(kargs.hdim_v)),
                                      make_tuple(sequence<0>{}, sequence<1>{}),
                                      make_tuple(sequence<0>{}, sequence<1>{}));

            return pad_tensor_view(
                v_dram_transformed,
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

        // Notice: When using double buffering, make sure both buffers are in the same array.
        // This prevents the compiler from using separate VGPRs to store the base address
        // and enables the use of immediate offsets in load/store instructions.
        __shared__ char
            smem_k[2]
                  [FmhaPipeline::Policy::template GetSmemSizeK<typename FmhaPipeline::Problem>()];
        __shared__ char
            smem_v[2]
                  [FmhaPipeline::Policy::template GetSmemSizeV<typename FmhaPipeline::Problem>()];
        __shared__ char smem[1];

        auto o_acc_tile = [&]() {
            return FmhaPipeline{}(q_dram_window,
                                  k_dram_window,
                                  v_dram_window,
                                  lse_dram_window,
                                  mask,
                                  kargs.scale_s,
                                  reinterpret_cast<KDataType*>(smem_k[0]),
                                  reinterpret_cast<KDataType*>(smem_k[1]),
                                  reinterpret_cast<VDataType*>(smem_v[0]),
                                  reinterpret_cast<VDataType*>(smem_v[1]),
                                  smem);
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

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};
} // namespace ck_tile
