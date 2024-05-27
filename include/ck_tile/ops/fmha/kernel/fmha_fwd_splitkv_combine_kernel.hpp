// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {

template <typename TilePartitioner_, typename MainPipeline_, typename EpiloguePipeline_>
struct FmhaFwdSplitKVCombineKernel
{
    using TilePartitioner  = ck_tile::remove_cvref_t<TilePartitioner_>;
    using MainPipeline     = ck_tile::remove_cvref_t<MainPipeline_>;
    using EpiloguePipeline = ck_tile::remove_cvref_t<EpiloguePipeline_>;

    static constexpr ck_tile::index_t kBlockSize  = MainPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = MainPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using LSEDataType  = ck_tile::remove_cvref_t<typename MainPipeline::LSEDataType>;
    using OaccDataType = ck_tile::remove_cvref_t<typename MainPipeline::OaccDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename MainPipeline::ODataType>;

    static constexpr bool kIsGroupMode      = MainPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = MainPipeline::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV      = MainPipeline::kPadHeadDimV;
    static constexpr bool kStoreLSE         = MainPipeline::kStoreLSE;
    static constexpr bool kDoFp8StaticQuant = MainPipeline::Problem::kDoFp8StaticQuant;

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct EmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct CommonKargs
    {
        const void* lse_acc_ptr;
        const void* o_acc_ptr;
        void* o_ptr;

        ck_tile::index_t batch;
        ck_tile::index_t nhead;
        ck_tile::index_t max_seqlen_q;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t hdim_v;
        ck_tile::index_t num_splits;

        ck_tile::index_t row_stride_o;
        ck_tile::index_t nhead_stride_o;
    };

    struct CommonLSEKargs
    {
        void* lse_ptr                     = nullptr;
        ck_tile::index_t nhead_stride_lse = 0;
    };

    struct Fp8StaticQuantKargs
    {
        float scale_o;
    };

    struct BatchModeLSEKargs : CommonLSEKargs
    {
        ck_tile::index_t batch_stride_lse = 0;
    };

    struct BatchModeKargs
        : CommonKargs,
          std::conditional_t<kStoreLSE, BatchModeLSEKargs, EmptyKargs<0>>,
          std::conditional_t<kDoFp8StaticQuant, Fp8StaticQuantKargs, EmptyKargs<1>>
    {
        ck_tile::index_t batch_stride_o;
    };

    struct GroupModeKargs
        : CommonKargs,
          std::conditional_t<kStoreLSE, CommonLSEKargs, EmptyKargs<0>>,
          std::conditional_t<kDoFp8StaticQuant, Fp8StaticQuantKargs, EmptyKargs<3>>
    {
        const int32_t* seqstart_q_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode, GroupModeKargs, BatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* lse_acc_ptr,
              const void* o_acc_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t batch,
              ck_tile::index_t nhead,
              ck_tile::index_t max_seqlen_q,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_splits,
              float scale_o,
              ck_tile::index_t row_stride_o,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t batch_stride_lse,
              ck_tile::index_t batch_stride_o)
    {
        Kargs kargs{{lse_acc_ptr,
                     o_acc_ptr,
                     o_ptr,
                     batch,
                     nhead,
                     max_seqlen_q,
                     seqlen_q,
                     hdim_v,
                     num_splits,
                     row_stride_o,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for lse
                    {},               // placeholder for fp8_static_quant args
                    batch_stride_o};

        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
            kargs.batch_stride_lse = batch_stride_lse;
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_o = scale_o;
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* lse_acc_ptr,
              const void* o_acc_ptr,
              void* lse_ptr,
              void* o_ptr,
              ck_tile::index_t batch,
              ck_tile::index_t nhead,
              ck_tile::index_t max_seqlen_q,
              const void* seqstart_q_ptr,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_splits,
              float scale_o,
              ck_tile::index_t row_stride_o,
              ck_tile::index_t nhead_stride_lse,
              ck_tile::index_t nhead_stride_o)
    {
        Kargs kargs{{lse_acc_ptr,
                     o_acc_ptr,
                     o_ptr,
                     batch,
                     nhead,
                     max_seqlen_q,
                     -1, // seqlen will be updated by another pointer
                     hdim_v,
                     num_splits,
                     row_stride_o,
                     nhead_stride_o}, // args for common karg
                    {},               // placeholder for lse
                    {},               // placeholder for fp8_static_quant args
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr)};

        if constexpr(kStoreLSE)
        {
            kargs.lse_ptr          = lse_ptr;
            kargs.nhead_stride_lse = nhead_stride_lse;
        }
        if constexpr(kDoFp8StaticQuant)
        {
            kargs.scale_o = scale_o;
        }

        return kargs;
    }

    __host__ static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_q_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_q_);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(MainPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_q, kargs.hdim_v);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * MainPipeline::kM0);

        long_index_t batch_offset_lse_acc = 0;
        long_index_t batch_offset_o_acc   = 0;
        long_index_t batch_offset_lse     = 0;
        long_index_t batch_offset_o       = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_lse_acc = query_start;
            batch_offset_o_acc   = query_start * kargs.hdim_v;
            if constexpr(kStoreLSE)
            {
                batch_offset_lse = query_start;
            }
            batch_offset_o = query_start * kargs.row_stride_o;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }
        }
        else
        {
            batch_offset_lse_acc =
                static_cast<long_index_t>(i_batch) * (kargs.nhead * kargs.max_seqlen_q);
            batch_offset_o_acc = static_cast<long_index_t>(i_batch) *
                                 (kargs.nhead * kargs.max_seqlen_q * kargs.hdim_v);
            if constexpr(kStoreLSE)
            {
                batch_offset_lse = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
            }
            batch_offset_o = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
        }

        // for simplicity, batch stride we just modify the pointer
        const LSEDataType* lse_acc_ptr = reinterpret_cast<const LSEDataType*>(kargs.lse_acc_ptr) +
                                         static_cast<long_index_t>(i_nhead) * (kargs.max_seqlen_q) +
                                         batch_offset_lse_acc;
        const OaccDataType* o_acc_ptr =
            reinterpret_cast<const OaccDataType*>(kargs.o_acc_ptr) +
            static_cast<long_index_t>(i_nhead) * (kargs.max_seqlen_q * kargs.hdim_v) +
            batch_offset_o_acc;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // LSEacc/Oacc DRAM and DRAM windows
        const auto lse_acc_dram = [&]() {
            const auto lse_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                lse_acc_ptr,
                make_tuple(kargs.num_splits, kargs.seqlen_q),
                make_tuple(kargs.batch * kargs.nhead * kargs.max_seqlen_q, 1),
                number<8>{},
                number<1>{});

            return pad_tensor_view(
                lse_acc_dram_naive,
                make_tuple(number<MainPipeline::kMaxSplits>{}, number<MainPipeline::kM0>{}),
                sequence<true, kPadSeqLenQ>{});
        }();

        auto o_acc_dram = [&]() {
            const auto o_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_acc_ptr,
                make_tuple(kargs.num_splits, kargs.max_seqlen_q, kargs.hdim_v),
                make_tuple(
                    kargs.batch * kargs.nhead * kargs.max_seqlen_q * kargs.hdim_v, kargs.hdim_v, 1),
                number<MainPipeline::kAlignmentO>{},
                number<1>{});

            auto o_acc_dram_view = pad_tensor_view(
                o_acc_dram_naive,
                make_tuple(number<1>{}, number<MainPipeline::kM0>{}, number<MainPipeline::kN1>{}),
                sequence<false, kPadSeqLenQ, kPadHeadDimV>{});

            const index_t new_seqlen_q =
                integer_divide_ceil(kargs.seqlen_q, MainPipeline::kM0) * MainPipeline::kM0;
            const index_t new_hdim_v =
                integer_divide_ceil(kargs.hdim_v, MainPipeline::kN1) * MainPipeline::kN1;

            return transform_tensor_view(
                o_acc_dram_view,
                make_tuple(make_merge_transform(make_tuple(kargs.num_splits, new_seqlen_q)),
                           make_pass_through_transform(new_hdim_v)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }();

        auto lse_acc_dram_window = make_tile_window(
            lse_acc_dram,
            [&]() {
                return make_tuple(number<MainPipeline::kMaxSplits>{}, number<MainPipeline::kM0>{});
            }(),
            {0, i_m0});

        auto o_acc_dram_window = make_tile_window(
            o_acc_dram,
            [&]() {
                return make_tuple(number<MainPipeline::kM0>{}, number<MainPipeline::kN1>{});
            }(),
            {i_m0, 0});

        // LSE DRAM window
        auto lse_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto lse_dram_window_lengths = make_tuple(number<MainPipeline::kM0>{});
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

        auto o_acc_tile = [&]() {
            if constexpr(kDoFp8StaticQuant)
            {
                return MainPipeline{}(
                    lse_acc_dram_window,
                    o_acc_dram_window,
                    lse_dram_window,
                    identity{},                                          // lse_element_func
                    composes(saturates<fp8_t>{}, scales{kargs.scale_o}), // o_acc_element_func
                    smem_ptr,
                    kargs.num_splits,
                    kargs.max_seqlen_q);
            }
            else
            {
                return MainPipeline{}(lse_acc_dram_window,
                                      o_acc_dram_window,
                                      lse_dram_window,
                                      smem_ptr,
                                      kargs.num_splits,
                                      kargs.max_seqlen_q);
            }
        }();

        // O DRAM and DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.row_stride_o, 1),
                number<MainPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<MainPipeline::kM0>{}, number<MainPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<MainPipeline::kM0>{}, number<MainPipeline::kN1>{}),
                             {i_m0, 0});

        EpiloguePipeline{}(o_dram_window, o_acc_tile);
    }
};

} // namespace ck_tile
