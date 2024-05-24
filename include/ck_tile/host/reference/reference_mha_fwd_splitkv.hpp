// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

#include "ck_tile/host/reference/reference_batched_elementwise.hpp"
#include "ck_tile/host/reference/reference_batched_gemm.hpp"
#include "ck_tile/host/reference/reference_batched_masking.hpp"
#include "ck_tile/host/reference/reference_batched_softmax.hpp"

namespace ck_tile {

template <typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename QueryTensor,
          typename KeyTensor,
          typename ValueTensor,
          typename BiasTensor,
          typename LSEAccTensor,
          typename OutputAccTensor,
          typename MaskingType,
          typename PComputeElementFunction = identity>
CK_TILE_HOST void
reference_mha_fwd_splitkv(const QueryTensor& query_bhsd,
                          const KeyTensor& key_bhsd,
                          const ValueTensor& value_bhsd,
                          std::optional<BiasTensor> bias_bhss,
                          LSEAccTensor& lse_acc_nbhs,
                          OutputAccTensor& output_acc_nbhsd,
                          index_t nhead_k,
                          float scale_s,
                          const MaskingType& mask,
                          std::optional<span<const int32_t>> seqstart_q, // only used in group mode
                          std::optional<span<const int32_t>> seqstart_k, // only used in group mode
                          PComputeElementFunction p_compute_element_func = {})
{
    assert(!(seqstart_q.has_value() ^ seqstart_k.has_value()));

    const bool is_batch_mode = !seqstart_q.has_value();
    const index_t batch      = (is_batch_mode ? query_bhsd.get_length(0) : seqstart_q->size() - 1);
    const index_t nhead      = query_bhsd.get_length(1);
    const index_t num_splits = lse_acc_nbhs.get_length(0);

    using QueryDataType = tensor_value_t<QueryTensor>;
    using KeyDataType   = tensor_value_t<KeyTensor>;
    using ValueDataType = tensor_value_t<ValueTensor>;
    using BiasDataType  = tensor_value_t<BiasTensor>;

    using OaccDataType = tensor_value_t<OutputAccTensor>;

    // verify result individually for each batch/group
    for(index_t i_batch = 0; i_batch < batch; ++i_batch)
    {
        const index_t real_seqlen_q =
            (is_batch_mode ? query_bhsd.get_length(2)
                           : (*seqstart_q)[i_batch + 1] - (*seqstart_q)[i_batch]);
        const index_t real_seqlen_k =
            (is_batch_mode ? key_bhsd.get_length(2)
                           : (*seqstart_k)[i_batch + 1] - (*seqstart_k)[i_batch]);

        // adjust matrix index according to the mode
        const index_t batch_start = (is_batch_mode ? i_batch : 0);
        const index_t batch_end   = batch_start + 1;
        const index_t query_start = (is_batch_mode ? 0 : (*seqstart_q)[i_batch]);
        const index_t query_end   = query_start + real_seqlen_q;
        const index_t key_start   = (is_batch_mode ? 0 : (*seqstart_k)[i_batch]);
        const index_t key_end     = key_start + real_seqlen_k;
        const index_t nr          = nhead / nhead_k;

        for(index_t i_split = 0; i_split < num_splits; ++i_split)
        {
            index_t num_key_per_split = real_seqlen_k / num_splits;
            index_t split_key_start   = key_start + i_split * num_key_per_split;
            index_t split_key_end =
                (i_split == num_splits - 1 ? key_end : split_key_start + num_key_per_split);

            // clang-format off
            using Slice = HostTensorSlice;
            // tensor layout will be in [h, s, d] layout in verification
            auto query_view_hsd = query_bhsd
                    .index({Slice(0, batch_start, batch_end), Slice(2, query_start, query_end)})
                    .squeeze(0);
            auto key_view_hsd = key_bhsd
                    .index({Slice(0, batch_start, batch_end), Slice(2, split_key_start, split_key_end)})
                    .squeeze(0)
                    .repeat({nr, 1, 1});
            auto value_view_hsd = value_bhsd
                    .index({Slice(0, batch_start, batch_end), Slice(3, split_key_start, split_key_end)})
                    .squeeze(0)
                    .repeat({nr, 1, 1});
            auto output_acc_view_hsd = output_acc_nbhsd
                    .index({Slice(0, i_split, i_split + 1),
                            Slice(1, batch_start, batch_end), Slice(3, query_start, query_end)})
                    .squeeze(0)
                    .squeeze(0);
            // clang-format on

            // create local tensors to speed-up computation
            HostTensor<QueryDataType> query_hsd(query_view_hsd.get_lengths());
            HostTensor<KeyDataType> key_hsd(key_view_hsd.get_lengths());
            HostTensor<ValueDataType> value_hsd(value_view_hsd.get_lengths());
            // create local tensors for holding intermediate result
            HostTensor<SMPLComputeDataType> s_hss(
                {nhead, real_seqlen_q, split_key_end - split_key_start});
            HostTensor<PDataType> p_hss({nhead, real_seqlen_q, split_key_end - split_key_start});

            query_hsd.for_each([&](auto& self, auto i) { self(i) = query_view_hsd(i); });
            key_hsd.for_each([&](auto& self, auto i) { self(i) = key_view_hsd(i); });
            value_hsd.for_each([&](auto& self, auto i) { self(i) = value_view_hsd(i); });

            // reference
            reference_batched_gemm<SaccDataType>(
                query_hsd, key_hsd, s_hss, identity{}, identity{}, scales(scale_s));

            if(bias_bhss.has_value())
            {
                // clang-format off
                auto bias_view_hss = (*bias_bhss)
                        .index({Slice(2, query_start, query_end)})
                        .squeeze(0);
                // clang-format on

                // create local tensor to speed-up computation
                HostTensor<BiasDataType> bias_hss(bias_view_hss.get_lengths());
                bias_hss.for_each([&](auto& self, auto i) { self(i) = bias_view_hss(i); });

                // broadcast from [1, real_seqlen_q, real_seqlen_k] to [nhead, real_seqlen_q,
                // real_seqlen_k]
                reference_batched_elementwise<SMPLComputeDataType>(s_hss, bias_hss, s_hss);
            }

            if(mask.type == mask_enum::no_mask)
            {
                reference_batched_masking(s_hss, FmhaMasks::NoMask{real_seqlen_q, real_seqlen_k});
            }
            else if(mask.type == mask_enum::window_generic)
            {
                reference_batched_masking(
                    s_hss,
                    make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                        mask.left, mask.right, real_seqlen_q, real_seqlen_k));
            }
            else
            {
                // if left window size is negative, means causal
                // else means generic (for current batch)
                if(mask.left < 0)
                    reference_batched_masking(
                        s_hss,
                        make_generic_attention_mask_from_lr_window<FmhaMasks::CausalMask>(
                            mask.left,
                            mask.right,
                            real_seqlen_q,
                            real_seqlen_k,
                            mask.type == mask_enum::mask_top_left));
                else
                    reference_batched_masking(
                        s_hss,
                        make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                            mask.left,
                            mask.right,
                            real_seqlen_q,
                            real_seqlen_k,
                            mask.type == mask_enum::mask_top_left));
            }

            // clang-format off
            auto lse_acc_view_hs = lse_acc_nbhs
                    .index({Slice(0, i_split, i_split + 1),
                            Slice(1, batch_start, batch_end), Slice(3, query_start, query_end)})
                    .squeeze(0)
                    .squeeze(0);
            // clang-format on

            reference_batched_softmax<SMPLComputeDataType>(
                s_hss, p_hss, p_compute_element_func, lse_acc_view_hs);

            reference_batched_gemm<OaccDataType>(p_hss, value_hsd, output_acc_view_hsd);
        }
    }
}

} // namespace ck_tile
