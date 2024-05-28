// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>
#include <functional>

#include "ck_tile/core/container/span.hpp"

enum class mode_enum
{
    batch = 0,
    group
};

std::ostream& operator<<(std::ostream& stream, mode_enum mode)
{
    return stream << (mode == mode_enum::batch ? "batch" : "group");
}

std::vector<int32_t> to_seqstarts(ck_tile::span<const int32_t> seqlens)
{
    std::vector<int32_t> seqstarts = {0};
    for(int32_t seqlen : seqlens)
    {
        seqstarts.push_back(seqstarts.back() + seqlen);
    }
    assert(seqstarts.size() == seqlens.size() + 1);
    return seqstarts;
}

std::vector<int32_t> generate_seqlens(mode_enum mode,
                                      unsigned count,
                                      int32_t seqlens_sum,
                                      std::optional<unsigned> seed = std::nullopt)
{
    assert(0 < count);

    std::vector<int32_t> seqlens(count, seqlens_sum);

    if(mode == mode_enum::group && 1 < count)
    {
        using size_type = std::vector<int32_t>::size_type;

        std::mt19937 random_engine(seed.has_value() ? *seed : std::random_device{}());
        std::uniform_int_distribution<size_type> idx_dist(0, count - 1);
        auto next_idx = std::bind(idx_dist, std::ref(random_engine));

        std::uniform_int_distribution<size_type> step_dist(1, count - 1);
        auto next_step = std::bind(step_dist, std::ref(random_engine));

        for(unsigned repeat = seqlens_sum * (count / 2); 0 < repeat; --repeat)
        {
            const size_type to_decrease = next_idx();
            // make sure each elements of seqlens is always greater than 0
            if(seqlens[to_decrease] == 1)
            {
                continue;
            }

            const size_type to_increase = (to_decrease + next_step()) % count;

            --seqlens[to_decrease];
            ++seqlens[to_increase];
        }
    }

    return seqlens;
}

std::vector<int32_t> generate_seqstarts(mode_enum mode,
                                        unsigned count,
                                        int32_t seqlens_sum,
                                        std::optional<unsigned> seed = std::nullopt)
{
    return to_seqstarts(generate_seqlens(mode, count, seqlens_sum, seed));
}

int env_get_int(const char* var_name, int default_int)
{
    char* v = getenv(var_name);
    int r   = default_int;
    if(v)
        r = atoi(v);
    return r;
}

inline int
num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
    // If we have enough to almost fill the SMs, then just use 1 split
    if(batch_nheads_mblocks >= 0.8f * num_SMs)
    {
        return 1;
    }
    max_splits           = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 ||
               ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if(!is_split_eligible(num_splits))
        {
            efficiency.push_back(0.f);
        }
        else
        {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff     = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if(eff > max_efficiency)
            {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if(!is_split_eligible(num_splits))
        {
            continue;
        }
        if(efficiency[num_splits - 1] >= 0.85 * max_efficiency)
        {
            return num_splits;
        }
    }
    return 1;
}

inline int override_num_splits_if_necessary(int batch_size,
                                            int num_heads,
                                            int head_size,
                                            int max_seqlen_k,
                                            int max_seqlen_q,
                                            float p_dropout,
                                            int num_splits)
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return num_splits;
    }

    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return num_splits;
    }
    const int kM0 = 64;
    const int kN1 = (head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64));

    const int num_m_blocks = ck_tile::integer_divide_ceil(max_seqlen_q, kM0);
    const int num_n_blocks = ck_tile::integer_divide_ceil(max_seqlen_k, kN1);

    if(p_dropout == 0.0f)
    { // SplitKV is not implemented for dropout
        if(num_splits < 1)
        {
            return num_splits_heuristic(batch_size * num_heads * num_m_blocks,
                                        props.multiProcessorCount * 2,
                                        num_n_blocks,
                                        128);
        }
    }

    return num_splits;
}
