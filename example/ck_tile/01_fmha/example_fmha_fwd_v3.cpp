// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <ck_tile/core/numeric/bfloat16.hpp>
#include <ck_tile/core/numeric/half.hpp>
#include <ck_tile/core/numeric/math.hpp>
#include <ck_tile/host/arg_parser.hpp>
#include <ck_tile/host/device_memory.hpp>
#include <ck_tile/host/fill.hpp>
#include <ck_tile/host/host_tensor.hpp>

#include "fmha_fwd_v3.hpp"

auto parse_cmd_args(int argc, char* argv[]) -> std::pair<bool, ck_tile::ArgParser>
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("prec", "fp16", "data type. fp16/bf16")
        .insert("b", "2", "batch size")
        .insert("h", "8", "num of head, for q")
        .insert("h_k",
                "-1",
                "num of head, for k/v, -1 means equal to h\n"
                "if not equal to h, then this is GQA/MQA case")
        .insert("s", "3328", "seqlen_q")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("d", "128", "head dim for q & k")
        .insert("scale_s", "0", "scale factor of S. 0 means equal to 1/sqrt(hdim)")
        .insert("iperm",
                "1",
                "permute input\n"
                "if true, will be b*h*s*d, else b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("mask",
                "0",
                "0: no mask, 1: top-left(same as 't'), 2:bottom-right(same as 'b')\n"
                "'t', top-left causal mask, 'b', bottom-r causal mask\n"
                "'t:l,r', top-left sliding window attn(swa) with FA style left right size\n"
                "'b:l,r', bottom-r sliding window attn(swa) with FA style left right size\n"
                "'xt:window_size', xformer style masking from top-left, window_size negative is "
                "causal, positive is swa\n"
                "'xb:window_size', xformer style masking from bottom-r, window_size negative is "
                "causal, positive is swa\n"
                "'g:y,x', generic attention mask coordinate with y/x size (only debug purpose for "
                "now)")
        .insert("v", "1", "0:no validation, 2:cpu validation")
        .insert("seed",
                "11939",
                "random seed used for initializing input tensors. 0 for "
                "non-deterministic seed")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "30", "number of iterations to benchmark the kernel");

    bool result = arg_parser.parse(argc, argv);
    return std::make_pair(result, arg_parser);
}

enum class TensorLayout
{
    bhsd,
    bshd,
};

std::ostream& operator<<(std::ostream& stream, TensorLayout layout)
{
    switch(layout)
    {
    case TensorLayout::bhsd: return stream << "bhsd";
    case TensorLayout::bshd: return stream << "bshd";
    default: return stream << "unknown";
    }
}

struct Problem
{
    explicit Problem(const ck_tile::ArgParser& args)
    {
        data_type = args.get_str("prec") == "fp16"
                        ? ck_tile::fmha_fwd_v3_args::data_type_enum::fp16
                        : ck_tile::fmha_fwd_v3_args::data_type_enum::bf16;
        batch     = args.get_int("b");
        seqlen_q  = args.get_int("s");
        seqlen_k  = args.get_int("s_k");
        if(seqlen_k < 0)
        {
            seqlen_k = seqlen_q;
        }
        nhead_q  = args.get_int("h");
        nhead_kv = args.get_int("h_k");
        if(nhead_kv < 0)
        {
            nhead_kv = nhead_q;
        }
        hdim          = args.get_int("d");
        softmax_scale = args.get_float("scale_s");
        if(softmax_scale == .0f)
            softmax_scale = 1.0 / ck_tile::sqrt(static_cast<float>(hdim));

        input_layout  = args.get_int("iperm") == 1 ? TensorLayout::bhsd : TensorLayout::bshd;
        output_layout = args.get_int("operm") == 1 ? TensorLayout::bhsd : TensorLayout::bshd;
    }

    std::vector<ck_tile::index_t> get_query_shape() const
    {
        if(input_layout == TensorLayout::bhsd)
        {
            return {batch, nhead_q, seqlen_q, hdim};
        }
        else
        {
            return {batch, seqlen_q, nhead_q, hdim};
        }
    }

    std::vector<ck_tile::index_t> get_key_shape() const
    {
        if(input_layout == TensorLayout::bhsd)
        {
            return {batch, nhead_kv, seqlen_k, hdim};
        }
        else
        {
            return {batch, seqlen_k, nhead_kv, hdim};
        }
    }

    std::vector<ck_tile::index_t> get_value_shape() const
    {
        if(input_layout == TensorLayout::bhsd)
        {
            return {batch, nhead_kv, seqlen_k, hdim};
        }
        else
        {
            return {batch, seqlen_k, nhead_kv, hdim};
        }
    }

    std::vector<ck_tile::index_t> get_output_shape() const
    {
        if(output_layout == TensorLayout::bhsd)
        {
            return {batch, nhead_q, seqlen_q, hdim};
        }
        else
        {
            return {batch, seqlen_q, nhead_q, hdim};
        }
    }

    ck_tile::fmha_fwd_v3_args::data_type_enum data_type;
    ck_tile::index_t batch;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_kv;
    ck_tile::index_t hdim;
    float softmax_scale;
    TensorLayout input_layout;
    TensorLayout output_layout;
};

struct RunConfig
{
    explicit RunConfig(const ck_tile::ArgParser& args)
    {
        seed = args.get_uint32("seed");
        if(*seed == 0)
        {
            seed.reset();
        }

        kernel_warmup = args.get_int("warmup");
        kernel_repeat = args.get_int("repeat");
    }

    std::optional<uint32_t> seed;
    int kernel_warmup;
    int kernel_repeat;
};

template <typename DataType>
auto generate_qkv(const Problem& problem,
                  [[maybe_unused]] std::optional<uint32_t> seed = std::nullopt)
    -> std::tuple<ck_tile::HostTensor<DataType>,
                  ck_tile::HostTensor<DataType>,
                  ck_tile::HostTensor<DataType>>
{
    ck_tile::HostTensor<DataType> q(problem.get_query_shape());
    ck_tile::HostTensor<DataType> k(problem.get_key_shape());
    ck_tile::HostTensor<DataType> v(problem.get_value_shape());

    ck_tile::FillNormalDistribution<DataType>{0.f, 3.f, seed}(q);
    ck_tile::FillNormalDistribution<DataType>{0.f, 3.f, seed}(k);
    ck_tile::FillNormalDistribution<DataType>{0.f, 3.f, seed}(v);

    return std::make_tuple(q, k, v);
}

template <typename DataType>
bool run_impl(const Problem& problem, const RunConfig& run_config)
{
    auto [q, k, v] = generate_qkv<DataType>(problem, run_config.seed);

    ck_tile::DeviceMem q_buf(q.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(v.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(q.get_element_space_size_in_bytes());

    ck_tile::fmha_fwd_v3_args args;

    args.data_type     = problem.data_type;
    args.batch         = problem.batch;
    args.seqlen_q      = problem.seqlen_q;
    args.seqlen_k      = problem.seqlen_k;
    args.nhead_q       = problem.nhead_q;
    args.nhead_kv      = problem.nhead_kv;
    args.hdim_qk       = problem.hdim;
    args.hdim_v        = problem.hdim;
    args.softmax_scale = problem.softmax_scale;

    // bshd: (batch, seqlen_q, nhead_q, hdim)
    // bhsd: (batch, nhead_q, seqlen_q, hdim)
    args.mask_type = 0;
    args.q_ptr     = q_buf.GetDeviceBuffer();
    args.stride_q =
        problem.input_layout == TensorLayout::bshd ? problem.nhead_q * problem.hdim : problem.hdim;
    args.nhead_stride_q =
        problem.input_layout == TensorLayout::bshd ? problem.hdim : problem.seqlen_q * problem.hdim;
    args.batch_stride_q = problem.seqlen_q * problem.nhead_q * problem.hdim;

    // bshd: (batch, seqlen_k, nhead_kv, hdim)
    // bhsd: (batch, nhead_kv, seqlen_k, hdim)
    args.k_ptr = k_buf.GetDeviceBuffer();
    args.stride_k =
        problem.input_layout == TensorLayout::bshd ? problem.nhead_kv * problem.hdim : problem.hdim;
    args.nhead_stride_k =
        problem.input_layout == TensorLayout::bshd ? problem.hdim : problem.seqlen_k * problem.hdim;
    args.batch_stride_k = problem.seqlen_k * problem.nhead_kv * problem.hdim;

    // bshd: (batch, seqlen_k, nhead_kv, hdim)
    // bhsd: (batch, nhead_kv, seqlen_k, hdim)
    args.v_ptr = v_buf.GetDeviceBuffer();
    args.stride_v =
        problem.input_layout == TensorLayout::bshd ? problem.nhead_kv * problem.hdim : problem.hdim;
    args.nhead_stride_v =
        problem.input_layout == TensorLayout::bshd ? problem.hdim : problem.seqlen_k * problem.hdim;
    args.batch_stride_v = problem.seqlen_k * problem.nhead_kv * problem.hdim;

    // bshd: (batch, seqlen_q, nhead_q, hdim)
    // bhsd: (batch, nhead_q, seqlen_q, hdim)
    args.o_ptr = o_buf.GetDeviceBuffer();
    args.stride_o =
        problem.output_layout == TensorLayout::bshd ? problem.nhead_q * problem.hdim : problem.hdim;
    args.nhead_stride_o = problem.output_layout == TensorLayout::bshd
                              ? problem.hdim
                              : problem.seqlen_q * problem.hdim;
    args.batch_stride_o = problem.seqlen_q * problem.nhead_q * problem.hdim;

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /*log_level=*/0,
                                         run_config.kernel_warmup,
                                         run_config.kernel_repeat};

    auto [result, time] = ck_tile::fmha_fwd_v3(args, stream_config);

    /// TODO: consider the real flop if we have mask
    std::size_t flop =
        4 * problem.batch * problem.nhead_q * problem.seqlen_q * problem.seqlen_k * problem.hdim;

    float tflops = static_cast<float>(flop) / 1.e9 / time;

    std::cout << "[" << problem.data_type << "|" << problem.input_layout << "-"
              << problem.output_layout << "] b:" << problem.batch << ", h:" << problem.nhead_q
              << "/" << problem.nhead_kv << ", s:" << problem.seqlen_q << "/" << problem.seqlen_k
              << ", d:" << problem.hdim << ", scale_s:" << problem.softmax_scale << std::fixed
              << ", " << std::setprecision(3) << time << " ms, " << std::setprecision(2) << tflops
              << " TFlops" << std::endl;

    return result;
}

int main(int argc, char* argv[])
{
    auto [parse_result, args] = parse_cmd_args(argc, argv);
    if(!parse_result)
    {
        std::cerr << "failed to parse command line arguments" << std::endl;
    }

    Problem problem(args);
    RunConfig run_config(args);

    const auto run = [&] {
        if(problem.data_type == ck_tile::fmha_fwd_v3_args::data_type_enum::fp16)
        {
            return run_impl<ck_tile::fp16_t>(problem, run_config);
        }
        else
        {
            return run_impl<ck_tile::bf16_t>(problem, run_config);
        }
    };

    return !run();
}
