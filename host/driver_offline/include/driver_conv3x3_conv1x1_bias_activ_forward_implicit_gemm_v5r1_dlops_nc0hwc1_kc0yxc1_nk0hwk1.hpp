#ifndef DRIVER_CONVOLUTION_BIAS_ACTIV_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP
#define DRIVER_CONVOLUTION_BIAS_ACTIV_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_dlops_v3.hpp"

namespace ck {

template <typename GemmArguments,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatC>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_conv_bias_activ_dlops_v4(const GemmArguments gemm_arg,
                                        const FloatAB* __restrict__ p_a_grid,
                                        const FloatAB* __restrict__ p_b_grid,
                                        const FloatBias* __restrict__ p_bias_grid,
                                        FloatC* __restrict__ p_c_grid,
                                        float scaleGemm)
{
    // constexpr auto gemm_arg = GemmArguments{};

    using GridwiseGemm = decltype(gemm_arg.gridwise_gemm_desc);

    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = gemm_arg.a_e0_e1_k0_k1_e2_grid_desc;
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        gemm_arg.b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc;
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc =
        gemm_arg.c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc;

    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        gemm_arg.c_blockid_to_k_n_h_w_block_cluster_adaptor;

    constexpr auto c_k1_n_h2_w2_thread_gemm_desc = GridwiseGemm::MakeCK1NH2W2ThreadDescriptor();

    // register allocation for output
    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatAcc,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        c_thread_buf;

    static_for<0, c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(), 1>{}(
        [&](auto i) { c_thread_buf(i) = 0; });

    const auto c_k_n_h_w_block_cluster_idx =
        GridwiseGemm::GetCBlockIndex(c_blockid_to_k_n_h_w_block_cluster_adaptor, get_block_1d_id());

    const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_a_grid, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
    const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_b_grid, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());

    const auto c_thread_mtx_index = GridwiseGemm::GetCThreadIndex();

    constexpr auto HasMainE0BlockLoop = false;

    // GemmOp
    GridwiseGemm::GemmOpHasE1Loop(a_global_buf,
                                  b_global_buf,
                                  c_thread_buf,
                                  p_shared_block,
                                  c_k_n_h_w_block_cluster_idx,
                                  c_thread_mtx_index,
                                  a_e0_e1_k0_k1_e2_grid_desc,
                                  b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                  c_k1_n_h2_w2_thread_gemm_desc,
                                  integral_constant<bool, HasMainE0BlockLoop>{});

    // Bias
    // const auto bias_k0_k1_grid_desc =
    // GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    // const auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
    // p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    // GridwiseGemm::BiasOp(bias_global_buf,
    // c_thread_buf,
    // c_k_n_h_w_block_cluster_idx,
    // c_thread_mtx_index,
    // bias_k0_k1_grid_desc,
    // c_k1_n_h2_w2_thread_gemm_desc);

    // GridwiseGemm::ActivOp(c_thread_buf,
    // c_k1_n_h2_w2_thread_gemm_desc,
    // ck::tensor_operation::element_wise::RequantHardTanh{scaleGemm});

    auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    GridwiseGemm::WriteOut(c_thread_buf,
                           c_global_buf,
                           c_k_n_h_w_block_cluster_idx,
                           c_thread_mtx_index,
                           c_k1_n_h2_w2_thread_gemm_desc,
                           c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                           ck::tensor_operation::element_wise::PassThrough{});
}

template <typename GridwiseGemmDesc,
          typename AE0E1K0K1E2GridDesc,
          typename BE0E1NH0H1H2W0W1W2E2GridDesc,
          typename CK0K1NH0H1H2W0W1W2GridDesc,
          typename CBlockIdToKNHoWoBlockClusterAdaptor>
struct GemmArguments
{
    // static constexpr auto gemmwise_gemm_desc = GridwiseGemmDesc{};

    // GridwiseGemmDesc gridwise_gemm_desc;
    // AE0E1K0K1E2GridDesc a_e0_e1_k0_k1_e2_grid_desc;
    // BE0E1NH0H1H2W0W1W2E2GridDesc b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc;
    // CK0K1NH0H1H2W0W1W2GridDesc c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc;
    // CBlockIdToKNHoWoBlockClusterAdaptor c_blockid_to_k_n_h_w_block_cluster_adaptor;

    static constexpr auto gridwise_gemm_desc                       = GridwiseGemmDesc{};
    static constexpr auto a_e0_e1_k0_k1_e2_grid_desc               = AE0E1K0K1E2GridDesc{};
    static constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc = BE0E1NH0H1H2W0W1W2E2GridDesc{};
    static constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc    = CK0K1NH0H1H2W0W1W2GridDesc{};
    static constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToKNHoWoBlockClusterAdaptor{};

    index_t grid_size;
    index_t block_size;
    bool has_main_e0_block_loop;

    GemmArguments(
        // const GridwiseGemmDesc gridwise_gemm_desc_,
        // const AE0E1K0K1E2GridDesc a_e0_e1_k0_k1_e2_grid_desc_,
        // const BE0E1NH0H1H2W0W1W2E2GridDesc b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_,
        // const CK0K1NH0H1H2W0W1W2GridDesc c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_,
        // const CBlockIdToKNHoWoBlockClusterAdaptor c_blockid_to_k_n_h_w_block_cluster_adaptor_,
        const index_t grid_size_,
        const index_t block_size_,
        const bool has_main_e0_block_loop_)
    {
        // gridwise_gemm_desc = gridwise_gemm_desc_;

        // a_e0_e1_k0_k1_e2_grid_desc                 = a_e0_e1_k0_k1_e2_grid_desc_;
        // b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc   = b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_;
        // c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc      = c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_;
        // c_blockid_to_k_n_h_w_block_cluster_adaptor = c_blockid_to_k_n_h_w_block_cluster_adaptor_;

        grid_size              = grid_size_;
        block_size             = block_size_;
        has_main_e0_block_loop = has_main_e0_block_loop_;
    }
};

template <typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          typename GridGemmTuningParameters,
          typename ConvDesc>
constexpr auto MakeGridwiseGemm(ConvDesc conv_desc)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    constexpr auto BlockSize = Number<GridGemmTuningParameters::BlockSize>{};

    constexpr auto E1 = Number<GridGemmTuningParameters::E1>{};
    constexpr auto E2 = Number<GridGemmTuningParameters::E2>{};
    constexpr auto K2 = Number<GridGemmTuningParameters::K2>{};

    constexpr auto E0PerBlock = Number<GridGemmTuningParameters::E0PerBlock>{};
    constexpr auto KPerBlock  = Number<GridGemmTuningParameters::KPerBlock>{};
    constexpr auto HoPerBlock = Number<GridGemmTuningParameters::HoPerBlock>{};
    constexpr auto WoPerBlock = Number<GridGemmTuningParameters::WoPerBlock>{};
    constexpr auto E1PerBlock = Number<GridGemmTuningParameters::E1PerBlock>{};

    constexpr auto KPerThread  = Number<GridGemmTuningParameters::KPerThread>{};
    constexpr auto HoPerThread = Number<GridGemmTuningParameters::HoPerThread>{};
    constexpr auto WoPerThread = Number<GridGemmTuningParameters::WoPerThread>{};
    constexpr auto EPerThread  = Number<GridGemmTuningParameters::EPerThread>{};

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2 =
        decltype(GridGemmTuningParameters::ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2);
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 =
        decltype(GridGemmTuningParameters::ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2);

    constexpr auto ABlockTransferSrcScalarPerVector_E2 =
        GridGemmTuningParameters::ABlockTransferSrcScalarPerVector_E2;
    constexpr auto ABlockTransferDstScalarPerVector_E2 =
        GridGemmTuningParameters::ABlockTransferDstScalarPerVector_E2;
    constexpr auto BThreadTransferSrcScalarPerVector_E2 =
        GridGemmTuningParameters::BThreadTransferSrcScalarPerVector_E2;
    constexpr auto CThreadTransferDstScalarPerVector_K =
        GridGemmTuningParameters::CThreadTransferDstScalarPerVector_K;

    constexpr auto in_n_c0_hi_wi_c1_global_desc = conv_desc.in_n_c0_hi_wi_c1_desc;

    constexpr auto N  = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I0)>{};
    constexpr auto C0 = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I1)>{};
    constexpr auto Hi = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I2)>{};
    constexpr auto Wi = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I3)>{};
    // constexpr auto C1 = Number<in_n_c0_hi_wi_c1_global_desc.GetLength(I4)>{};

    constexpr auto out_n_k0_ho_wo_k1_global_desc = conv_desc.out_n_k0_ho_wo_k1_desc;

    constexpr auto K0 = Number<out_n_k0_ho_wo_k1_global_desc.GetLength(I1)>{};
    constexpr auto Ho = Number<out_n_k0_ho_wo_k1_global_desc.GetLength(I2)>{};
    constexpr auto Wo = Number<out_n_k0_ho_wo_k1_global_desc.GetLength(I3)>{};
    constexpr auto K1 = Number<out_n_k0_ho_wo_k1_global_desc.GetLength(I4)>{};

    constexpr auto wei_k_c0_y_x_c1_global_desc = conv_desc.wei_c0_y_x_k_c1_desc;

    constexpr auto K = Number<wei_k_c0_y_x_c1_global_desc.GetLength(I0)>{};
    constexpr auto Y = Number<wei_k_c0_y_x_c1_global_desc.GetLength(I2)>{};
    constexpr auto X = Number<wei_k_c0_y_x_c1_global_desc.GetLength(I3)>{};

    constexpr auto ConvStrideH = Number<conv_desc.conv_strides[I0]>{};
    constexpr auto ConvStrideW = Number<conv_desc.conv_strides[I1]>{};

    constexpr auto ConvDilationH = Number<conv_desc.conv_dilations[I0]>{};
    constexpr auto ConvDilationW = Number<conv_desc.conv_dilations[I1]>{};

    constexpr auto Hop = Number<(Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock>{};
    constexpr auto Wop = Number<(Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock>{};

    constexpr auto InLeftPadH = Number<conv_desc.in_left_pads[I0]>{};
    constexpr auto InLeftPadW = Number<conv_desc.in_left_pads[I1]>{};

    constexpr auto OutRightPadH = Hop - Ho;
    constexpr auto OutRightPadW = Wop - Wo;

    constexpr auto InRightPadH = Number<conv_desc.in_right_pads[I0] + OutRightPadH * ConvStrideH>{};
    constexpr auto InRightPadW = Number<conv_desc.in_right_pads[I1] + OutRightPadW * ConvStrideW>{};

    if((C0 * Y * X) % (E1 * E0PerBlock) != 0)
    {
        std::cerr << "E = " << C0 * Y * X << " E1 = " << E1 << " E0PerBlock = " << E0PerBlock
                  << std::endl;
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    constexpr auto E  = Number<C0 * Y * X>{};
    constexpr auto E0 = Number<E / E1>{};

    // weight tensor
    constexpr auto a_e_k_e2_grid_desc =
        transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, E, E2)),
                                    make_tuple(make_pass_through_transform(K),
                                               make_pass_through_transform(E),
                                               make_pass_through_transform(E2)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                    make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}));

    static_assert(a_e_k_e2_grid_desc.IsKnownAtCompileTime(), "");

    constexpr auto a_e0_e1_k_e2_grid_desc =
        transform_tensor_descriptor(a_e_k_e2_grid_desc,
                                    make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                                               make_pass_through_transform(K),
                                               make_pass_through_transform(E2)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

    static_assert(a_e0_e1_k_e2_grid_desc.IsKnownAtCompileTime(), "");

    // input tensor
    constexpr auto in_n_c0_hip_wip_e2_global_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, E2)),
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C0),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW),
                   make_pass_through_transform(E2)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

    static_assert(in_n_c0_hip_wip_e2_global_desc.IsKnownAtCompileTime(), "");

    constexpr auto in_n_c0_y_ho_x_wo_e2_global_desc = transform_tensor_descriptor(
        in_n_c0_hip_wip_e2_global_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C0),
                   make_embed_transform(make_tuple(Y, Hop), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wop), make_tuple(ConvDilationW, ConvStrideW)),
                   make_pass_through_transform(E2)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(
            Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}, Sequence<6>{}));

    static_assert(in_n_c0_y_ho_x_wo_e2_global_desc.IsKnownAtCompileTime(), "");

    constexpr auto in_e_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
        in_n_c0_y_ho_x_wo_e2_global_desc,
        make_tuple(make_merge_transform(make_tuple(C0, Y, X)),
                   make_pass_through_transform(N),
                   make_pass_through_transform(Hop),
                   make_pass_through_transform(Wop),
                   make_pass_through_transform(E2)),
        make_tuple(Sequence<1, 2, 4>{}, Sequence<0>{}, Sequence<3>{}, Sequence<5>{}, Sequence<6>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

    static_assert(in_e_n_ho_wo_e2_grid_desc.IsKnownAtCompileTime(), "");

    constexpr auto b_e0_e1_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
        in_e_n_ho_wo_e2_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                   make_pass_through_transform(N),
                   make_pass_through_transform(Hop),
                   make_pass_through_transform(Wop),
                   make_pass_through_transform(E2)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

    static_assert(b_e0_e1_n_ho_wo_e2_grid_desc.IsKnownAtCompileTime(), "");

    // output tensor
    constexpr auto c_k_n_hop_wop_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1)),
        make_tuple(make_merge_transform(make_tuple(K0, K1)),
                   make_pass_through_transform(N),
                   make_pad_transform(Ho, I0, OutRightPadH),
                   make_pad_transform(Wo, I0, OutRightPadW)),
        make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    std::cerr << "Hop = " << Hop << " Wop = " << Wop << std::endl;

    if(!((K % KPerBlock) == 0 && (Hop % HoPerBlock) == 0 && (Wop % WoPerBlock) == 0 &&
         (E1 % E1PerBlock) == 0))
    {
        std::cerr << "K = " << K << " KPerBlock = " << KPerBlock << " Hop = " << Hop
                  << " HoPerBlock = " << HoPerBlock << " Wop = " << Wop
                  << " WoPerBlock = " << WoPerBlock << " E1 = " << E1
                  << " E1PerBlock = " << E1PerBlock << std::endl;
        throw std::runtime_error("wrong! GEMM size no divisible");
    }

    std::cerr << "a_size = " << a_e0_e1_k_e2_grid_desc.GetElementSpaceSize() * sizeof(FloatAB)
              << ", b_size = "
              << b_e0_e1_n_ho_wo_e2_grid_desc.GetElementSpaceSize() * sizeof(FloatAB)
              << ", c = " << c_k_n_hop_wop_grid_desc.GetElementSpaceSize() * sizeof(FloatC)
              << std::endl;

    // GEMM
    using GridwiseGemm = GridwiseGemmDlops_km_kn_mn_v3<
        BlockSize,
        FloatAB,
        FloatAcc,
        FloatC,
        InMemoryDataOperationEnum_t::Set,
        decltype(a_e0_e1_k_e2_grid_desc),
        decltype(b_e0_e1_n_ho_wo_e2_grid_desc),
        decltype(c_k_n_hop_wop_grid_desc),
        E1,
        E2,
        K2,
        KPerBlock,
        HoPerBlock,
        WoPerBlock,
        E0PerBlock,
        E1PerBlock,
        KPerThread,
        HoPerThread,
        WoPerThread,
        EPerThread,
        ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
        ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
        Sequence<2, 3, 0, 1, 4>,
        Sequence<0, 1, 2, 3, 4>,
        4,
        ABlockTransferSrcScalarPerVector_E2,
        ABlockTransferDstScalarPerVector_E2,
        false, // don't move back src coordinate after threadwise copy
        Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>, // E0, E1, N, H0, H1, H2, W0, W1, W2, E2
        9,
        BThreadTransferSrcScalarPerVector_E2,
        false, // don't move back src coordinate after threadwise copy, which will be fused with
               // MoveSrcSliceWindow() to save addr computation
        Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>, // K0, K1, N, H0, H1, H2, W0, W1, W2
        1,
        CThreadTransferDstScalarPerVector_K>;

    static_assert(c_k_n_hop_wop_grid_desc.IsKnownAtCompileTime(), "");

    const auto a_e0_e1_k0_k1_e2_grid_desc =
        GridwiseGemm::MakeAE0E1K0K1E2GridDescriptor(a_e0_e1_k_e2_grid_desc);
    const auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        GridwiseGemm::MakeBE0E1NH0H1H2W0W1W2E2GridDescriptor(b_e0_e1_n_ho_wo_e2_grid_desc);
    const auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc =
        GridwiseGemm::MakeCK0K1NH0H1H2W0W1W2GridDescriptor(c_k_n_hop_wop_grid_desc);

    static_assert(c_k_n_hop_wop_grid_desc.IsKnownAtCompileTime(), "");

    const auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        GridwiseGemm::MakeCBlockIdToKNHoWoBlockClusterAdaptor(c_k_n_hop_wop_grid_desc);

    const auto grid_size = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;

    const bool has_main_e0_block_loop = E0 > 1;

    std::cerr << "grid_size = " << grid_size
              << " has_main_e0_block_loop = " << has_main_e0_block_loop << std::endl;

    static_assert(a_e0_e1_k0_k1_e2_grid_desc.IsKnownAtCompileTime(), "");
    static_assert(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.IsKnownAtCompileTime(), "");
    static_assert(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.IsKnownAtCompileTime(), "");
    static_assert(c_blockid_to_k_n_h_w_block_cluster_adaptor.IsKnownAtCompileTime(), "");

    using AGridDesc_E0_E1_K0_K1_E2 = decltype(a_e0_e1_k0_k1_e2_grid_desc);
    using BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 =
        decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc);
    using CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 = decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
    using CBlockIdToBlockClusterAdaptor_K_N_H_W =
        decltype(c_blockid_to_k_n_h_w_block_cluster_adaptor);

    return GemmArguments<GridwiseGemm,
                         AGridDesc_E0_E1_K0_K1_E2,
                         BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
                         CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
                         CBlockIdToBlockClusterAdaptor_K_N_H_W>(
        grid_size, BlockSize, has_main_e0_block_loop);
}

template <typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          typename GridGemmTuningParameters1,
          typename GridGemmTuningParameters2>
struct DriverDynamicConv3x3Conv1x1BiasActivForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1
{
    template <typename Conv1Desc, typename Conv2Desc>
    __host__ float Run(Conv1Desc conv1_desc,
                       const FloatAB* __restrict__ p_a1_grid,
                       const FloatAB* __restrict__ p_b1_grid,
                       const FloatC* __restrict__ p_bias1_grid,
                       FloatC* __restrict__ p_c1_grid,
                       Conv2Desc conv2_desc,
                       const FloatAB* __restrict__ p_a2_grid,
                       const FloatC* __restrict__ p_bias2_grid,
                       FloatC* __restrict__ p_c2_grid,
                       const int nrepeat) const
    {
        const auto GemmArg1 =
            MakeGridwiseGemm<FloatAB, FloatAcc, FloatC, GridGemmTuningParameters1>(conv1_desc);

        const auto GemmArg2 =
            MakeGridwiseGemm<FloatAB, FloatAcc, FloatC, GridGemmTuningParameters2>(conv2_desc);

        conv1_desc.printConvDesc();
        GridGemmTuningParameters1{}.printTuningParameters();

        float ave_time = 0;
        static_assert(GemmArg1.a_e0_e1_k0_k1_e2_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(GemmArg1.b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(GemmArg1.c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(GemmArg1.c_blockid_to_k_n_h_w_block_cluster_adaptor.IsKnownAtCompileTime(),
                      "");

        // using GridwiseGemm             = decltype(GemmArg1.gridwise_gemm_desc);
        // using AGridDesc_E0_E1_K0_K1_E2 = decltype(GemmArg1.a_e0_e1_k0_k1_e2_grid_desc);
        // using BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 =
        // decltype(GemmArg1.b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc);
        // using CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 =
        // decltype(GemmArg1.c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
        // using CBlockIdToBlockClusterAdaptor_K_N_H_W =
        // decltype(GemmArg1.c_blockid_to_k_n_h_w_block_cluster_adaptor);

        const auto kernel =
            kernel_conv_bias_activ_dlops_v4<decltype(GemmArg1), FloatAB, FloatAcc, FloatC, FloatC>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(GemmArg1.grid_size),
                                          dim3(GemmArg1.block_size),
                                          0,
                                          GemmArg1,
                                          p_a1_grid,
                                          p_b1_grid,
                                          p_bias1_grid,
                                          p_c1_grid,
                                          0.3);

        return ave_time;
    }
};

} // namespace ck

#endif
