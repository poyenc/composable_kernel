#ifndef DRIVER_CONVOLUTION_BIAS_ACTIV_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP
#define DRIVER_CONVOLUTION_BIAS_ACTIV_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_dlops_v3.hpp"

template <ck::index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatC,
          ck::index_t E1_,
          ck::index_t E2_,
          ck::index_t K2_,
          ck::index_t KPerBlock,
          ck::index_t HoPerBlock,
          ck::index_t WoPerBlock,
          ck::index_t E0PerBlock,
          ck::index_t E1PerBlock,
          ck::index_t KPerThread,
          ck::index_t HoPerThread,
          ck::index_t WoPerThread,
          ck::index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
          typename ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
          ck::index_t ABlockTransferSrcScalarPerVector_E2,
          ck::index_t ABlockTransferDstScalarPerVector_E2,
          ck::index_t BThreadTransferSrcScalarPerVector_E2,
          ck::index_t CThreadTransferDstScalarPerVector_K,
          const ck::index_t group_count,
          ck::ActivTypeEnum_t activ_type>
struct DriverDynamicConvolutionBiasActivForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1
{
    template <typename... Wei,
              typename... In,
              typename... Out,
              typename ConvStrides,
              typename ConvDilations,
              typename InLeftPads,
              typename InRightPads>
    __host__ float Run(const ck::TensorDescriptor<Wei...>& wei_k_c0_y_x_c1_global_desc,
                       const ck::TensorDescriptor<In...>& in_n_c0_hi_wi_c1_global_desc,
                       const ck::TensorDescriptor<Out...>& out_n_k0_ho_wo_k1_global_desc,
                       const ConvStrides&,
                       const ConvDilations&,
                       const InLeftPads&,
                       const InRightPads&,
                       const FloatAB* __restrict__ p_a_grid,
                       const FloatAB* __restrict__ p_b_grid,
                       const FloatBias* __restrict__ p_bias_grid,
                       FloatC* __restrict__ p_c_grid,
                       const int nrepeat) const
    {
        using namespace ck;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N  = in_n_c0_hi_wi_c1_global_desc.GetLength(I0);
        const auto C0 = in_n_c0_hi_wi_c1_global_desc.GetLength(I1);
        const auto Hi = in_n_c0_hi_wi_c1_global_desc.GetLength(I2);
        const auto Wi = in_n_c0_hi_wi_c1_global_desc.GetLength(I3);
        // const auto C1 = in_n_c0_hi_wi_c1_global_desc.GetLength(I4);

        const auto K0 = out_n_k0_ho_wo_k1_global_desc.GetLength(I1);
        const auto Ho = out_n_k0_ho_wo_k1_global_desc.GetLength(I2);
        const auto Wo = out_n_k0_ho_wo_k1_global_desc.GetLength(I3);
        const auto K1 = out_n_k0_ho_wo_k1_global_desc.GetLength(I4);

        const auto K = wei_k_c0_y_x_c1_global_desc.GetLength(I0);
        const auto Y = wei_k_c0_y_x_c1_global_desc.GetLength(I2);
        const auto X = wei_k_c0_y_x_c1_global_desc.GetLength(I3);

        // const auto ConvStrideH = conv_strides[I0];
        // const auto ConvStrideW = conv_strides[I1];

        // const auto ConvDilationH = conv_dilations[I0];
        // const auto ConvDilationW = conv_dilations[I1];

#if CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR
        const auto Hop = Number<(Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock>{};
        const auto Wop = Number<(Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock>{};
#else
        const auto Hop = (Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock;
        const auto Wop = (Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock;
#endif

        const auto OutRightPadH = Hop - Ho;
        const auto OutRightPadW = Wop - Wo;

        // const auto InLeftPadH = in_left_pads[I0];
        // const auto InLeftPadW = in_left_pads[I1];

        const auto InRightPadH = OutRightPadH;
        const auto InRightPadW = OutRightPadW;

        constexpr auto E1 = Number<E1_>{};
        constexpr auto E2 = Number<E2_>{};
        constexpr auto K2 = Number<K2_>{};

        if((C0 * Y * X) % (E1 * E0PerBlock) != 0)
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        const auto E0 = (C0 * Y * X) / E1;

        // weight tensor
        const auto a_e_k_e2_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(K, C0 * Y * X, E2)),
            make_tuple(make_pass_through_transform(K),
                       make_pass_through_transform(C0 * Y * X),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}));

        const auto a_e0_e1_k_e2_grid_desc =
            transform_tensor_descriptor(a_e_k_e2_grid_desc,
                                        make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                                                   make_pass_through_transform(K),
                                                   make_pass_through_transform(E2)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

        // input tensor
        const auto in_n_c0_hip_wip_e2_global_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, E2)),
            make_tuple(make_pass_through_transform(N),
                       make_pass_through_transform(C0),
                       make_right_pad_transform(Hi, InRightPadH),
                       make_right_pad_transform(Wi, InRightPadW),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        const auto in_e_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
            in_n_c0_hip_wip_e2_global_desc,
            make_tuple(make_pass_through_transform(C0),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        const auto b_e0_e1_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
            in_e_n_ho_wo_e2_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

        // input2 tensor
        const auto in2_hip_wip_global_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi)),
            make_tuple(make_pad_transform(Hi, I1, Number<I1 + InRightPadH>{}),
                       make_pad_transform(Wi, I1, Number<I1 + InRightPadH>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto in2_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in2_hip_wip_global_desc,
            make_tuple(make_embed_transform(make_tuple(I3, Hop), make_tuple(I1, I1)),
                       make_embed_transform(make_tuple(I3, Wop), make_tuple(I1, I1))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        const auto in2_e_ho_wo_grid_desc =
            transform_tensor_descriptor(in2_y_ho_x_wo_global_desc,
                                        make_tuple(make_merge_transform(make_tuple(I3, I3)),
                                                   make_pass_through_transform(Hop),
                                                   make_pass_through_transform(Wop)),
                                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        const auto d_e_ho_wo_grid_desc =
            transform_tensor_descriptor(in2_e_ho_wo_grid_desc,
                                        make_tuple(make_right_pad_transform(Number<9>{}, I1),
                                                   make_pass_through_transform(Hop),
                                                   make_pass_through_transform(Wop)),
                                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        std::cout << "in2: " << d_e_ho_wo_grid_desc.GetLength(I0) << ", "
                  << d_e_ho_wo_grid_desc.GetLength(I1) << ", " << d_e_ho_wo_grid_desc.GetLength(I2)
                  << std::endl;

        // output tensor
        const auto c_k_n_hop_wop_grid_desc = transform_tensor_descriptor(
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
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

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

        const auto a_e0_e1_k0_k1_e2_grid_desc =
            GridwiseGemm::MakeAE0E1K0K1E2GridDescriptor(a_e0_e1_k_e2_grid_desc);
        const auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
            GridwiseGemm::MakeBE0E1NH0H1H2W0W1W2E2GridDescriptor(b_e0_e1_n_ho_wo_e2_grid_desc);
        const auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc =
            GridwiseGemm::MakeCK0K1NH0H1H2W0W1W2GridDescriptor(c_k_n_hop_wop_grid_desc);

        using AGridDesc_E0_E1_K0_K1_E2 = decltype(a_e0_e1_k0_k1_e2_grid_desc);
        using BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 =
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc);
        using CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 = decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

        const auto grid_size = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;

        const bool has_main_e0_block_loop = E0 > 1;

        std::cerr << "has_main_e0_block_loop = " << has_main_e0_block_loop << std::endl;

        const auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
            GridwiseGemm::MakeCBlockIdToKNHoWoBlockClusterAdaptor(c_k_n_hop_wop_grid_desc);

        using CBlockIdToBlockClusterAdaptor_K_N_H_W =
            decltype(c_blockid_to_k_n_h_w_block_cluster_adaptor);

        float ave_time = 0;

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
        const auto kernel =
            kernel_gemm_dlops_v3<GridwiseGemm,
                                 FloatAB,
                                 FloatC,
                                 remove_reference_t<AGridDesc_E0_E1_K0_K1_E2>,
                                 remove_reference_t<BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2>,
                                 remove_reference_t<CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>,
                                 remove_reference_t<CBlockIdToBlockClusterAdaptor_K_N_H_W>,
                                 true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_e0_e1_k0_k1_e2_grid_desc,
                                          b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                          c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                                          c_blockid_to_k_n_h_w_block_cluster_adaptor);
#elif CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR
        static_assert(a_e0_e1_k_e2_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.IsKnownAtCompileTime(), "");
        static_assert(c_blockid_to_k_n_h_w_block_cluster_adaptor.IsKnownAtCompileTime(), "");

        const auto kernel =
            kernel_gemm_softmax_dlops_v3<GridwiseGemm,
                                         FloatAB,
                                         FloatAcc,
                                         FloatBias,
                                         FloatC,
                                         remove_reference_t<AGridDesc_E0_E1_K0_K1_E2>,
                                         remove_reference_t<BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2>,
                                         remove_reference_t<CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>,
                                         remove_reference_t<CBlockIdToBlockClusterAdaptor_K_N_H_W>,
                                         has_main_e0_block_loop,
                                         activ_type>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_bias_grid,
                                          p_c_grid,
                                          0.3);
#endif
        return ave_time;
    }
};
#endif
