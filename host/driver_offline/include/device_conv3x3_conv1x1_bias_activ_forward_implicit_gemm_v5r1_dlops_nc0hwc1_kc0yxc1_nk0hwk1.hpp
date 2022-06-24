#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_conv3x3_conv1x1_bias_activ_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp"
#include "ck_conv_fig.h"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

template <ck::index_t BlockSize_,
          ck::index_t E1_,
          ck::index_t E2_,
          ck::index_t K2_,
          ck::index_t E0PerBlock_,
          ck::index_t KPerBlock_,
          ck::index_t HoPerBlock_,
          ck::index_t WoPerBlock_,
          ck::index_t E1PerBlock_,
          ck::index_t KPerThread_,
          ck::index_t HoPerThread_,
          ck::index_t WoPerThread_,
          ck::index_t EPerThread_,
          typename ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2_,
          typename ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2_,
          ck::index_t ABlockTransferSrcScalarPerVector_E2_,
          ck::index_t ABlockTransferDstScalarPerVector_E2_,
          ck::index_t BThreadTransferSrcScalarPerVector_E2_,
          ck::index_t CThreadTransferDstScalarPerVector_K_>
struct GridGemmTuningParameters
{
    static constexpr auto BlockSize = BlockSize_;
    static constexpr auto E1        = E1_;
    static constexpr auto E2        = E2_;
    static constexpr auto K2        = K2_;

    static constexpr auto E0PerBlock = E0PerBlock_;
    static constexpr auto KPerBlock  = KPerBlock_;
    static constexpr auto HoPerBlock = HoPerBlock_;
    static constexpr auto WoPerBlock = WoPerBlock_;
    static constexpr auto E1PerBlock = E1PerBlock_;

    static constexpr auto KPerThread  = KPerThread_;
    static constexpr auto HoPerThread = HoPerThread_;
    static constexpr auto WoPerThread = WoPerThread_;
    static constexpr auto EPerThread  = EPerThread_;

    static constexpr auto ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2 =
        ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2_{};
    static constexpr auto ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 =
        ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2_{};

    static constexpr auto ABlockTransferSrcScalarPerVector_E2 =
        ABlockTransferSrcScalarPerVector_E2_;
    static constexpr auto ABlockTransferDstScalarPerVector_E2 =
        ABlockTransferDstScalarPerVector_E2_;
    static constexpr auto BThreadTransferSrcScalarPerVector_E2 =
        BThreadTransferSrcScalarPerVector_E2_;
    static constexpr auto CThreadTransferDstScalarPerVector_K =
        CThreadTransferDstScalarPerVector_K_;

    void printTuningParameters()
    {
        using namespace ck;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        std::cout << "BlockSize_" << BlockSize << "_E1_" << E1 << "_E2_" << E2 << "_K2_" << K2
                  << "_KPerBlock_" << KPerBlock << "_HoPerBlock_" << HoPerBlock << "_WoPerBlock_"
                  << WoPerBlock << "_E0PerBlock_" << E0PerBlock << "_E1PerBlock_" << E1PerBlock
                  << "_KPerThread_" << KPerThread << "_HoPerThread_" << HoPerThread
                  << "_WoPerThread_" << WoPerThread << "_EPerThread_" << EPerThread
                  << "_ABlockTransferThreadSliceLengths_<"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I0] << "_"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I1] << "_"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I2] << "_"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I3] << "_"
                  << ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2[I4] << ">"
                  << "_ABlockTransferThreadClusterLengths_<"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I0] << "_"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I1] << "_"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I2] << "_"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I3] << "_"
                  << ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2[I4] << ">"
                  << "_ABlockTransferSrcScalarPerVector_E2_" << ABlockTransferSrcScalarPerVector_E2
                  << "_ABlockTransferDstScalarPerVector_E2_" << ABlockTransferDstScalarPerVector_E2
                  << "_BThreadTransferSrcScalarPerVector_E2_"
                  << BThreadTransferSrcScalarPerVector_E2 << "_CThreadTransferDstScalarPerVector_K_"
                  << CThreadTransferDstScalarPerVector_K << std::endl;
    }
};

template <typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
struct ConvDesc
{
    InLengths in_n_c0_hi_wi_c1_desc;
    WeiLengths wei_c0_y_x_k_c1_desc;
    OutLengths out_n_k0_ho_wo_k1_desc;

    ConvStrides conv_strides;
    ConvDilations conv_dilations;

    InLeftPads in_left_pads;
    InRightPads in_right_pads;

    ConvDesc(InLengths in_n_c0_hi_wi_c1_desc_,
             WeiLengths wei_c0_y_x_k_c1_desc_,
             OutLengths out_n_k0_ho_wo_k1_desc_,
             ConvStrides conv_strides_,
             ConvDilations conv_dilations_,
             InLeftPads in_left_pads_,
             InRightPads in_right_pads_)
    {
        in_n_c0_hi_wi_c1_desc  = in_n_c0_hi_wi_c1_desc_;
        wei_c0_y_x_k_c1_desc   = wei_c0_y_x_k_c1_desc_;
        out_n_k0_ho_wo_k1_desc = out_n_k0_ho_wo_k1_desc_;
        conv_strides           = conv_strides_;
        conv_dilations         = conv_dilations_;
        in_left_pads           = in_left_pads_;
        in_right_pads          = in_right_pads_;
    }

    void printConvDesc()
    {
        using namespace ck;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N  = in_n_c0_hi_wi_c1_desc.GetLength(I0);
        const auto C0 = in_n_c0_hi_wi_c1_desc.GetLength(I1);
        const auto Hi = in_n_c0_hi_wi_c1_desc.GetLength(I2);
        const auto Wi = in_n_c0_hi_wi_c1_desc.GetLength(I3);
        const auto C1 = in_n_c0_hi_wi_c1_desc.GetLength(I4);

        const auto K0 = out_n_k0_ho_wo_k1_desc.GetLength(I1);
        const auto Ho = out_n_k0_ho_wo_k1_desc.GetLength(I2);
        const auto Wo = out_n_k0_ho_wo_k1_desc.GetLength(I3);
        const auto K1 = out_n_k0_ho_wo_k1_desc.GetLength(I4);

        const auto K = wei_c0_y_x_k_c1_desc.GetLength(I0);
        const auto Y = wei_c0_y_x_k_c1_desc.GetLength(I2);
        const auto X = wei_c0_y_x_k_c1_desc.GetLength(I3);

        const auto ConvStrideH = conv_strides[I0];
        const auto ConvStrideW = conv_strides[I1];

        const auto ConvDilationH = conv_dilations[I0];
        const auto ConvDilationW = conv_dilations[I1];

        std::cout << "input_"
                  << "n" << N << "c" << C0 << "h" << Hi << "w" << Wi << "c" << C1 << "_filter_k"
                  << K << "c" << C0 << "y" << Y << "x" << X << "c" << C1 << "_out_n" << N << "k"
                  << K0 << "h" << Ho << "w" << Wo << "k" << K1 << std::endl;

        std::cout << "ConvStride = " << ConvStrideH << "," << ConvStrideW << std::endl;
        std::cout << "ConvDilation = " << ConvDilationH << "," << ConvDilationW << std::endl;
    }
};

template <typename TInWei,
          typename TAcc,
          typename TOut,
          typename In1Lengths,
          typename Wei1Lengths,
          typename Out1Lengths,
          typename In2Lengths,
          typename Wei2Lengths,
          typename Out2Lengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_conv3x3_conv1x1_bias_activ_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1(
    const In1Lengths& in1_n_c0_hi_wi_c1_lengths,
    const Wei1Lengths& wei1_k_c0_y_x_c1_lengths,
    const Out1Lengths& out1_n_k0_ho_wo_k1_lengths,
    const In2Lengths& in2_n_c0_hi_wi_c1_lengths,
    const Wei2Lengths& wei2_k_c0_y_x_c1_lengths,
    const Out2Lengths& out2_n_k0_ho_wo_k1_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in1_n_c0_hi_wi_c1,
    const Tensor<TInWei>& wei1_k_c0_y_x_c1,
    const Tensor<TOut>& bias1_k0_k1,
    Tensor<TOut>& out1_n_k0_ho_wo_k1,
    const Tensor<TInWei>& wei2_k_c0_y_x_c1,
    const Tensor<TOut>& bias2_k0_k1,
    Tensor<TOut>& out2_n_k0_ho_wo_k1,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    const auto CONV1_N  = out1_n_k0_ho_wo_k1_lengths[I0];
    const auto CONV1_K0 = out1_n_k0_ho_wo_k1_lengths[I1];
    const auto CONV1_Ho = out1_n_k0_ho_wo_k1_lengths[I2];
    const auto CONV1_Wo = out1_n_k0_ho_wo_k1_lengths[I3];
    const auto CONV1_K1 = out1_n_k0_ho_wo_k1_lengths[I4];

    const auto CONV1_C0 = in1_n_c0_hi_wi_c1_lengths[I1];
    const auto CONV1_Hi = in1_n_c0_hi_wi_c1_lengths[I2];
    const auto CONV1_Wi = in1_n_c0_hi_wi_c1_lengths[I3];
    const auto CONV1_C1 = in1_n_c0_hi_wi_c1_lengths[I4];

    const auto CONV1_K = wei1_k_c0_y_x_c1_lengths[I0];
    const auto CONV1_Y = wei1_k_c0_y_x_c1_lengths[I2];
    const auto CONV1_X = wei1_k_c0_y_x_c1_lengths[I3];

    const auto CONV2_N  = out2_n_k0_ho_wo_k1_lengths[I0];
    const auto CONV2_K0 = out2_n_k0_ho_wo_k1_lengths[I1];
    const auto CONV2_Ho = out2_n_k0_ho_wo_k1_lengths[I2];
    const auto CONV2_Wo = out2_n_k0_ho_wo_k1_lengths[I3];
    const auto CONV2_K1 = out2_n_k0_ho_wo_k1_lengths[I4];

    const auto CONV2_C0 = in2_n_c0_hi_wi_c1_lengths[I1];
    const auto CONV2_Hi = in2_n_c0_hi_wi_c1_lengths[I2];
    const auto CONV2_Wi = in2_n_c0_hi_wi_c1_lengths[I3];
    const auto CONV2_C1 = in2_n_c0_hi_wi_c1_lengths[I4];

    const auto CONV2_K = wei2_k_c0_y_x_c1_lengths[I0];
    const auto CONV2_Y = wei2_k_c0_y_x_c1_lengths[I2];
    const auto CONV2_X = wei2_k_c0_y_x_c1_lengths[I3];

    DeviceMem in1_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                           in1_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem wei1_k_c0_y_x_c1_device_buf(sizeof(TInWei) *
                                          wei1_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem bias1_k0_k1_device_buf(sizeof(TOut) * bias1_k0_k1.mDesc.GetElementSpace());
    DeviceMem out1_n_k0_ho_wo_k1_device_buf(sizeof(TOut) *
                                            out1_n_k0_ho_wo_k1.mDesc.GetElementSpace());

    DeviceMem wei2_k_c0_y_x_c1_device_buf(sizeof(TInWei) *
                                          wei2_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem bias2_k0_k1_device_buf(sizeof(TOut) * bias2_k0_k1.mDesc.GetElementSpace());
    DeviceMem out2_n_k0_ho_wo_k1_device_buf(sizeof(TOut) *
                                            out2_n_k0_ho_wo_k1.mDesc.GetElementSpace());

    in1_n_c0_hi_wi_c1_device_buf.ToDevice(in1_n_c0_hi_wi_c1.mData.data());
    wei1_k_c0_y_x_c1_device_buf.ToDevice(wei1_k_c0_y_x_c1.mData.data());
    bias1_k0_k1_device_buf.ToDevice(bias1_k0_k1.mData.data());

    wei2_k_c0_y_x_c1_device_buf.ToDevice(wei2_k_c0_y_x_c1.mData.data());
    bias2_k0_k1_device_buf.ToDevice(bias2_k0_k1.mData.data());

    GridGemmTuningParameters<256,                          // BlockSize
                             CONV1_C0 * CONV1_Y * CONV1_X, // E1
                             CONV1_C1,                     // E2
                             2,                            // K2
                             1,                            // E0PerBlock
                             CONV1_K,                      // KPerBlock
                             16,                           // HoPerBlock
                             64,                           // WoPerBlock
                             1,                            // E1PerBlock
                             CONV1_K,                      // KPerThread
                             2,                            // HoPerThread
                             2,                            // WoPerThread
                             1,                            // EPerThread
                             Sequence<1,
                                      CONV1_C0 * CONV1_Y * CONV1_X,
                                      1,
                                      CONV1_K,
                                      CONV1_C1>, // ABlockTransferBlockSliceLengths_E0_E1_K0_K1_E2
                             Sequence<1,
                                      CONV1_C0,
                                      1,
                                      CONV1_K,
                                      1>, // ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2
                             CONV1_C1,    // ABlockTransferSrcScalarPerVector_E2
                             CONV1_C1,    // ABlockTransferDstScalarPerVector_E2
                             CONV1_C1,    // BThreadTransferSrcScalarPerVector_E2
                             CONV1_K1     // CThreadTransferDstScalarPerVector_K
                             >
        conv1_tuning_parameters{};

    GridGemmTuningParameters<256,                          // BlockSize
                             CONV2_C0 * CONV2_Y * CONV2_X, // E1
                             CONV2_C1,                     // E2
                             2,                            // K2
                             1,                            // E0PerBlock
                             CONV2_K,                      // KPerBlock
                             16,                           // HoPerBlock
                             64,                           // WoPerBlock
                             CONV2_C0 * CONV2_Y * CONV2_X, // E1PerBlock
                             CONV2_K,                      // KPerThread
                             2,                            // HoPerThread
                             2,                            // WoPerThread
                             1,                            // EPerThread
                             Sequence<1,
                                      CONV2_C0 * CONV2_Y * CONV2_X,
                                      1,
                                      CONV2_K,
                                      CONV2_C1>, // ABlockTransferBlockSliceLengths_E0_E1_K0_K1_E2
                             Sequence<1,
                                      CONV2_C0,
                                      1,
                                      CONV2_K,
                                      1>, // ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2
                             CONV2_C1,    // ABlockTransferSrcScalarPerVector_E2
                             CONV2_C1,    // ABlockTransferDstScalarPerVector_E2
                             CONV2_C1,    // BThreadTransferSrcScalarPerVector_E2
                             CONV2_K1     // CThreadTransferDstScalarPerVector_K
                             >
        conv2_tuning_parameters{};

    const auto in1_n_c0_hi_wi_c1_desc = make_naive_tensor_descriptor_packed(
        make_tuple(CONV1_N, CONV1_C0, CONV1_Hi, CONV1_Wi, CONV1_C1));
    const auto wei1_k_c0_y_x_c1_desc = make_naive_tensor_descriptor_packed(
        make_tuple(CONV1_K, CONV1_C0, CONV1_Y, CONV1_X, CONV1_C1));
    const auto out1_n_k0_ho_wo_k1_desc = make_naive_tensor_descriptor_packed(
        make_tuple(CONV1_N, CONV1_K0, CONV1_Ho, CONV1_Wo, CONV1_K1));

    ConvDesc conv1_desc(in1_n_c0_hi_wi_c1_desc,
                        wei1_k_c0_y_x_c1_desc,
                        out1_n_k0_ho_wo_k1_desc,
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads);

    const auto in2_n_c0_hi_wi_c1_desc = make_naive_tensor_descriptor_packed(
        make_tuple(CONV2_N, CONV2_C0, CONV2_Hi, CONV2_Wi, CONV2_C1));
    const auto wei2_k_c0_y_x_c1_desc = make_naive_tensor_descriptor_packed(
        make_tuple(CONV2_K, CONV2_K0, CONV2_Y, CONV2_X, CONV2_K1));
    const auto out2_n_k0_ho_wo_k1_desc = make_naive_tensor_descriptor_packed(
        make_tuple(CONV2_N, CONV2_K0, CONV2_Ho, CONV2_Wo, CONV2_K1));

    ConvDesc conv2_desc(in2_n_c0_hi_wi_c1_desc,
                        wei2_k_c0_y_x_c1_desc,
                        out2_n_k0_ho_wo_k1_desc,
                        make_tuple(I1, I1),
                        make_tuple(I1, I1),
                        make_tuple(I0, I0),
                        make_tuple(I0, I0));

#if 1
    constexpr auto conv_driver =
        DriverDynamicConv3x3Conv1x1BiasActivForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1<
            TInWei,
            TAcc,
            TOut,
            decltype(conv1_tuning_parameters),
            decltype(conv2_tuning_parameters)>{};

    for(int i = 0; i < 5; i++)
    {
        const auto ave_time =
            conv_driver.Run(conv1_desc,
                            static_cast<TInWei*>(wei1_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TInWei*>(in1_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(bias1_k0_k1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(out1_n_k0_ho_wo_k1_device_buf.GetDeviceBuffer()),
                            conv2_desc,
                            static_cast<TInWei*>(wei2_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(bias2_k0_k1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(out2_n_k0_ho_wo_k1_device_buf.GetDeviceBuffer()),
                            nrepeat);

        {
            // float perf =
            // static_cast<float>(std::size_t(2) * (CONV2_N * CONV2_K * CONV2_Ho * CONV2_Wo *
            // CONV2_C0 * CONV2_C1 * CONV2_Y * CONV2_X) +
            // std::size_t(2) * (CONV1_N * CONV1_K * CONV1_Ho * CONV1_Wo *
            // CONV1_C0 * CONV1_C1 * CONV1_Y * CONV1_X)) /
            //(std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms" << std::endl;
        }
    }
#endif

    out1_n_k0_ho_wo_k1_device_buf.FromDevice(out1_n_k0_ho_wo_k1.mData.data());
    out2_n_k0_ho_wo_k1_device_buf.FromDevice(out2_n_k0_ho_wo_k1.mData.data());
}
