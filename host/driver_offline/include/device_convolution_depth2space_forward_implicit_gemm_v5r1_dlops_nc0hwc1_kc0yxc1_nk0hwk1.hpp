#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_convolution_depth2space_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp"
#include "ck_conv_fig.h"

template <typename TInWei,
          typename TAcc,
          typename TBias,
          typename TOut,
          ck::ActivTypeEnum_t activ_type,
          typename InLengths,
          typename WeiLengths,
          typename MaxLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_depth2space_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1(
    const InLengths& in_n_c0_hi_wi_c1_lengths,
    const WeiLengths& wei_k_c0_y_x_c1_lengths,
    const MaxLengths& dp2sp_n_k0_hx_wx_k1_lengths,
    const OutLengths& out_n_k0_ho_wo_k1_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_c0_hi_wi_c1,
    const Tensor<TInWei>& wei_k_c0_y_x_c1,
    const Tensor<TBias>& bias_k0_k1,
    Tensor<TOut>& out_n_k0_ho_wo_k1,
    Tensor<TOut>& dp2sp_n_k0_hx_wx_k1,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    const auto N  = out_n_k0_ho_wo_k1_lengths[I0];
    const auto K0 = out_n_k0_ho_wo_k1_lengths[I1];
    const auto Ho = out_n_k0_ho_wo_k1_lengths[I2];
    const auto Wo = out_n_k0_ho_wo_k1_lengths[I3];
    const auto K1 = out_n_k0_ho_wo_k1_lengths[I4];

    const auto C0 = in_n_c0_hi_wi_c1_lengths[I1];
    const auto Hi = in_n_c0_hi_wi_c1_lengths[I2];
    const auto Wi = in_n_c0_hi_wi_c1_lengths[I3];
    const auto C1 = in_n_c0_hi_wi_c1_lengths[I4];

    const auto K = wei_k_c0_y_x_c1_lengths[I0];
    const auto Y = wei_k_c0_y_x_c1_lengths[I2];
    const auto X = wei_k_c0_y_x_c1_lengths[I3];

    const auto Hx  = dp2sp_n_k0_hx_wx_k1_lengths[I2];
    const auto Wx  = dp2sp_n_k0_hx_wx_k1_lengths[I3];
    const auto K1x = dp2sp_n_k0_hx_wx_k1_lengths[I4];

    DeviceMem in_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                          in_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem wei_k_c0_y_x_c1_device_buf(sizeof(TInWei) * wei_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem bias_k0_k1_device_buf(sizeof(TBias) * bias_k0_k1.mDesc.GetElementSpace());
    DeviceMem out_n_k0_ho_wo_k1_device_buf(sizeof(TOut) *
                                           out_n_k0_ho_wo_k1.mDesc.GetElementSpace());
    DeviceMem dp2sp_n_k0_hx_wx_k1_device_buf(sizeof(TOut) *
                                             dp2sp_n_k0_hx_wx_k1.mDesc.GetElementSpace());

    in_n_c0_hi_wi_c1_device_buf.ToDevice(in_n_c0_hi_wi_c1.mData.data());
    wei_k_c0_y_x_c1_device_buf.ToDevice(wei_k_c0_y_x_c1.mData.data());
    bias_k0_k1_device_buf.ToDevice(bias_k0_k1.mData.data());
    dp2sp_n_k0_hx_wx_k1_device_buf.ToDevice(dp2sp_n_k0_hx_wx_k1.mData.data());

    constexpr index_t InWeiVectorSize = 8;

    if(C1 % InWeiVectorSize != 0)
    {
        throw std::runtime_error("wrong! C1 cannot be divided by InWeiVectorSize");
    }

#if USE_CONV_FIG
    constexpr index_t BlockSize = CONV_BLOCK_SIZE;

    constexpr index_t E1 = CONV_E1;
    constexpr index_t E2 = CONV_E2;
    constexpr index_t K2 = CONV_K2;

    constexpr index_t E0PerBlock = CONV_E0_PER_BLOCK;
    constexpr index_t KPerBlock  = CONV_K_PER_BLOCK;
    constexpr index_t HoPerBlock = CONV_HO_PER_BLOCK;
    constexpr index_t WoPerBlock = CONV_WO_PER_BLOCK;
    constexpr index_t E1PerBlock = CONV_E1_PER_BLOCK;

    constexpr index_t KPerThread  = CONV_K_PER_THREAD;
    constexpr index_t HoPerThread = CONV_HO_PER_THREAD;
    constexpr index_t WoPerThread = CONV_WO_PER_THREAD;
    constexpr index_t EPerThread  = CONV_E_PER_THREAD;

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2 =
        Sequence<CONV_ABLOCK_TRANS_THREAD_SLICE_LENGTHS>;
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 =
        Sequence<CONV_ABLOCK_TRANS_THREAD_CLUSTER_LENGTHS>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E2  = C1;
    constexpr index_t ABlockTransferDstScalarPerVector_E2  = C1;
    constexpr index_t BThreadTransferSrcScalarPerVector_E2 = C1;
    constexpr index_t CThreadTransferDstScalarPerVector_K  = K1x;
#elif 0
    constexpr index_t BlockSize = 256;

    constexpr index_t E1 = C0 * Y * X;
    constexpr index_t E2 = C1;
    constexpr index_t K2 = 4;

    constexpr index_t E0PerBlock = 1;
    constexpr index_t KPerBlock  = 16;
    constexpr index_t HoPerBlock = 16;
    constexpr index_t WoPerBlock = 64;
    constexpr index_t E1PerBlock = 2;

    constexpr index_t KPerThread  = 16;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = 1;

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2   = Sequence<1, Y * X, 1, 1, C1>;
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 = Sequence<1, C0, 1, KPerBlock, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E2  = C1;
    constexpr index_t ABlockTransferDstScalarPerVector_E2  = C1;
    constexpr index_t BThreadTransferSrcScalarPerVector_E2 = C1;
    constexpr index_t CThreadTransferDstScalarPerVector_K  = K1;
#elif 0
    constexpr index_t BlockSize = 128;

    constexpr index_t E0PerBlock = 1;
    constexpr index_t KPerBlock  = 16;
    constexpr index_t HoPerBlock = 8;
    constexpr index_t WoPerBlock = 64;
    constexpr index_t E1PerBlock = 2;

    constexpr index_t E1 = C0 * Y * X;
    constexpr index_t E2 = C1;
    constexpr index_t K2 = 4;

    constexpr index_t KPerThread  = 16;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = 1;

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2   = Sequence<1, 7, 1, 1, C1>;
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 = Sequence<1, 2, 1, KPerBlock, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E2  = C1;
    constexpr index_t ABlockTransferDstScalarPerVector_E2  = C1;
    constexpr index_t BThreadTransferSrcScalarPerVector_E2 = C1;
    constexpr index_t CThreadTransferDstScalarPerVector_K  = K1;
#elif 0
    constexpr index_t BlockSize = 64;

    constexpr index_t E1 = C0 * Y * X;
    constexpr index_t E2 = C1;
    constexpr index_t K2 = 4;

    constexpr index_t E0PerBlock = 1;
    constexpr index_t KPerBlock  = 16;
    constexpr index_t HoPerBlock = 8;
    constexpr index_t WoPerBlock = 32;
    constexpr index_t E1PerBlock = 1;

    constexpr index_t KPerThread  = 16;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = 1;

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2   = Sequence<1, 16 * Y * X, 1, 1, C1>;
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 = Sequence<1, 4, 1, KPerBlock, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E2  = C1;
    constexpr index_t ABlockTransferDstScalarPerVector_E2  = C1;
    constexpr index_t BThreadTransferSrcScalarPerVector_E2 = C1;
    constexpr index_t CThreadTransferDstScalarPerVector_K  = K1x;
#endif

    if(KPerThread % InWeiVectorSize != 0)
    {
        throw std::runtime_error("wrong! C1 cannot be divided by InWeiVectorSize");
    }

    const auto in_n_c0_hi_wi_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, E2));
    const auto wei_k_c0_y_x_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(K, C0, Y, X, E2));
    const auto dp2sp_n_k0_hx_wx_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Hx, Wx, K1x));
    const auto out_n_k0_ho_wo_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1));

    constexpr auto conv_driver =
        DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1_depth2space<
            BlockSize,
            TInWei,
            TAcc,
            TBias,
            TOut,
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
            ABlockTransferSrcScalarPerVector_E2,
            ABlockTransferDstScalarPerVector_E2,
            BThreadTransferSrcScalarPerVector_E2,
            CThreadTransferDstScalarPerVector_K,
            activ_type>{};

    std::cerr << "input_"
              << "n" << N << "c" << C0 << "h" << Hi << "w" << Wi << "c" << C1 << "_filter_k" << K
              << "c" << C0 << "y" << Y << "x" << X << "c" << C1 << "_convout_n" << N << "k" << K0
              << "h" << Ho << "w" << Wo << "k" << K1 << "_depth2space_out_n" << N << "k" << K0
              << "h" << Hx << "w" << Wx << "k" << K1x << std::endl;

    std::cerr << "BlockSize_" << BlockSize << "_E1_" << E1 << "_E2_" << E2 << "_K2_" << K2
              << "_KPerBlock_" << KPerBlock << "_HoPerBlock_" << HoPerBlock << "_WoPerBlock_"
              << WoPerBlock << "_E0PerBlock_" << E0PerBlock << "_E1PerBlock_" << E1PerBlock
              << "_KPerThread_" << KPerThread << "_HoPerThread_" << HoPerThread << "_WoPerThread_"
              << WoPerThread << "_EPerThread_" << EPerThread << std::endl;

    for(int i = 0; i < 5; i++)
    {
        const auto ave_time =
            conv_driver.Run(wei_k_c0_y_x_c1_desc,
                            in_n_c0_hi_wi_c1_desc,
                            out_n_k0_ho_wo_k1_desc,
                            dp2sp_n_k0_hx_wx_k1_desc,
                            conv_strides,
                            conv_dilations,
                            in_left_pads,
                            in_right_pads,
                            static_cast<TInWei*>(wei_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TInWei*>(in_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TBias*>(bias_k0_k1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(out_n_k0_ho_wo_k1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(dp2sp_n_k0_hx_wx_k1_device_buf.GetDeviceBuffer()),
                            nrepeat);

        {
            float perf = static_cast<float>(std::size_t(2) * N * K * Ho * Wo * C0 * C1 * Y * X) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }

    out_n_k0_ho_wo_k1_device_buf.FromDevice(out_n_k0_ho_wo_k1.mData.data());
    dp2sp_n_k0_hx_wx_k1_device_buf.FromDevice(dp2sp_n_k0_hx_wx_k1.mData.data());
}
