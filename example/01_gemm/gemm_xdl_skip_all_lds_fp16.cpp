#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_gemm_xdl_skip_b_lds.hpp"
#include "device_gemm_xdl_skip_all_lds.hpp"
#include "device_gemm_xdl.hpp"
#include "device_gemm_xdl_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;


// clang-format off
using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmXdlSkipAllLds
        //###########| AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|          GEMM| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds| BThreadTransfer| CThreadTransfer| CThreadTransfer|
        //###########|  Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise|Spacialization|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|       SrcScalar| SrcDstVectorDim|       DstScalar|
        //###########|      |      |      |        |        |        |        |   Operation|   Operation|   Operation|              |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |       PerVector|                |       PerVector|
        //###########|      |      |      |        |        |        |        |            |            |            |              |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |                |                |
    
                    //<   F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   64,    16,   64,     4,  8,   16,   16,    1,    4,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,            8,               7,               1>;
                    <   F16,   F16,   F16,     F32,     Row,     Col,     Row, PassThrough, PassThrough, PassThrough,   GemmDefault,   64,    16,   16,     4,  8,   16,   16,    1,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true,            8,               7,               1>;
using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

    // clang-format on

    using ReferenceGemmInstance = ck::tensor_operation::host::
        ReferenceGemm<ADataType, BDataType, CDataType, float, AElementOp, BElementOp, CElementOp>;

template <typename DataType>
std::ostream& show_2d_matrix(std::ostream& os, Tensor<DataType>& matrix)
{
    os << "[" << std::endl;
    for(size_t x = 0; x < matrix.mDesc.GetLengths()[0]; x++)
    {
        os << "[";
        for(size_t y = 0; y < matrix.mDesc.GetLengths()[1]; y++)
        {
            os << std::setw(5) << static_cast<float>(matrix(x, y));
        }
        os << "]" << std::endl;
    }
    os << "]";
    return os;
}
int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    bool time_kernel     = false;

    // GEMM shape
#if 1
    ck::index_t M = 128 * 360;
    ck::index_t N = 64;
    ck::index_t K = 4096;

    ck::index_t StrideA = 4096;
    ck::index_t StrideB = 4096;
    ck::index_t StrideC = N;
#else
    ck::index_t M = 16;
    ck::index_t N = 16;
    ck::index_t K = 8;

    ck::index_t StrideA = 8;
    ck::index_t StrideB = 8;
    ck::index_t StrideC = 16;
#endif

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideC = std::stoi(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=n0, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideC\n");
        exit(0);
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        //a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        //a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        a_m_k.GenerateTensorValue(GeneratorTensor_2<BDataType>{0, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{0, 2});
        //b_k_n.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    default:
        // a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
    }

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpace());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

    if(do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

#if 0
        {
            show_2d_matrix(std::cout << "a : ", a_m_k) << std::endl;
            show_2d_matrix(std::cout << "b: ", b_k_n) << std::endl;
            show_2d_matrix(std::cout << "c_device: ", c_m_n_device_result) << std::endl;
            show_2d_matrix(std::cout << "c_host  :", c_m_n_host_result) << std::endl;
        }
#endif
        ck::utils::check_err(c_m_n_device_result.mData, c_m_n_host_result.mData);
    }

    return 0;
}
