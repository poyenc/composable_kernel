#ifndef CK_GRIDWISE_GEMM_V3_HPP
#define CK_GRIDWISE_GEMM_V3_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_gemm_dlops_v3.hpp"
#include "blockwise_tensor_slice_transfer_v4r1.hpp"
#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename T>
__host__ __device__ inline T max_(T a, T b)
{
    return a > b ? a : b;
}

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v3(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AGridDesc_E0_E1_K0_K1_E2 a_e0_e1_k0_k1_e2_grid_desc,
            const BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
            const CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
            const CBlockIdToBlockClusterAdaptor_K_N_H_W c_blockid_to_k_n_h_w_block_cluster_adaptor)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];
}
#elif CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v3(const FloatAB* __restrict__ p_a_grid,
                             const FloatAB* __restrict__ p_b_grid,
                             FloatC* __restrict__ p_c_grid)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};

    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

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

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    GridwiseGemm::WriteOut(c_thread_buf,
                           c_global_buf,
                           c_k_n_h_w_block_cluster_idx,
                           c_thread_mtx_index,
                           c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                           ck::tensor_operation::element_wise::PassThrough{});
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          // typename FloatC,
          typename FloatD,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename DGridDesc_K0_K1x_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v3_int4(const FloatAB* __restrict__ p_a_grid,
                                  const FloatAB* __restrict__ p_b_grid,
                                  // FloatC* __restrict__ p_c_grid,
                                  FloatD* __restrict__ p_d_grid)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    // constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};

    constexpr auto d_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc = DGridDesc_K0_K1x_N_H0_H1_H2_W0_W1_W2{};

    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

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

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    // auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
    // p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    // GridwiseGemm::WriteOut(c_thread_buf,
    // c_global_buf,
    // c_k_n_h_w_block_cluster_idx,
    // c_thread_mtx_index,
    // c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
    // ck::tensor_operation::element_wise::PassThrough{});

    constexpr auto d_k1x_n_h2_w2_thread_gemm_desc = GridwiseGemm::MakeDK1xNH2W2ThreadDescriptor();

    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatD,
                 d_k1x_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        d_thread_buf;

    GridwiseGemm::RequantInt4Op(
        c_thread_buf, d_thread_buf, c_k1_n_h2_w2_thread_gemm_desc, d_k1x_n_h2_w2_thread_gemm_desc);

    auto d_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_d_grid, d_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    GridwiseGemm::WriteOutInt4(d_thread_buf,
                               d_global_buf,
                               c_k_n_h_w_block_cluster_idx,
                               c_thread_mtx_index,
                               d_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc,
                               ck::tensor_operation::element_wise::PassThrough{});
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatScale,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          ActivTypeEnum_t ActivType>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v3_resize_add_per_channel(const FloatAB* __restrict__ p_a_grid,
                                                    const FloatAB* __restrict__ p_b_grid,
                                                    const FloatBias* __restrict__ p_bias_grid,
                                                    const FloatScale* __restrict__ p_scale_grid,
                                                    FloatC* __restrict__ p_d_grid)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};
    constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc = DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx{};
    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

    // static constexpr auto activ_type = integral_constant<ActivTypeEnum_t, ActivType>{};

    const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_a_grid, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
    const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_b_grid, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());

    constexpr auto c_k1_n_h2_w2_thread_gemm_desc = GridwiseGemm::MakeCK1NH2W2ThreadDescriptor();

    // register allocation for output
    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatAcc,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        c_thread_buf;

    const index_t block_id = get_block_1d_id();

    const auto c_k_n_h_w_block_cluster_idx =
        GridwiseGemm::GetCBlockIndex(c_blockid_to_k_n_h_w_block_cluster_adaptor, block_id);

    const auto c_thread_mtx_index = GridwiseGemm::GetCThreadIndex();

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    // Activ
    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatC,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        d_thread_buf;

    const auto scale_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_scale_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    // Bias
    GridwiseGemm::BiasActivRequantOp(bias_global_buf,
                                     scale_global_buf,
                                     c_thread_buf,
                                     d_thread_buf,
                                     c_k_n_h_w_block_cluster_idx,
                                     c_thread_mtx_index,
                                     bias_k0_k1_grid_desc,
                                     c_k1_n_h2_w2_thread_gemm_desc);

    auto d_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_d_grid, d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc.GetElementSpaceSize());

    // Resize_Add
    GridwiseGemm::ResizeAdd(d_thread_buf,
                            d_global_buf,
                            c_k_n_h_w_block_cluster_idx,
                            c_thread_mtx_index,
                            c_k1_n_h2_w2_thread_gemm_desc,
                            d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc);
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          ActivTypeEnum_t ActivType>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v3_resize_add(const FloatAB* __restrict__ p_a_grid,
                                        const FloatAB* __restrict__ p_b_grid,
                                        const FloatBias* __restrict__ p_bias_grid,
                                        FloatC* __restrict__ p_d_grid,
                                        float scaleRelu)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};
    constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc = DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx{};
    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

    // static constexpr auto activ_type = integral_constant<ActivTypeEnum_t, ActivType>{};

    const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_a_grid, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
    const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_b_grid, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());

    constexpr auto c_k1_n_h2_w2_thread_gemm_desc = GridwiseGemm::MakeCK1NH2W2ThreadDescriptor();

    // register allocation for output
    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatAcc,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        c_thread_buf;

    const index_t block_id = get_block_1d_id();

    const auto c_k_n_h_w_block_cluster_idx =
        GridwiseGemm::GetCBlockIndex(c_blockid_to_k_n_h_w_block_cluster_adaptor, block_id);

    const auto c_thread_mtx_index = GridwiseGemm::GetCThreadIndex();

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    // Bias
    GridwiseGemm::BiasOp(bias_global_buf,
                         c_thread_buf,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         bias_k0_k1_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc);

    // Activ
    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatC,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        d_thread_buf;

    // Activ
    GridwiseGemm::ActivationV2(c_thread_buf,
                               d_thread_buf,
                               c_k1_n_h2_w2_thread_gemm_desc,
                               // ck::tensor_operation::element_wise::PassThrough{},
                               ck::tensor_operation::element_wise::ReluQuant{scaleRelu});

    auto d_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_d_grid, d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc.GetElementSpaceSize());

    // Resize_Add
    GridwiseGemm::ResizeAdd(d_thread_buf,
                            d_global_buf,
                            c_k_n_h_w_block_cluster_idx,
                            c_thread_mtx_index,
                            c_k1_n_h2_w2_thread_gemm_desc,
                            d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc);
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatScale,
          typename FloatC,
          typename FloatCPacked,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename CPackedGridDesc_K0_K1x_N_H0_H1_H2_W0_W1_W2,
          typename DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          ActivTypeEnum_t ActivType>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v3_maxpool_int4(const FloatAB* __restrict__ p_a_grid,
                                          const FloatAB* __restrict__ p_b_grid,
                                          const FloatBias* __restrict__ p_bias_grid,
                                          const FloatScale* __restrict__ p_scale_grid,
                                          // FloatC* __restrict__ p_c_grid,
                                          FloatCPacked* __restrict__ p_c_packed_grid,
                                          FloatCPacked* __restrict__ p_d_grid)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};
    constexpr auto c_packed_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc =
        CPackedGridDesc_K0_K1x_N_H0_H1_H2_W0_W1_W2{};
    constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc = DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx{};
    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

    // static constexpr auto activ_type = integral_constant<ActivTypeEnum_t, ActivType>{};

    const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_a_grid, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
    const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_b_grid, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());

    constexpr auto c_k1_n_h2_w2_thread_gemm_desc = GridwiseGemm::MakeCK1NH2W2ThreadDescriptor();

    constexpr auto c_packed_k1x_n_h2_w2_thread_gemm_desc =
        GridwiseGemm::MakeDK1xNH2W2ThreadDescriptor();

    // register allocation for output
    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatAcc,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        c_thread_buf;

    const index_t block_id = get_block_1d_id();

    const auto c_k_n_h_w_block_cluster_idx =
        GridwiseGemm::GetCBlockIndex(c_blockid_to_k_n_h_w_block_cluster_adaptor, block_id);

    const auto c_thread_mtx_index = GridwiseGemm::GetCThreadIndex();

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    const auto scale_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_scale_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    // StaticBuffer<AddressSpaceEnum_t::Vgpr,
    // FloatC,
    // c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
    // true>
    // d_thread_buf;

    // Bias
    // GridwiseGemm::BiasActivRequantOp(bias_global_buf,
    // scale_global_buf,
    // c_thread_buf,
    // d_thread_buf,
    // c_k_n_h_w_block_cluster_idx,
    // c_thread_mtx_index,
    // bias_k0_k1_grid_desc,
    // c_k1_n_h2_w2_thread_gemm_desc);

    // auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
    // p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    // Output
    // GridwiseGemm::WriteOut(d_thread_buf,
    // c_global_buf,
    // c_k_n_h_w_block_cluster_idx,
    // c_thread_mtx_index,
    // c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatCPacked,
                 c_packed_k1x_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        c_packed_thread_buf;

    GridwiseGemm::BiasActivRequantInt4Op(bias_global_buf,
                                         scale_global_buf,
                                         c_thread_buf,
                                         c_packed_thread_buf,
                                         c_k_n_h_w_block_cluster_idx,
                                         c_thread_mtx_index,
                                         bias_k0_k1_grid_desc,
                                         c_k1_n_h2_w2_thread_gemm_desc,
                                         c_packed_k1x_n_h2_w2_thread_gemm_desc);

    // GridwiseGemm::RequantInt4Op(d_thread_buf,
    // d_packed_thread_buf,
    // c_k1_n_h2_w2_thread_gemm_desc,
    // c_packed_k1x_n_h2_w2_thread_gemm_desc);

    auto c_packed_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_c_packed_grid, c_packed_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    GridwiseGemm::WriteOutInt4(c_packed_thread_buf,
                               c_packed_global_buf,
                               c_k_n_h_w_block_cluster_idx,
                               c_thread_mtx_index,
                               c_packed_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc,
                               ck::tensor_operation::element_wise::PassThrough{});

    auto d_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_d_grid, d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc.GetElementSpaceSize());

    // MaxPool
    GridwiseGemm::MaxPool(c_packed_thread_buf,
                          d_global_buf,
                          c_k_n_h_w_block_cluster_idx,
                          c_thread_mtx_index,
                          c_k1_n_h2_w2_thread_gemm_desc,
                          d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc);
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatScale,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          ActivTypeEnum_t ActivType>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v3_maxpool_per_channel(const FloatAB* __restrict__ p_a_grid,
                                                 const FloatAB* __restrict__ p_b_grid,
                                                 const FloatBias* __restrict__ p_bias_grid,
                                                 const FloatScale* __restrict__ p_scale_grid,
                                                 FloatC* __restrict__ p_c_grid,
                                                 FloatC* __restrict__ p_d_grid)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};
    constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc = DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx{};
    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

    // static constexpr auto activ_type = integral_constant<ActivTypeEnum_t, ActivType>{};

    const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_a_grid, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
    const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_b_grid, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());

    constexpr auto c_k1_n_h2_w2_thread_gemm_desc = GridwiseGemm::MakeCK1NH2W2ThreadDescriptor();

    // register allocation for output
    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatAcc,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        c_thread_buf;

    const index_t block_id = get_block_1d_id();

    const auto c_k_n_h_w_block_cluster_idx =
        GridwiseGemm::GetCBlockIndex(c_blockid_to_k_n_h_w_block_cluster_adaptor, block_id);

    const auto c_thread_mtx_index = GridwiseGemm::GetCThreadIndex();

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    const auto scale_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_scale_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatC,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        d_thread_buf;

    // Bias
    GridwiseGemm::BiasActivRequantOp(bias_global_buf,
                                     scale_global_buf,
                                     c_thread_buf,
                                     d_thread_buf,
                                     c_k_n_h_w_block_cluster_idx,
                                     c_thread_mtx_index,
                                     bias_k0_k1_grid_desc,
                                     c_k1_n_h2_w2_thread_gemm_desc);

    auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    // Output
    GridwiseGemm::WriteOut(d_thread_buf,
                           c_global_buf,
                           c_k_n_h_w_block_cluster_idx,
                           c_thread_mtx_index,
                           c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    auto d_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_d_grid, d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc.GetElementSpaceSize());

    // MaxPool
    GridwiseGemm::MaxPool(d_thread_buf,
                          d_global_buf,
                          c_k_n_h_w_block_cluster_idx,
                          c_thread_mtx_index,
                          c_k1_n_h2_w2_thread_gemm_desc,
                          d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc);
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          ActivTypeEnum_t ActivType>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_dlops_v3_maxpool(const FloatAB* __restrict__ p_a_grid,
                                     const FloatAB* __restrict__ p_b_grid,
                                     const FloatBias* __restrict__ p_bias_grid,
                                     FloatC* __restrict__ p_c_grid,
                                     FloatC* __restrict__ p_d_grid,
                                     float scaleRelu)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};
    constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc = DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx{};
    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

    // static constexpr auto activ_type = integral_constant<ActivTypeEnum_t, ActivType>{};

    const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_a_grid, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
    const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_b_grid, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());

    constexpr auto c_k1_n_h2_w2_thread_gemm_desc = GridwiseGemm::MakeCK1NH2W2ThreadDescriptor();

    // register allocation for output
    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatAcc,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        c_thread_buf;

    const index_t block_id = get_block_1d_id();

    const auto c_k_n_h_w_block_cluster_idx =
        GridwiseGemm::GetCBlockIndex(c_blockid_to_k_n_h_w_block_cluster_adaptor, block_id);

    const auto c_thread_mtx_index = GridwiseGemm::GetCThreadIndex();

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    // Bias
    GridwiseGemm::BiasOp(bias_global_buf,
                         c_thread_buf,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         bias_k0_k1_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc);

    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatC,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        d_thread_buf;

    // Activ
    GridwiseGemm::ActivationV2(c_thread_buf,
                               d_thread_buf,
                               c_k1_n_h2_w2_thread_gemm_desc,
                               ck::tensor_operation::element_wise::ReluQuant{scaleRelu});

    auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    // Output
    GridwiseGemm::WriteOut(d_thread_buf,
                           c_global_buf,
                           c_k_n_h_w_block_cluster_idx,
                           c_thread_mtx_index,
                           c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    auto d_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_d_grid, d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc.GetElementSpaceSize());

    // MaxPool
    GridwiseGemm::MaxPool(d_thread_buf,
                          d_global_buf,
                          c_k_n_h_w_block_cluster_idx,
                          c_thread_mtx_index,
                          c_k1_n_h2_w2_thread_gemm_desc,
                          d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc);
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          ActivTypeEnum_t ActivType>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_bias_activ_dlops_v3(const FloatAB* __restrict__ p_a_grid,
                                        const FloatAB* __restrict__ p_b_grid,
                                        const FloatBias* __restrict__ p_bias_grid,
                                        FloatC* __restrict__ p_c_grid,
                                        float scaleRelu)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};

    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

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

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    const auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    // Bias
    GridwiseGemm::BiasOp(bias_global_buf,
                         c_thread_buf,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         bias_k0_k1_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc);

    auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    GridwiseGemm::WriteOut(c_thread_buf,
                           c_global_buf,
                           c_k_n_h_w_block_cluster_idx,
                           c_thread_mtx_index,
                           c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                           ck::tensor_operation::element_wise::ReluQuant{scaleRelu}
                           // ck::tensor_operation::element_wise::PassThrough{}
    );
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatScale,
          typename FloatC,
          typename FloatD,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename DGridDesc_K0_K1x_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          ActivTypeEnum_t ActivType>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_bias_activ_dlops_int4(const FloatAB* __restrict__ p_a_grid,
                                          const FloatAB* __restrict__ p_b_grid,
                                          const FloatBias* __restrict__ p_bias_grid,
                                          const FloatScale* __restrict__ p_scale_grid,
                                          // FloatC* __restrict__ p_c_grid,
                                          FloatD* __restrict__ p_d_grid)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};

    constexpr auto d_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc = DGridDesc_K0_K1x_N_H0_H1_H2_W0_W1_W2{};

    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

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

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    const auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    const auto scale_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_scale_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    constexpr auto d_k1x_n_h2_w2_thread_gemm_desc = GridwiseGemm::MakeDK1xNH2W2ThreadDescriptor();

    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatD,
                 d_k1x_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        e_thread_buf;

    // StaticBuffer<AddressSpaceEnum_t::Vgpr,
    // FloatC,
    // c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
    // true>
    // d_thread_buf;

    // Bias
    GridwiseGemm::BiasActivRequantInt4Op(bias_global_buf,
                                         scale_global_buf,
                                         c_thread_buf,
                                         e_thread_buf,
                                         c_k_n_h_w_block_cluster_idx,
                                         c_thread_mtx_index,
                                         bias_k0_k1_grid_desc,
                                         c_k1_n_h2_w2_thread_gemm_desc,
                                         d_k1x_n_h2_w2_thread_gemm_desc);

    // auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
    // p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    // GridwiseGemm::WriteOut(d_thread_buf,
    // c_global_buf,
    // c_k_n_h_w_block_cluster_idx,
    // c_thread_mtx_index,
    // c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
    // GridwiseGemm::RequantInt4Op(
    // d_thread_buf, e_thread_buf, c_k1_n_h2_w2_thread_gemm_desc, d_k1x_n_h2_w2_thread_gemm_desc);

    // static_for<0, d_k1x_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(), 1>{}(
    //[&](auto i) { e_thread_buf(i) = 0; });

    auto d_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_d_grid, d_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    GridwiseGemm::WriteOutInt4(e_thread_buf,
                               d_global_buf,
                               c_k_n_h_w_block_cluster_idx,
                               c_thread_mtx_index,
                               d_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc,
                               ck::tensor_operation::element_wise::PassThrough{});
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatAcc,
          typename FloatBias,
          typename FloatScale,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          ActivTypeEnum_t ActivType>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_bias_activ_dlops_v3_per_channel(const FloatAB* __restrict__ p_a_grid,
                                                    const FloatAB* __restrict__ p_b_grid,
                                                    const FloatBias* __restrict__ p_bias_grid,
                                                    const FloatScale* __restrict__ p_scale_grid,
                                                    FloatC* __restrict__ p_c_grid)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};

    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

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

    // GemmOp
    GridwiseGemm::GemmOp(a_global_buf,
                         b_global_buf,
                         c_thread_buf,
                         p_shared_block,
                         c_k_n_h_w_block_cluster_idx,
                         c_thread_mtx_index,
                         a_e0_e1_k0_k1_e2_grid_desc,
                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                         c_k1_n_h2_w2_thread_gemm_desc,
                         integral_constant<bool, HasMainE0BlockLoop>{});

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    const auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_bias_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    const auto scale_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_scale_grid, bias_k0_k1_grid_desc.GetElementSpaceSize());

    StaticBuffer<AddressSpaceEnum_t::Vgpr,
                 FloatC,
                 c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                 true>
        d_thread_buf;

    // Bias
    GridwiseGemm::BiasActivRequantOp(bias_global_buf,
                                     scale_global_buf,
                                     c_thread_buf,
                                     d_thread_buf,
                                     c_k_n_h_w_block_cluster_idx,
                                     c_thread_mtx_index,
                                     bias_k0_k1_grid_desc,
                                     c_k1_n_h2_w2_thread_gemm_desc);

    auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
        p_c_grid, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

    GridwiseGemm::WriteOut(d_thread_buf,
                           c_global_buf,
                           c_k_n_h_w_block_cluster_idx,
                           c_thread_mtx_index,
                           c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          index_t group_grid_size>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_dlops_v3(const FloatAB* __restrict__ p_a_grid,
                                     const FloatAB* __restrict__ p_b_grid,
                                     FloatC* __restrict__ p_c_grid)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};
    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

    const index_t group_id = get_block_1d_id() / group_grid_size;

    constexpr index_t a_stride_grp = a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize();
    constexpr index_t b_stride_grp = b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize();
    constexpr index_t c_stride_grp = c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize();

    const FloatAB* p_a_grid_grp = p_a_grid + group_id * a_stride_grp;
    const FloatAB* p_b_grid_grp = p_b_grid + group_id * b_stride_grp;

    FloatC* p_c_grid_grp = p_c_grid + group_id * c_stride_grp;

    GridwiseGemm::GroupConv(p_a_grid_grp,
                            p_b_grid_grp,
                            p_c_grid_grp,
                            p_shared_block,
                            group_grid_size,
                            a_e0_e1_k0_k1_e2_grid_desc,
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                            c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                            c_blockid_to_k_n_h_w_block_cluster_adaptor,
                            integral_constant<bool, HasMainE0BlockLoop>{});
}

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_E0_E1_K0_K1_E2,
          typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
          typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
          typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
          bool HasMainE0BlockLoop,
          index_t group_grid_size,
          index_t group_count,
          ActivTypeEnum_t ActivType>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_bias_activ_dlops_v3(const FloatAB* __restrict__ p_a_grid,
                                                const FloatAB* __restrict__ p_b_grid,
                                                const FloatC* __restrict__ p_bias_grid,
                                                FloatC* __restrict__ p_c_grid)
{
    constexpr index_t shared_block_size =
        GridwiseGemm::GetSharedMemoryNumberOfByte() / sizeof(FloatAB);

    __shared__ FloatAB p_shared_block[shared_block_size];

    constexpr auto a_e0_e1_k0_k1_e2_grid_desc = AGridDesc_E0_E1_K0_K1_E2{};
    constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
        BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2{};
    constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2{};
    constexpr auto c_blockid_to_k_n_h_w_block_cluster_adaptor =
        CBlockIdToBlockClusterAdaptor_K_N_H_W{};

    // const index_t group_id = get_block_1d_id() / group_grid_size;

    constexpr index_t a_stride_grp = a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize();
    constexpr index_t b_stride_grp = b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize();
    constexpr index_t c_stride_grp = c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize();

    const auto bias_k0_k1_grid_desc =
        GridwiseGemm::MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

    constexpr index_t bias_stride_grp = bias_k0_k1_grid_desc.GetElementSpaceSize();

    const index_t group_id = get_block_1d_id() / group_grid_size;
    // for(index_t group_id = 0; group_id < group_count; ++group_id)
    {
        const FloatAB* p_a_grid_grp = p_a_grid + group_id * a_stride_grp;
        const FloatAB* p_b_grid_grp = p_b_grid + group_id * b_stride_grp;

        const FloatC* p_bias_grid_grp = p_bias_grid + group_id * bias_stride_grp;

        FloatC* p_c_grid_grp = p_c_grid + group_id * c_stride_grp;

        GridwiseGemm::GroupConvBiasActiv(p_a_grid_grp,
                                         p_b_grid_grp,
                                         p_bias_grid_grp,
                                         p_c_grid_grp,
                                         p_shared_block,
                                         group_grid_size,
                                         a_e0_e1_k0_k1_e2_grid_desc,
                                         b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                         c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                                         c_blockid_to_k_n_h_w_block_cluster_adaptor,
                                         integral_constant<bool, HasMainE0BlockLoop>{},
                                         integral_constant<ActivTypeEnum_t, ActivType>{});
    }
}
#endif

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperationEnum_t CGlobalMemoryDataOperation,
          typename AGridDesc_E0_E1_K_E2,
          typename BGridDesc_E0_E1_N_Ho_Wo_E2,
          typename CGridDesc_K_N_Ho_Wo,
          index_t E1_,
          index_t E2_,
          index_t K2_,
          index_t KPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t E0PerBlock,
          index_t E1PerBlock,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
          typename ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_E2,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector>
struct GridwiseGemmDlops_km_kn_mn_v3
{

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto E1 = Number<E1_>{};
    static constexpr auto E2 = Number<E2_>{};
    static constexpr auto K2 = Number<K2_>{};

    static constexpr auto NPerBlock = I1;

    static constexpr FloatAcc alpha = 1.0;

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        constexpr auto max_lds_align = Number<ABlockTransferDstScalarPerVector_E2>{};

        // A matrix in LDS memory, dst of blockwise copy
        //   be careful of LDS alignment
        constexpr auto a_e0_e1_k1_e2_block_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(E0PerBlock, Number<E1>{}, Number<KPerBlock>{}, Number<E2>{}), max_lds_align);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size = math::integer_least_multiple(
            a_e0_e1_k1_e2_block_desc.GetElementSpaceSize(), max_lds_align);

        return a_block_space_size * sizeof(FloatAB);
    }

    __host__ __device__ static constexpr index_t
    CalculateGridSize(const CGridDesc_K_N_Ho_Wo& c_k_n_ho_wo_grid_desc)
    {
        const auto K  = c_k_n_ho_wo_grid_desc.GetLength(I0);
        const auto N  = c_k_n_ho_wo_grid_desc.GetLength(I1);
        const auto Ho = c_k_n_ho_wo_grid_desc.GetLength(I2);
        const auto Wo = c_k_n_ho_wo_grid_desc.GetLength(I3);

        const auto K0 = K / KPerBlock;
        const auto N0 = N / NPerBlock;
        const auto H0 = Ho / HoPerBlock;
        const auto W0 = Wo / WoPerBlock;

        const index_t grid_size = K0 * N0 * H0 * W0;

        return grid_size;
    }

    __host__ __device__ static constexpr bool CalculateHasMainE0BlockLoop(const index_t E0)
    {
        const bool has_main_e0_block_loop = E0 > 1;

        return has_main_e0_block_loop;
    }

    __host__ __device__ static constexpr bool CalculateHasMainE1BlockLoop()
    {
        const bool has_main_e1_block_loop = ((E1 + E1PerBlock) / (2 * E1PerBlock)) > 1;

        return has_main_e1_block_loop;
    }

    __host__ __device__ static constexpr bool CalculateHasDoubleTailE1BlockLoop()
    {
        const bool has_double_tail_e1_block_loop = (E1 / E1PerBlock) % 2 == 0;

        return has_double_tail_e1_block_loop;
    }

    __host__ __device__ static constexpr auto
    MakeAE0E1K0K1E2GridDescriptor(const AGridDesc_E0_E1_K_E2& a_e0_e1_k_e2_grid_desc)
    {
        const auto E0 = a_e0_e1_k_e2_grid_desc.GetLength(I0);
        const auto K  = a_e0_e1_k_e2_grid_desc.GetLength(I2);

        const auto K1 = Number<KPerBlock>{};
        const auto K0 = K / K1;

        const auto a_e0_e1_k0_k1_e2_grid_desc = transform_tensor_descriptor(
            a_e0_e1_k_e2_grid_desc,
            make_tuple(make_pass_through_transform(E0),
                       make_pass_through_transform(E1),
                       make_unmerge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}));

        return a_e0_e1_k0_k1_e2_grid_desc;
    }

    __host__ __device__ static constexpr auto MakeBE0E1NH0H1H2W0W1W2E2GridDescriptor(
        const BGridDesc_E0_E1_N_Ho_Wo_E2& b_e0_e1_n_ho_wo_e2_grid_desc)
    {
        const auto E0 = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I0);
        // const auto E1 = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I1);
        const auto N  = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I2);
        const auto Ho = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I3);
        const auto Wo = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I4);
        // const auto E2 = b_e0_e1_n_ho_wo_e2_grid_desc.GetLength(I5);

        const auto H2 = Number<HoPerThread>{};
        const auto H1 = Number<HoPerBlock / HoPerThread>{};
        const auto H0 = Ho / (H1 * H2);

        const auto W2 = Number<WoPerThread>{};
        const auto W1 = Number<WoPerBlock / WoPerThread>{};
        const auto W0 = Wo / (W1 * W2);

        const auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
            transform_tensor_descriptor(b_e0_e1_n_ho_wo_e2_grid_desc,
                                        make_tuple(make_pass_through_transform(E0),
                                                   make_pass_through_transform(E1),
                                                   make_pass_through_transform(N),
                                                   make_unmerge_transform(make_tuple(H0, H1, H2)),
                                                   make_unmerge_transform(make_tuple(W0, W1, W2)),
                                                   make_pass_through_transform(E2)),
                                        make_tuple(Sequence<0>{},
                                                   Sequence<1>{},
                                                   Sequence<2>{},
                                                   Sequence<3>{},
                                                   Sequence<4>{},
                                                   Sequence<5>{}),
                                        make_tuple(Sequence<0>{},
                                                   Sequence<1>{},
                                                   Sequence<2>{},
                                                   Sequence<3, 4, 5>{},
                                                   Sequence<6, 7, 8>{},
                                                   Sequence<9>{}));

        return b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc;
    }

    template <typename DGridDesc_Kx_N_Ho_Wo>
    __host__ __device__ static constexpr auto
    MakeDK0K1xNH0H1H2W0W1W2GridDescriptor(const DGridDesc_Kx_N_Ho_Wo& d_kx_n_ho_wo_grid_desc)
    {
        const auto K  = d_kx_n_ho_wo_grid_desc.GetLength(I0);
        const auto N  = d_kx_n_ho_wo_grid_desc.GetLength(I1);
        const auto Ho = d_kx_n_ho_wo_grid_desc.GetLength(I2);
        const auto Wo = d_kx_n_ho_wo_grid_desc.GetLength(I3);

        const auto K1 = Number<KPerBlock / 2>{};
        const auto K0 = K / K1;

        const auto H2 = Number<HoPerThread>{};
        const auto H1 = Number<HoPerBlock / HoPerThread>{};
        const auto H0 = Ho / (H1 * H2);

        const auto W2 = Number<WoPerThread>{};
        const auto W1 = Number<WoPerBlock / WoPerThread>{};
        const auto W0 = Wo / (W1 * W2);

        const auto d_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc = transform_tensor_descriptor(
            d_kx_n_ho_wo_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_unmerge_transform(make_tuple(H0, H1, H2)),
                       make_unmerge_transform(make_tuple(W0, W1, W2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}, Sequence<6, 7, 8>{}));

        return d_k0_k1x_n_h0_h1_h2_w0_w1_w2_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCK0K1NH0H1H2W0W1W2GridDescriptor(const CGridDesc_K_N_Ho_Wo& c_k_n_ho_wo_grid_desc)
    {
        const auto K  = c_k_n_ho_wo_grid_desc.GetLength(I0);
        const auto N  = c_k_n_ho_wo_grid_desc.GetLength(I1);
        const auto Ho = c_k_n_ho_wo_grid_desc.GetLength(I2);
        const auto Wo = c_k_n_ho_wo_grid_desc.GetLength(I3);

        const auto K1 = Number<KPerBlock>{};
        const auto K0 = K / K1;

        const auto H2 = Number<HoPerThread>{};
        const auto H1 = Number<HoPerBlock / HoPerThread>{};
        const auto H0 = Ho / (H1 * H2);

        const auto W2 = Number<WoPerThread>{};
        const auto W1 = Number<WoPerBlock / WoPerThread>{};
        const auto W0 = Wo / (W1 * W2);

        const auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc = transform_tensor_descriptor(
            c_k_n_ho_wo_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_unmerge_transform(make_tuple(H0, H1, H2)),
                       make_unmerge_transform(make_tuple(W0, W1, W2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}, Sequence<6, 7, 8>{}));

        return c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc;
    }

    template <typename DGridDesc_K_N_Hx_Wx>
    __host__ __device__ static constexpr auto
    MakeDK0K1NH0H1HxW0W1WxGridDescriptorMaxPool(const DGridDesc_K_N_Hx_Wx& d_k_n_hx_wx_grid_desc)
    {
        const auto K  = d_k_n_hx_wx_grid_desc.GetLength(I0);
        const auto N  = d_k_n_hx_wx_grid_desc.GetLength(I1);
        const auto Hx = d_k_n_hx_wx_grid_desc.GetLength(I2);
        const auto Wx = d_k_n_hx_wx_grid_desc.GetLength(I3);

        const auto K1 = Number<KPerBlock / 2>{};
        const auto K0 = K / K1;

#if CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR
        const auto H2 = Number<HoPerThread / 2>{};
        const auto H1 = Number<HoPerBlock / HoPerThread>{};
        const auto H0 = Number<Hx / (H1 * H2)>{};

        const auto W2 = Number<WoPerThread / 2>{};
        const auto W1 = Number<WoPerBlock / WoPerThread>{};
        const auto W0 = Number<Wx / (W1 * W2)>{};
#else
        const auto H2 = HoPerThread / 2;
        const auto H1 = HoPerBlock / HoPerThread;
        const auto H0 = Hx / (H1 * H2);

        const auto W2 = WoPerThread / 2;
        const auto W1 = WoPerBlock / WoPerThread;
        const auto W0 = Wx / (W1 * W2);
#endif

        const auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc = transform_tensor_descriptor(
            d_k_n_hx_wx_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_unmerge_transform(make_tuple(H0, H1, H2)),
                       make_unmerge_transform(make_tuple(W0, W1, W2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}, Sequence<6, 7, 8>{}));

        return d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc;
    }

    template <typename DGridDesc_K_N_Hx_Wx>
    __host__ __device__ static constexpr auto
    MakeDK0K1NH0H1HxW0W1WxGridDescriptorResizeAdd(const DGridDesc_K_N_Hx_Wx& d_k_n_hx_wx_grid_desc)
    {
        const auto K  = d_k_n_hx_wx_grid_desc.GetLength(I0);
        const auto N  = d_k_n_hx_wx_grid_desc.GetLength(I1);
        const auto Hx = d_k_n_hx_wx_grid_desc.GetLength(I2);
        const auto Wx = d_k_n_hx_wx_grid_desc.GetLength(I3);

        const auto K1 = Number<KPerBlock>{};
        const auto K0 = K / K1;

        const auto H2 = Number<HoPerThread * 2>{};
        const auto H1 = Number<HoPerBlock / HoPerThread>{};

        const auto W2 = Number<WoPerThread * 2>{};
        const auto W1 = Number<WoPerBlock / WoPerThread>{};

#if CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR
        const auto H0 = Number<Hx / (H1 * H2)>{};
        const auto W0 = Number<Wx / (W1 * W2)>{};
#else
        const auto H0 = Hx / (H1 * H2);
        const auto W0 = Wx / (W1 * W2);
#endif

        const auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc = transform_tensor_descriptor(
            d_k_n_hx_wx_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_unmerge_transform(make_tuple(H0, H1, H2)),
                       make_unmerge_transform(make_tuple(W0, W1, W2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}, Sequence<6, 7, 8>{}));

        return d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc;
    }

    __host__ __device__ static constexpr auto
    MakeCBlockIdToKNHoWoBlockClusterAdaptor(const CGridDesc_K_N_Ho_Wo&)
    {
        constexpr index_t K  = CGridDesc_K_N_Ho_Wo{}.GetLength(I0);
        constexpr index_t N  = CGridDesc_K_N_Ho_Wo{}.GetLength(I1);
        constexpr index_t Ho = CGridDesc_K_N_Ho_Wo{}.GetLength(I2);
        constexpr index_t Wo = CGridDesc_K_N_Ho_Wo{}.GetLength(I3);

        // printf("%d %d %d %d\n", K, N, Ho, Wo);

        constexpr auto K0 = Number<K / KPerBlock>{};
        constexpr auto N0 = Number<N / NPerBlock>{};
        constexpr auto H0 = Number<Ho / HoPerBlock>{};
        constexpr auto W0 = Number<Wo / WoPerBlock>{};
        //#else
        // const index_t K0 = K / KPerBlock;
        // const index_t N0 = N / NPerBlock;
        // const index_t H0 = Ho / HoPerBlock;
        // const index_t W0 = Wo / WoPerBlock;
        //#endif
        constexpr auto c_blockid_to_k_n_ho_wo_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(K0, N0, H0, W0))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

        return c_blockid_to_k_n_ho_wo_block_cluster_adaptor;
    }

    using CBlockIdToBlockClusterAdaptor_K_N_H_W =
        decltype(MakeCBlockIdToKNHoWoBlockClusterAdaptor(CGridDesc_K_N_Ho_Wo{}));

    template <typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>
    __host__ __device__ static constexpr auto MakeBiasK0K1GridDescriptor(
        const CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2& c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc)
    {
        const auto K0 = c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetLength(I0);
        const auto K1 = c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetLength(I1);

        return make_naive_tensor_descriptor_packed(make_tuple(K0, K1));
    }

    __host__ __device__ static constexpr auto MakeCK1NH2W2ThreadDescriptor()
    {
        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<KPerThread>{}, I1, Number<HoPerThread>{}, Number<WoPerThread>{}));
        return c_k1_n_h2_w2_thread_gemm_desc;
    }

    __host__ __device__ static constexpr auto MakeDK1xNH2W2ThreadDescriptor()
    {
        constexpr auto d_k1x_n_h2_w2_thread_gemm_desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<KPerThread / 2>{}, I1, Number<HoPerThread>{}, Number<WoPerThread>{}));
        return d_k1x_n_h2_w2_thread_gemm_desc;
    }

    __host__ __device__ static constexpr auto GetBlockWiseGemm()
    {
        constexpr auto max_lds_align = Number<ABlockTransferDstScalarPerVector_E2>{};

        constexpr auto a_e1_k1_e2_block_gemm_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E1PerBlock>{}, Number<KPerBlock>{}, Number<E2>{}), max_lds_align);

        constexpr auto b_e1_n_h_w_e2_block_gemm_desc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<E1PerBlock>{},
                                                           I1,
                                                           Number<HoPerBlock>{},
                                                           Number<WoPerBlock>{},
                                                           Number<E2>{}));

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = MakeCK1NH2W2ThreadDescriptor();

        auto blockwise_gemm =
            BlockwiseGemmDlops_km_kn_m0m1n0n1_v3<BlockSize,
                                                 FloatAB,
                                                 FloatAB,
                                                 FloatAcc,
                                                 decltype(a_e1_k1_e2_block_gemm_desc),
                                                 decltype(b_e1_n_h_w_e2_block_gemm_desc),
                                                 decltype(c_k1_n_h2_w2_thread_gemm_desc),
                                                 EPerThread,
                                                 K2>{};

        return blockwise_gemm;
    }

    __device__ static constexpr auto GetCThreadIndex()
    {
        auto blockwise_gemm = GetBlockWiseGemm();
        auto c_thread_mtx_index =
            blockwise_gemm.GetBeginOfCThreadDesc_K_N_Ho_Wo(get_thread_local_1d_id());

        return c_thread_mtx_index;
    };

    __device__ static constexpr auto GetCBlockIndex(
        const CBlockIdToBlockClusterAdaptor_K_N_H_W& c_blockid_to_k_n_h_w_block_cluster_adaptor,
        const index_t block_id)
    {
        const auto c_k_n_h_w_block_cluster_idx =
            c_blockid_to_k_n_h_w_block_cluster_adaptor.CalculateBottomIndex(
                make_multi_index(block_id));
        return c_k_n_h_w_block_cluster_idx;
    }

    template <typename CThreadBuff,
              typename DThreadBuff,
              typename CThreadDesc_K1_N_H2_W2,
              typename DThreadDesc_K1x_N_H2_W2>
    __device__ static void RequantInt4Op(CThreadBuff& c_thread_buf,
                                         DThreadBuff& d_thread_buf,
                                         const CThreadDesc_K1_N_H2_W2&,
                                         const DThreadDesc_K1x_N_H2_W2&)

    {
        constexpr auto c_k1_n_h2_w2_thread_gemm_desc  = CThreadDesc_K1_N_H2_W2{};
        constexpr auto d_k1x_n_h2_w2_thread_gemm_desc = DThreadDesc_K1x_N_H2_W2{};

        static_for<0, d_k1x_n_h2_w2_thread_gemm_desc.GetLength(I0), 1>{}([&](auto ki) {
            static_for<0, HoPerThread, 1>{}([&](auto hi) {
                static_for<0, WoPerThread, 1>{}([&](auto wi) {
                    constexpr index_t c_offset0 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                        make_tuple(ki * 2, 0, hi, wi));
                    constexpr index_t c_offset1 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                        make_tuple(ki * 2 + 1, 0, hi, wi));

                    int8_t low  = c_thread_buf[Number<c_offset0>{}];
                    int8_t high = c_thread_buf[Number<c_offset1>{}];

                    ck::int8x2_t tmp{low, high};
                    ck::int4x2_t value = ck::type_convert<ck::int4x2_t>(tmp);

                    constexpr index_t d_offset =
                        d_k1x_n_h2_w2_thread_gemm_desc.CalculateOffset(make_tuple(ki, 0, hi, wi));
                    d_thread_buf(Number<d_offset>{}) = value;
                });
            });
        });
    }

    template <typename BiasGlobalBuff,
              typename ScaleGlobalBuff,
              typename CThreadBuff,
              typename DThreadBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename BiasGridDesc_K0_K1,
              typename CThreadDesc_K1_N_H2_W2,
              typename DThreadDesc_K1x_N_H2_W2>
    __device__ static void BiasActivRequantInt4Op(const BiasGlobalBuff& bias_global_buf,
                                                  const ScaleGlobalBuff& scale_global_buf,
                                                  CThreadBuff& c_thread_buf,
                                                  DThreadBuff& d_thread_buf,
                                                  const CBlockIndex& c_block_idx,
                                                  const CThreadIndex& c_thread_idx,
                                                  const BiasGridDesc_K0_K1& bias_k0_k1_grid_desc,
                                                  const CThreadDesc_K1_N_H2_W2&,
                                                  const DThreadDesc_K1x_N_H2_W2)

    {
        const index_t k_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);

        const auto k_thread_id = c_thread_idx[I0];

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc  = CThreadDesc_K1_N_H2_W2{};
        constexpr auto d_k1x_n_h2_w2_thread_gemm_desc = DThreadDesc_K1x_N_H2_W2{};

        constexpr auto bias_k0_k1_thread_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1, Number<KPerThread>{}));

        using FloatBias = remove_cvref_t<typename BiasGlobalBuff::type>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatBias,
                     bias_k0_k1_thread_desc.GetElementSpaceSize(),
                     true>
            bias_thread_buf;

        const index_t k_thread_data_on_global = k_thread_id * KPerThread;

        auto bias_threadwise_transfer =
            ThreadwiseTensorSliceTransfer_v2<FloatBias,
                                             FloatBias,
                                             decltype(bias_k0_k1_grid_desc),
                                             decltype(bias_k0_k1_thread_desc),
                                             Sequence<I1, Number<KPerThread>{}>,
                                             Sequence<0, 1>,
                                             1,
                                             CThreadTransferDstScalarPerVector,
                                             false,
                                             true>(
                bias_k0_k1_grid_desc, make_multi_index(k_block_work_id, k_thread_data_on_global));

        const auto hacks = make_tuple(make_tuple(Sequence<0>{}, Sequence<0>{}),
                                      make_tuple(Sequence<0>{}, Sequence<0>{}));

        bias_threadwise_transfer.Run(bias_k0_k1_grid_desc,
                                     bias_global_buf,
                                     bias_k0_k1_thread_desc,
                                     make_tuple(I0, I0),
                                     bias_thread_buf,
                                     hacks);

        using FloatScale = remove_cvref_t<typename ScaleGlobalBuff::type>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatScale,
                     bias_k0_k1_thread_desc.GetElementSpaceSize(),
                     true>
            scale_thread_buf;

        auto scale_threadwise_transfer =
            ThreadwiseTensorSliceTransfer_v2<FloatScale,
                                             FloatScale,
                                             decltype(bias_k0_k1_grid_desc),
                                             decltype(bias_k0_k1_thread_desc),
                                             Sequence<I1, Number<KPerThread>{}>,
                                             Sequence<0, 1>,
                                             1,
                                             CThreadTransferDstScalarPerVector,
                                             false,
                                             true>(
                bias_k0_k1_grid_desc, make_multi_index(k_block_work_id, k_thread_data_on_global));

        scale_threadwise_transfer.Run(bias_k0_k1_grid_desc,
                                      scale_global_buf,
                                      bias_k0_k1_thread_desc,
                                      make_tuple(I0, I0),
                                      scale_thread_buf,
                                      hacks);

        static_for<0, d_k1x_n_h2_w2_thread_gemm_desc.GetLength(I0), 1>{}([&](auto ki) {
            static_for<0, HoPerThread, 1>{}([&](auto hi) {
                static_for<0, WoPerThread, 1>{}([&](auto wi) {
                    constexpr auto k_l = Number<ki * 2>{};
                    constexpr auto k_h = Number<ki * 2 + 1>{};

                    constexpr index_t c_offset0 =
                        c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(make_tuple(k_l, 0, hi, wi));
                    constexpr index_t c_offset1 =
                        c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(make_tuple(k_h, 0, hi, wi));

                    const auto element_wise_op0 =
                        ck::tensor_operation::element_wise::HardTanhQuant{scale_thread_buf[k_l]};
                    const auto element_wise_op1 =
                        ck::tensor_operation::element_wise::HardTanhQuant{scale_thread_buf[k_h]};

                    int8_t low =
                        element_wise_op0(c_thread_buf[Number<c_offset0>{}] + bias_thread_buf[k_l]);
                    int8_t high =
                        element_wise_op1(c_thread_buf[Number<c_offset1>{}] + bias_thread_buf[k_h]);

                    ck::int8x2_t tmp{low, high};
                    ck::int4x2_t value = ck::type_convert<ck::int4x2_t>(tmp);

                    constexpr index_t d_offset =
                        d_k1x_n_h2_w2_thread_gemm_desc.CalculateOffset(make_tuple(ki, 0, hi, wi));
                    d_thread_buf(Number<d_offset>{}) = value;
                });
            });
        });
    }

    template <typename BiasGlobalBuff,
              typename ScaleGlobalBuff,
              typename CThreadBuff,
              typename DThreadBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename BiasGridDesc_K0_K1,
              typename CThreadDesc_K1_N_H2_W2>
    __device__ static void BiasActivRequantOp(const BiasGlobalBuff& bias_global_buf,
                                              const ScaleGlobalBuff& scale_global_buf,
                                              CThreadBuff& c_thread_buf,
                                              DThreadBuff& d_thread_buf,
                                              const CBlockIndex& c_block_idx,
                                              const CThreadIndex& c_thread_idx,
                                              const BiasGridDesc_K0_K1& bias_k0_k1_grid_desc,
                                              const CThreadDesc_K1_N_H2_W2&)

    {
        const index_t k_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);

        const auto k_thread_id = c_thread_idx[I0];

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

        constexpr auto bias_k0_k1_thread_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1, Number<KPerThread>{}));

        using FloatBias = remove_cvref_t<typename BiasGlobalBuff::type>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatBias,
                     bias_k0_k1_thread_desc.GetElementSpaceSize(),
                     true>
            bias_thread_buf;

        const index_t k_thread_data_on_global = k_thread_id * KPerThread;

        auto bias_threadwise_transfer =
            ThreadwiseTensorSliceTransfer_v2<FloatBias,
                                             FloatBias,
                                             decltype(bias_k0_k1_grid_desc),
                                             decltype(bias_k0_k1_thread_desc),
                                             Sequence<I1, Number<KPerThread>{}>,
                                             Sequence<0, 1>,
                                             1,
                                             CThreadTransferDstScalarPerVector,
                                             false,
                                             true>(
                bias_k0_k1_grid_desc, make_multi_index(k_block_work_id, k_thread_data_on_global));

        const auto hacks = make_tuple(make_tuple(Sequence<0>{}, Sequence<0>{}),
                                      make_tuple(Sequence<0>{}, Sequence<0>{}));

        bias_threadwise_transfer.Run(bias_k0_k1_grid_desc,
                                     bias_global_buf,
                                     bias_k0_k1_thread_desc,
                                     make_tuple(I0, I0),
                                     bias_thread_buf,
                                     hacks);

        using FloatScale = remove_cvref_t<typename ScaleGlobalBuff::type>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatScale,
                     bias_k0_k1_thread_desc.GetElementSpaceSize(),
                     true>
            scale_thread_buf;

        auto scale_threadwise_transfer =
            ThreadwiseTensorSliceTransfer_v2<FloatScale,
                                             FloatScale,
                                             decltype(bias_k0_k1_grid_desc),
                                             decltype(bias_k0_k1_thread_desc),
                                             Sequence<I1, Number<KPerThread>{}>,
                                             Sequence<0, 1>,
                                             1,
                                             CThreadTransferDstScalarPerVector,
                                             false,
                                             true>(
                bias_k0_k1_grid_desc, make_multi_index(k_block_work_id, k_thread_data_on_global));

        scale_threadwise_transfer.Run(bias_k0_k1_grid_desc,
                                      scale_global_buf,
                                      bias_k0_k1_thread_desc,
                                      make_tuple(I0, I0),
                                      scale_thread_buf,
                                      hacks);

        static_for<0, KPerThread, 1>{}([&](auto ki) {
            static_for<0, HoPerThread, 1>{}([&](auto hi) {
                static_for<0, WoPerThread, 1>{}([&](auto wi) {
                    constexpr index_t c_offset =
                        c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(make_tuple(ki, 0, hi, wi));
                    auto tmp = c_thread_buf[Number<c_offset>{}] + bias_thread_buf[ki];
                    const auto element_wise_op =
                        ck::tensor_operation::element_wise::HardTanhQuant{scale_thread_buf[ki]};
                    d_thread_buf(Number<c_offset>{}) = element_wise_op(tmp);
                });
            });
        });
    }

    template <typename BiasGlobalBuff,
              typename CThreadBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename BiasGridDesc_K0_K1,
              typename CThreadDesc_K1_N_H2_W2>
    __device__ static void BiasOp(const BiasGlobalBuff& bias_global_buf,
                                  CThreadBuff& c_thread_buf,
                                  const CBlockIndex& c_block_idx,
                                  const CThreadIndex& c_thread_idx,
                                  const BiasGridDesc_K0_K1& bias_k0_k1_grid_desc,
                                  const CThreadDesc_K1_N_H2_W2&)

    {
        const index_t k_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);

        const auto k_thread_id = c_thread_idx[I0];

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

        constexpr auto bias_k0_k1_thread_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1, Number<KPerThread>{}));

        using FloatBias = typename BiasGlobalBuff::type;

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatBias,
                     bias_k0_k1_thread_desc.GetElementSpaceSize(),
                     true>
            bias_thread_buf;

        const index_t k_thread_data_on_global = k_thread_id * KPerThread;

        auto bias_threadwise_transfer =
            ThreadwiseTensorSliceTransfer_v2<FloatBias,
                                             FloatBias,
                                             decltype(bias_k0_k1_grid_desc),
                                             decltype(bias_k0_k1_thread_desc),
                                             Sequence<I1, Number<KPerThread>{}>,
                                             Sequence<0, 1>,
                                             1,
                                             CThreadTransferDstScalarPerVector,
                                             false,
                                             true>(
                bias_k0_k1_grid_desc, make_multi_index(k_block_work_id, k_thread_data_on_global));

        const auto hacks = make_tuple(make_tuple(Sequence<0>{}, Sequence<0>{}),
                                      make_tuple(Sequence<0>{}, Sequence<0>{}));

        bias_threadwise_transfer.Run(bias_k0_k1_grid_desc,
                                     bias_global_buf,
                                     bias_k0_k1_thread_desc,
                                     make_tuple(I0, I0),
                                     bias_thread_buf,
                                     hacks);

        static_for<0, KPerThread, 1>{}([&](auto ki) {
            static_for<0, HoPerThread, 1>{}([&](auto hi) {
                static_for<0, WoPerThread, 1>{}([&](auto wi) {
                    constexpr index_t c_offset =
                        c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(make_tuple(ki, 0, hi, wi));
                    c_thread_buf(Number<c_offset>{}) =
                        c_thread_buf[Number<c_offset>{}] + bias_thread_buf[ki];
                });
            });
        });
    }

    template <typename CThreadBuff,
              typename DThreadBuff,
              typename CThreadDesc_K1_N_H2_W2,
              typename ElementwiseOp>
    __device__ static void ActivationV2(CThreadBuff& c_thread_buf,
                                        DThreadBuff& d_thread_buf,
                                        const CThreadDesc_K1_N_H2_W2&,
                                        ElementwiseOp element_wise_op)
    {
        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

        static_for<0, c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(), 1>{}(
            [&](auto i) { d_thread_buf(i) = element_wise_op(c_thread_buf[i]); });
    }

    template <typename CThreadBuff, typename CThreadDesc_K1_N_H2_W2, ActivTypeEnum_t activ_type_>
    __device__ static void Activation(CThreadBuff& c_thread_buf,
                                      const CThreadDesc_K1_N_H2_W2&,
                                      integral_constant<ActivTypeEnum_t, activ_type_>)
    {
        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

        static_for<0, c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(), 1>{}([&](auto i) {
            if constexpr(activ_type_ == 1)
            {
                // c_thread_buf(i) = c_thread_buf[i] >= 0 ? c_thread_buf[i] : alpha *
                // c_thread_buf[i];
                c_thread_buf(i) = c_thread_buf[i] > 0 ? c_thread_buf[i] : 0;
            }
            else if constexpr(activ_type_ == 2)
            {
                FloatAcc x = 1.0 + exp(-c_thread_buf[i]);

                asm volatile("\n \
                        v_rcp_f32 %0, %1 \n"
                             : "=v"(x)
                             : "0"(x));

                c_thread_buf(i) = x;
            }
        });
    }

    template <typename CThreadBuff,
              typename CGlobalBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
              typename CElementWiseOp = ck::tensor_operation::element_wise::PassThrough>
    __device__ static void WriteOut(
        const CThreadBuff& c_thread_buf,
        CGlobalBuff& c_global_buf,
        const CBlockIndex& c_block_idx,
        const CThreadIndex& c_thread_idx,
        const CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2& c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
        const CElementWiseOp c_element_wise_op = ck::tensor_operation::element_wise::PassThrough{})
    {
        const index_t k_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);
        const index_t n_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I1]);
        const index_t ho_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I2]);
        const index_t wo_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I3]);

        const auto k_thread_id  = c_thread_idx[I0];
        const auto ho_thread_id = c_thread_idx[I2];
        const auto wo_thread_id = c_thread_idx[I3];

        constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_thread_copy_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<KPerThread>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<HoPerThread>{},
                                                           I1,
                                                           I1,
                                                           Number<WoPerThread>{}));

        const index_t k_thread_data_on_global = k_thread_id * KPerThread;

        using FloatAcc_ = typename CThreadBuff::type;
        using FloatC_   = typename CGlobalBuff::type;

        ThreadwiseTensorSliceTransfer_v1r3<
            FloatAcc_,
            FloatC_,
            decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_thread_copy_desc),
            decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc),
            CElementWiseOp,
            Sequence<I1, KPerThread, I1, I1, I1, HoPerThread, I1, I1, WoPerThread>,
            CThreadTransferSrcDstAccessOrder,
            CThreadTransferSrcDstVectorDim,
            CThreadTransferDstScalarPerVector,
            CGlobalMemoryDataOperation,
            1,
            true>(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                  make_multi_index(k_block_work_id,
                                   k_thread_data_on_global,
                                   n_block_work_id,
                                   ho_block_work_id,
                                   ho_thread_id,
                                   0,
                                   wo_block_work_id,
                                   wo_thread_id,
                                   0),
                  c_element_wise_op)
            .Run(c_k0_k1_n_h0_h1_h2_w0_w1_w2_thread_copy_desc,
                 make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                 c_thread_buf,
                 c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                 c_global_buf);
    }

    template <typename CThreadBuff,
              typename CGlobalBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
              typename CElementWiseOp = ck::tensor_operation::element_wise::PassThrough>
    __device__ static void WriteOutInt4(
        const CThreadBuff& c_thread_buf,
        CGlobalBuff& c_global_buf,
        const CBlockIndex& c_block_idx,
        const CThreadIndex& c_thread_idx,
        const CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2& c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
        const CElementWiseOp c_element_wise_op = ck::tensor_operation::element_wise::PassThrough{})
    {
        const index_t k_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);
        const index_t n_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I1]);
        const index_t ho_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I2]);
        const index_t wo_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I3]);

        const auto k_thread_id  = c_thread_idx[I0];
        const auto ho_thread_id = c_thread_idx[I2];
        const auto wo_thread_id = c_thread_idx[I3];

        constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_thread_copy_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<KPerThread / 2>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<HoPerThread>{},
                                                           I1,
                                                           I1,
                                                           Number<WoPerThread>{}));

        const index_t k_thread_data_on_global = k_thread_id * KPerThread / 2;

        using FloatAcc_ = typename CThreadBuff::type;
        using FloatC_   = typename CGlobalBuff::type;

        ThreadwiseTensorSliceTransfer_v1r3<
            FloatAcc_,
            FloatC_,
            decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_thread_copy_desc),
            decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc),
            CElementWiseOp,
            Sequence<I1, KPerThread / 2, I1, I1, I1, HoPerThread, I1, I1, WoPerThread>,
            CThreadTransferSrcDstAccessOrder,
            CThreadTransferSrcDstVectorDim,
            KPerThread / 2,
            CGlobalMemoryDataOperation,
            1,
            true>(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                  make_multi_index(k_block_work_id,
                                   k_thread_data_on_global,
                                   n_block_work_id,
                                   ho_block_work_id,
                                   ho_thread_id,
                                   0,
                                   wo_block_work_id,
                                   wo_thread_id,
                                   0),
                  c_element_wise_op)
            .Run(c_k0_k1_n_h0_h1_h2_w0_w1_w2_thread_copy_desc,
                 make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                 c_thread_buf,
                 c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                 c_global_buf);
    }

    template <typename CThreadBuff,
              typename DGlobalBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename CThreadDesc_K1_N_H2_W2,
              typename DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx>
    __device__ static void
    MaxPool(const CThreadBuff& c_thread_buf,
            DGlobalBuff& d_global_buf,
            const CBlockIndex& c_block_idx,
            const CThreadIndex& c_thread_idx,
            const CThreadDesc_K1_N_H2_W2&,
            const DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx& d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc)
    {

        const index_t k_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);
        const index_t n_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I1]);
        const index_t ho_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I2]);
        const index_t wo_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I3]);

        const auto k_thread_id  = c_thread_idx[I0];
        const auto ho_thread_id = c_thread_idx[I2];
        const auto wo_thread_id = c_thread_idx[I3];

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

        static_assert(HoPerThread % 2 == 0 && WoPerThread % 2 == 0, "");

        constexpr auto HoPerThread_2 = HoPerThread / 2;
        constexpr auto WoPerThread_2 = WoPerThread / 2;

        constexpr auto K1PerThread = KPerThread / 2;

        constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<K1PerThread>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<HoPerThread_2>{},
                                                           I1,
                                                           I1,
                                                           Number<WoPerThread_2>{}));

        using FloatC_ = remove_cvref_t<typename CThreadBuff::type>;

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatC_,
                     d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc.GetElementSpaceSize(),
                     true>
            d_thread_buf;

        static_for<0, K1PerThread, 1>{}([&](auto ki) {
            static_for<0, HoPerThread_2, 1>{}([&](auto hi) {
                static_for<0, WoPerThread_2, 1>{}([&](auto wi) {
                    constexpr index_t d_offset =
                        d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc.CalculateOffset(
                            make_tuple(0, ki, 0, 0, 0, hi, 0, 0, wi));

                    constexpr index_t c_offset_0 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                        make_tuple(ki, 0, hi * 2, wi * 2));
                    constexpr index_t c_offset_1 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                        make_tuple(ki, 0, hi * 2, wi * 2 + 1));
                    constexpr index_t c_offset_2 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                        make_tuple(ki, 0, hi * 2 + 1, wi * 2));
                    constexpr index_t c_offset_3 = c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                        make_tuple(ki, 0, hi * 2 + 1, wi * 2 + 1));

                    auto v0 = c_thread_buf[Number<c_offset_0>{}];
                    auto v1 = c_thread_buf[Number<c_offset_1>{}];
                    auto v2 = c_thread_buf[Number<c_offset_2>{}];
                    auto v3 = c_thread_buf[Number<c_offset_3>{}];

                    d_thread_buf(Number<d_offset>{}) = max_(v0, max_(v1, max_(v2, v3)));
                });
            });
        });

        const index_t k_thread_data_on_global = k_thread_id * K1PerThread;

        ThreadwiseTensorSliceTransfer_v1r3<
            FloatC_,
            FloatC_,
            decltype(d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc),
            decltype(d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc),
            ck::tensor_operation::element_wise::PassThrough,
            Sequence<I1, K1PerThread, I1, I1, I1, HoPerThread_2, I1, I1, WoPerThread_2>,
            CThreadTransferSrcDstAccessOrder,
            CThreadTransferSrcDstVectorDim,
            K1PerThread,
            InMemoryDataOperationEnum_t::Set,
            1,
            true>(d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc,
                  make_multi_index(k_block_work_id,
                                   k_thread_data_on_global,
                                   n_block_work_id,
                                   ho_block_work_id,
                                   ho_thread_id,
                                   0,
                                   wo_block_work_id,
                                   wo_thread_id,
                                   0),
                  ck::tensor_operation::element_wise::PassThrough{})
            .Run(d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc,
                 make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                 d_thread_buf,
                 d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc,
                 d_global_buf);
    }

    template <typename CThreadBuff,
              typename DGlobalBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename CThreadDesc_K1_N_H2_W2,
              typename DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx>
    __device__ static void
    ResizeAdd(const CThreadBuff& c_thread_buf,
              DGlobalBuff& d_global_buf,
              const CBlockIndex& c_block_idx,
              const CThreadIndex& c_thread_idx,
              const CThreadDesc_K1_N_H2_W2&,
              const DGridDesc_K0_K1_N_H0_H1_Hx_W0_W1_Wx& d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc)
    {

        const index_t k_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);
        const index_t n_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I1]);
        const index_t ho_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I2]);
        const index_t wo_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I3]);

        const auto k_thread_id  = c_thread_idx[I0];
        const auto ho_thread_id = c_thread_idx[I2];
        const auto wo_thread_id = c_thread_idx[I3];

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

        constexpr auto HoPerThreadx2 = HoPerThread * 2;
        constexpr auto WoPerThreadx2 = WoPerThread * 2;

        constexpr auto d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<KPerThread>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<HoPerThreadx2>{},
                                                           I1,
                                                           I1,
                                                           Number<WoPerThreadx2>{}));

        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatC,
                     d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc.GetElementSpaceSize(),
                     true>
            d_thread_buf;

        static_for<0, KPerThread, 1>{}([&](auto k_i) {
            static_for<0, HoPerThreadx2, 1>{}([&](auto h_i) {
                static_for<0, WoPerThreadx2, 1>{}([&](auto w_i) {
                    d_thread_buf(Number<d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc.CalculateOffset(
                                     make_tuple(0, k_i, 0, 0, 0, h_i, 0, 0, w_i))>{}) =
                        c_thread_buf[Number<c_k1_n_h2_w2_thread_gemm_desc.CalculateOffset(
                            make_tuple(k_i, 0, h_i / 2, w_i / 2))>{}];
                });
            });
        });

        const index_t k_thread_data_on_global = k_thread_id * KPerThread;

        ThreadwiseTensorSliceTransfer_v1r3<
            FloatC,
            FloatC,
            decltype(d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc),
            decltype(d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc),
            ck::tensor_operation::element_wise::PassThrough,
            Sequence<I1, KPerThread, I1, I1, I1, HoPerThreadx2, I1, I1, WoPerThreadx2>,
            CThreadTransferSrcDstAccessOrder,
            CThreadTransferSrcDstVectorDim,
            CThreadTransferDstScalarPerVector,
            InMemoryDataOperationEnum_t::Set,
            1,
            true>(d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc,
                  make_multi_index(k_block_work_id,
                                   k_thread_data_on_global,
                                   n_block_work_id,
                                   ho_block_work_id,
                                   ho_thread_id,
                                   0,
                                   wo_block_work_id,
                                   wo_thread_id,
                                   0),
                  ck::tensor_operation::element_wise::PassThrough{})
            .Run(d_k0_k1_n_h0_h1_hx_w0_w1_wx_thread_desc,
                 make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                 d_thread_buf,
                 d_k0_k1_n_h0_h1_hx_w0_w1_wx_grid_desc,
                 d_global_buf);
    }

    template <typename AGlobalBuff,
              typename BGlobalBuff,
              typename CThreadBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename AGridDesc_E0_E1_K0_K1_E2,
              typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
              typename CThreadDesc_K1_N_H2_W2,
              bool HasMainE0BlockLoop>
    __device__ static void GemmOpHasE0Loop(
        const AGlobalBuff& a_global_buf,
        const BGlobalBuff& b_global_buf,
        CThreadBuff& c_thread_buf,
        FloatAB* __restrict__ p_shared_block,
        const CBlockIndex& c_block_idx,
        const CThreadIndex& c_thread_idx,
        const AGridDesc_E0_E1_K0_K1_E2& a_e0_e1_k0_k1_e2_grid_desc,
        const BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2& b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
        const CThreadDesc_K1_N_H2_W2&,
        integral_constant<bool, HasMainE0BlockLoop>)
    {
        constexpr auto HasMainE1BlockLoop       = CalculateHasMainE1BlockLoop();
        constexpr auto HasDoubleTailE1BlockLoop = CalculateHasDoubleTailE1BlockLoop();

        const index_t k_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);
        const index_t n_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I1]);
        const index_t ho_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I2]);
        const index_t wo_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I3]);

        constexpr auto max_lds_align = Number<ABlockTransferDstScalarPerVector_E2>{};

        constexpr auto a_e1_k1_e2_block_gemm_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E1PerBlock>{}, Number<KPerBlock>{}, Number<E2>{}), max_lds_align);

        constexpr auto b_e1_n_h_w_e2_block_gemm_desc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<E1PerBlock>{},
                                                           I1,
                                                           Number<HoPerBlock>{},
                                                           Number<WoPerBlock>{},
                                                           Number<E2>{}));

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

        auto blockwise_gemm =
            BlockwiseGemmDlops_km_kn_m0m1n0n1_v3<BlockSize,
                                                 FloatAB,
                                                 FloatAB,
                                                 FloatAcc,
                                                 decltype(a_e1_k1_e2_block_gemm_desc),
                                                 decltype(b_e1_n_h_w_e2_block_gemm_desc),
                                                 decltype(c_k1_n_h2_w2_thread_gemm_desc),
                                                 EPerThread,
                                                 K2>{};

        const auto ho_thread_id = c_thread_idx[I2];
        const auto wo_thread_id = c_thread_idx[I3];

        constexpr auto a_e0_e1_k0_k1_e2_block_copy_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E0PerBlock>{}, Number<E1>{}, I1, Number<KPerBlock>{}, Number<E2>{}),
            max_lds_align);

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4r1<BlockSize,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              InMemoryDataOperationEnum_t::Set,
                                              Sequence<E0PerBlock, E1, I1, KPerBlock, E2>,
                                              ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
                                              ABlockTransferThreadClusterArrangeOrder,
                                              FloatAB,
                                              FloatAB,
                                              decltype(a_e0_e1_k0_k1_e2_grid_desc),
                                              decltype(a_e0_e1_k0_k1_e2_block_copy_desc),
                                              ABlockTransferSrcAccessOrder,
                                              Sequence<0, 1, 2, 3, 4>,
                                              ABlockTransferSrcVectorDim,
                                              4,
                                              ABlockTransferSrcScalarPerVector,
                                              ABlockTransferDstScalarPerVector_E2,
                                              1,
                                              1,
                                              AThreadTransferSrcResetCoordinateAfterRun,
                                              false>(
                a_e0_e1_k0_k1_e2_grid_desc,
                make_multi_index(0, 0, k_block_work_id, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{},
                a_e0_e1_k0_k1_e2_block_copy_desc,
                make_multi_index(0, 0, 0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<E1PerBlock>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<HoPerThread>{},
                                                           I1,
                                                           I1,
                                                           Number<WoPerThread>{},
                                                           Number<E2>{}));

        auto b_threadwise_transfer = ThreadwiseTensorSliceTransfer_v2<
            FloatAB,
            FloatAB,
            BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc),
            Sequence<I1, E1PerBlock, I1, I1, I1, HoPerThread, I1, I1, WoPerThread, E2>,
            BBlockTransferSrcAccessOrder,
            BBlockTransferSrcVectorDim,
            BBlockTransferSrcScalarPerVector,
            BThreadTransferSrcResetCoordinateAfterRun,
            true>(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                  make_multi_index(0,
                                   0,
                                   n_block_work_id,
                                   ho_block_work_id,
                                   ho_thread_id,
                                   0,
                                   wo_block_work_id,
                                   wo_thread_id,
                                   0,
                                   0));

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_shared_block, a_e0_e1_k0_k1_e2_block_copy_desc.GetElementSpaceSize());

        constexpr auto b_thread_slice_copy_step =
            make_multi_index(0, E1PerBlock, 0, 0, 0, 0, 0, 0, 0, 0);

        // double regsiter buffer for b
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatAB,
                     b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc.GetElementSpaceSize(),
                     true>
            b_thread_even_buf, b_thread_odd_buf;

        // LDS double buffer: preload data
        {
            a_blockwise_copy.RunRead(a_e0_e1_k0_k1_e2_grid_desc, a_global_buf);
            a_blockwise_copy.RunWrite(a_e0_e1_k0_k1_e2_block_copy_desc, a_block_buf);
        }

        __syncthreads();

        index_t e0_block_data_begin = 0;
        do
        {
            b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                      b_global_buf,
                                      b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                      make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                      b_thread_even_buf);

            index_t b_copy_e_offset = 0;

            if constexpr(HasMainE1BlockLoop)
            {
                static_for<0, E1 - 2 * E1PerBlock, 2 * E1PerBlock>{}([&](auto) {
                    // even iteration
                    b_threadwise_transfer.MoveSrcSliceWindow(
                        b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc, b_thread_slice_copy_step);
                    b_copy_e_offset += E1PerBlock;

                    b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              b_global_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                              b_thread_odd_buf);

                    // LDS double buffer: GEMM on current data
                    blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                    blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                    b_threadwise_transfer.MoveSrcSliceWindow(
                        b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc, b_thread_slice_copy_step);
                    b_copy_e_offset += E1PerBlock;

                    b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              b_global_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                              b_thread_even_buf);

                    // LDS double buffer: GEMM on current data
                    blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);

                    blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));
                });
            }

            // LDS double buffer: tail
            if constexpr(HasDoubleTailE1BlockLoop) // if has 2 iteration left
            {
                b_threadwise_transfer.MoveSrcSliceWindow(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                                         b_thread_slice_copy_step);
                b_copy_e_offset += E1PerBlock;

                b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                          b_global_buf,
                                          b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                          make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                          b_thread_odd_buf);

                // LDS double buffer: GEMM on 2nd-last data
                blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);
            }
            else // if has 1 iteration left
            {
                // LDS double buffer: GEMM on last data
                blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);
            }

            // if constexpr(e0_block_data_begin < Number<E0PerBlock - 1>{})
            {
                blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                b_threadwise_transfer.MoveSrcSliceWindow(
                    b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                    make_multi_index(I1, -b_copy_e_offset, 0, 0, 0, 0, 0, 0, 0, 0));
            }
            e0_block_data_begin++;
        } while(e0_block_data_begin < E0PerBlock);
    }

    template <typename AGlobalBuff,
              typename BGlobalBuff,
              typename CThreadBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename AGridDesc_E0_E1_K0_K1_E2,
              typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
              typename CThreadDesc_K1_N_H2_W2,
              bool HasMainE0BlockLoop>
    __device__ static void GemmOpHasE0BlockLoop(
        const AGlobalBuff& a_global_buf,
        const BGlobalBuff& b_global_buf,
        CThreadBuff& c_thread_buf,
        FloatAB* __restrict__ p_shared_block,
        const CBlockIndex& c_block_idx,
        const CThreadIndex& c_thread_idx,
        const AGridDesc_E0_E1_K0_K1_E2& a_e0_e1_k0_k1_e2_grid_desc,
        const BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2& b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
        const CThreadDesc_K1_N_H2_W2&,
        integral_constant<bool, HasMainE0BlockLoop>)
    {
        constexpr auto HasMainE1BlockLoop       = CalculateHasMainE1BlockLoop();
        constexpr auto HasDoubleTailE1BlockLoop = CalculateHasDoubleTailE1BlockLoop();

        const index_t k_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);
        const index_t n_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I1]);
        const index_t ho_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I2]);
        const index_t wo_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I3]);

        constexpr auto max_lds_align = Number<ABlockTransferDstScalarPerVector_E2>{};

        constexpr auto a_e1_k1_e2_block_gemm_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E1PerBlock>{}, Number<KPerBlock>{}, Number<E2>{}), max_lds_align);

        constexpr auto b_e1_n_h_w_e2_block_gemm_desc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<E1PerBlock>{},
                                                           I1,
                                                           Number<HoPerBlock>{},
                                                           Number<WoPerBlock>{},
                                                           Number<E2>{}));

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

        auto blockwise_gemm =
            BlockwiseGemmDlops_km_kn_m0m1n0n1_v3<BlockSize,
                                                 FloatAB,
                                                 FloatAB,
                                                 FloatAcc,
                                                 decltype(a_e1_k1_e2_block_gemm_desc),
                                                 decltype(b_e1_n_h_w_e2_block_gemm_desc),
                                                 decltype(c_k1_n_h2_w2_thread_gemm_desc),
                                                 EPerThread,
                                                 K2>{};

        const auto ho_thread_id = c_thread_idx[I2];
        const auto wo_thread_id = c_thread_idx[I3];

        constexpr auto a_e0_e1_k0_k1_e2_block_copy_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E0PerBlock>{}, Number<E1>{}, I1, Number<KPerBlock>{}, Number<E2>{}),
            max_lds_align);

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4r1<BlockSize,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              InMemoryDataOperationEnum_t::Set,
                                              Sequence<E0PerBlock, E1, I1, KPerBlock, E2>,
                                              ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
                                              ABlockTransferThreadClusterArrangeOrder,
                                              FloatAB,
                                              FloatAB,
                                              decltype(a_e0_e1_k0_k1_e2_grid_desc),
                                              decltype(a_e0_e1_k0_k1_e2_block_copy_desc),
                                              ABlockTransferSrcAccessOrder,
                                              Sequence<0, 1, 2, 3, 4>,
                                              ABlockTransferSrcVectorDim,
                                              4,
                                              ABlockTransferSrcScalarPerVector,
                                              ABlockTransferDstScalarPerVector_E2,
                                              1,
                                              1,
                                              AThreadTransferSrcResetCoordinateAfterRun,
                                              false>(
                a_e0_e1_k0_k1_e2_grid_desc,
                make_multi_index(0, 0, k_block_work_id, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{},
                a_e0_e1_k0_k1_e2_block_copy_desc,
                make_multi_index(0, 0, 0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<E1PerBlock>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<HoPerThread>{},
                                                           I1,
                                                           I1,
                                                           Number<WoPerThread>{},
                                                           Number<E2>{}));

        auto b_threadwise_transfer = ThreadwiseTensorSliceTransfer_v2<
            FloatAB,
            FloatAB,
            BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc),
            Sequence<I1, E1PerBlock, I1, I1, I1, HoPerThread, I1, I1, WoPerThread, E2>,
            BBlockTransferSrcAccessOrder,
            BBlockTransferSrcVectorDim,
            BBlockTransferSrcScalarPerVector,
            BThreadTransferSrcResetCoordinateAfterRun,
            true>(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                  make_multi_index(0,
                                   0,
                                   n_block_work_id,
                                   ho_block_work_id,
                                   ho_thread_id,
                                   0,
                                   wo_block_work_id,
                                   wo_thread_id,
                                   0,
                                   0));

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_shared_block, a_e0_e1_k0_k1_e2_block_copy_desc.GetElementSpaceSize());

        constexpr auto b_thread_slice_copy_step =
            make_multi_index(0, E1PerBlock, 0, 0, 0, 0, 0, 0, 0, 0);

        // double regsiter buffer for b
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatAB,
                     b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc.GetElementSpaceSize(),
                     true>
            b_thread_even_buf, b_thread_odd_buf;

        const auto E0 = b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetLength(I0);

        index_t e0_block_data_begin = 0;

        do
        {

            // LDS double buffer: preload data
            {
                a_blockwise_copy.RunRead(a_e0_e1_k0_k1_e2_grid_desc, a_global_buf);
                a_blockwise_copy.RunWrite(a_e0_e1_k0_k1_e2_block_copy_desc, a_block_buf);
            }

            __syncthreads();

            index_t gemm_e_offset = 0;

            static_for<0, E0PerBlock, 1>{}([&](auto) {
                b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                          b_global_buf,
                                          b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                          make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                          b_thread_even_buf);

                index_t b_copy_e_offset = 0;

                if constexpr(HasMainE1BlockLoop)
                {
                    static_for<0, E1 - 2 * E1PerBlock, 2 * E1PerBlock>{}([&](auto) {
                        // even iteration
                        b_threadwise_transfer.MoveSrcSliceWindow(
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc, b_thread_slice_copy_step);
                        b_copy_e_offset += E1PerBlock;

                        b_threadwise_transfer.Run(
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                            b_global_buf,
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                            make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                            b_thread_odd_buf);

                        // LDS double buffer: GEMM on current data
                        blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                        blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));
                        gemm_e_offset += E1PerBlock;

                        b_threadwise_transfer.MoveSrcSliceWindow(
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc, b_thread_slice_copy_step);
                        b_copy_e_offset += E1PerBlock;

                        b_threadwise_transfer.Run(
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                            b_global_buf,
                            b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                            make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                            b_thread_even_buf);

                        // LDS double buffer: GEMM on current data
                        blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);

                        blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));
                        gemm_e_offset += E1PerBlock;
                    });
                }

                // LDS double buffer: tail
                if constexpr(HasDoubleTailE1BlockLoop) // if has 2 iteration left
                {
                    b_threadwise_transfer.MoveSrcSliceWindow(
                        b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc, b_thread_slice_copy_step);
                    b_copy_e_offset += E1PerBlock;

                    b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              b_global_buf,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                              b_thread_odd_buf);

                    // LDS double buffer: GEMM on 2nd-last data
                    blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                    blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));
                    gemm_e_offset += E1PerBlock;

                    // LDS double buffer: GEMM on last data
                    blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);
                }
                else // if has 1 iteration left
                {
                    // LDS double buffer: GEMM on last data
                    blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);
                }

                // if constexpr(e0_block_data_begin < Number<E0PerBlock - 1>{})
                {
                    blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));
                    gemm_e_offset += E1PerBlock;

                    b_threadwise_transfer.MoveSrcSliceWindow(
                        b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                        make_multi_index(I1, -b_copy_e_offset, 0, 0, 0, 0, 0, 0, 0, 0));
                }
            });

            blockwise_gemm.MoveABlockSliceWindow(make_tuple(-gemm_e_offset, 0, 0));

            a_blockwise_copy.MoveSrcSliceWindow(a_e0_e1_k0_k1_e2_grid_desc,
                                                make_multi_index(E0PerBlock, 0, 0, 0, 0));

            e0_block_data_begin += E0PerBlock;

        } while(e0_block_data_begin < E0);
    }

    template <typename AGlobalBuff,
              typename BGlobalBuff,
              typename CThreadBuff,
              typename CBlockIndex,
              typename CThreadIndex,
              typename AGridDesc_E0_E1_K0_K1_E2,
              typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
              typename CThreadDesc_K1_N_H2_W2,
              bool HasMainE0BlockLoop>
    __device__ static void
    GemmOp(const AGlobalBuff& a_global_buf,
           const BGlobalBuff& b_global_buf,
           CThreadBuff& c_thread_buf,
           FloatAB* __restrict__ p_shared_block,
           const CBlockIndex& c_block_idx,
           const CThreadIndex& c_thread_idx,
           const AGridDesc_E0_E1_K0_K1_E2& a_e0_e1_k0_k1_e2_grid_desc,
           const BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2& b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
           const CThreadDesc_K1_N_H2_W2&,
           integral_constant<bool, HasMainE0BlockLoop>)
    {
        constexpr auto HasMainE1BlockLoop       = CalculateHasMainE1BlockLoop();
        constexpr auto HasDoubleTailE1BlockLoop = CalculateHasDoubleTailE1BlockLoop();

        const index_t k_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I0]);
        const index_t n_block_work_id  = __builtin_amdgcn_readfirstlane(c_block_idx[I1]);
        const index_t ho_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I2]);
        const index_t wo_block_work_id = __builtin_amdgcn_readfirstlane(c_block_idx[I3]);

        constexpr auto max_lds_align = Number<ABlockTransferDstScalarPerVector_E2>{};

        constexpr auto a_e1_k1_e2_block_gemm_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E1PerBlock>{}, Number<KPerBlock>{}, Number<E2>{}), max_lds_align);

        constexpr auto b_e1_n_h_w_e2_block_gemm_desc =
            make_naive_tensor_descriptor_packed(make_tuple(Number<E1PerBlock>{},
                                                           I1,
                                                           Number<HoPerBlock>{},
                                                           Number<WoPerBlock>{},
                                                           Number<E2>{}));

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = CThreadDesc_K1_N_H2_W2{};

        auto blockwise_gemm =
            BlockwiseGemmDlops_km_kn_m0m1n0n1_v3<BlockSize,
                                                 FloatAB,
                                                 FloatAB,
                                                 FloatAcc,
                                                 decltype(a_e1_k1_e2_block_gemm_desc),
                                                 decltype(b_e1_n_h_w_e2_block_gemm_desc),
                                                 decltype(c_k1_n_h2_w2_thread_gemm_desc),
                                                 EPerThread,
                                                 K2>{};

        const auto ho_thread_id = c_thread_idx[I2];
        const auto wo_thread_id = c_thread_idx[I3];

#if 1
        constexpr auto a_e0_e1_k0_k1_e2_block_copy_desc = make_naive_tensor_descriptor_aligned(
            make_tuple(Number<E0PerBlock>{}, Number<E1>{}, I1, Number<KPerBlock>{}, Number<E2>{}),
            max_lds_align);

        // A matrix blockwise copy
        auto a_blockwise_copy =
            BlockwiseTensorSliceTransfer_v4r1<BlockSize,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              InMemoryDataOperationEnum_t::Set,
                                              Sequence<E0PerBlock, E1, I1, KPerBlock, E2>,
                                              ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
                                              ABlockTransferThreadClusterArrangeOrder,
                                              FloatAB,
                                              FloatAB,
                                              decltype(a_e0_e1_k0_k1_e2_grid_desc),
                                              decltype(a_e0_e1_k0_k1_e2_block_copy_desc),
                                              ABlockTransferSrcAccessOrder,
                                              Sequence<0, 1, 2, 3, 4>,
                                              ABlockTransferSrcVectorDim,
                                              4,
                                              ABlockTransferSrcScalarPerVector,
                                              ABlockTransferDstScalarPerVector_E2,
                                              1,
                                              1,
                                              AThreadTransferSrcResetCoordinateAfterRun,
                                              false>(
                a_e0_e1_k0_k1_e2_grid_desc,
                make_multi_index(0, 0, k_block_work_id, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{},
                a_e0_e1_k0_k1_e2_block_copy_desc,
                make_multi_index(0, 0, 0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});
#endif

        constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc =
            make_naive_tensor_descriptor_packed(make_tuple(I1,
                                                           Number<E1PerBlock>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<HoPerThread>{},
                                                           I1,
                                                           I1,
                                                           Number<WoPerThread>{},
                                                           Number<E2>{}));

        auto b_threadwise_transfer = ThreadwiseTensorSliceTransfer_v2<
            FloatAB,
            FloatAB,
            BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc),
            Sequence<I1, E1PerBlock, I1, I1, I1, HoPerThread, I1, I1, WoPerThread, E2>,
            BBlockTransferSrcAccessOrder,
            BBlockTransferSrcVectorDim,
            BBlockTransferSrcScalarPerVector,
            BThreadTransferSrcResetCoordinateAfterRun,
            true>(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                  make_multi_index(0,
                                   0,
                                   n_block_work_id,
                                   ho_block_work_id,
                                   ho_thread_id,
                                   0,
                                   wo_block_work_id,
                                   wo_thread_id,
                                   0,
                                   0));

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
            p_shared_block, a_e0_e1_k0_k1_e2_block_copy_desc.GetElementSpaceSize());

        constexpr auto b_thread_slice_copy_step =
            make_multi_index(0, E1PerBlock, 0, 0, 0, 0, 0, 0, 0, 0);

        // double regsiter buffer for b
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatAB,
                     b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc.GetElementSpaceSize(),
                     true>
            b_thread_even_buf, b_thread_odd_buf;

        // LDS double buffer: preload data
        {
            a_blockwise_copy.RunRead(a_e0_e1_k0_k1_e2_grid_desc, a_global_buf);
            a_blockwise_copy.RunWrite(a_e0_e1_k0_k1_e2_block_copy_desc, a_block_buf);
        }

        __syncthreads();

        b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                  b_global_buf,
                                  b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                  make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                  b_thread_even_buf);

        if constexpr(HasMainE1BlockLoop)
        {
            index_t e1_block_data_begin = 0;

            do
            {
                b_threadwise_transfer.MoveSrcSliceWindow(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                                         b_thread_slice_copy_step);

                b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                          b_global_buf,
                                          b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                          make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                          b_thread_odd_buf);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

                blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                b_threadwise_transfer.MoveSrcSliceWindow(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                                         b_thread_slice_copy_step);

                b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                          b_global_buf,
                                          b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                          make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                          b_thread_even_buf);

                // LDS double buffer: GEMM on current data
                blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);

                blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

                e1_block_data_begin += 2 * E1PerBlock;
            } while(e1_block_data_begin < E1 - 2 * E1PerBlock);
        }

        // LDS double buffer: tail
        if constexpr(HasDoubleTailE1BlockLoop) // if has 2 iteration left
        {
            b_threadwise_transfer.MoveSrcSliceWindow(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                                     b_thread_slice_copy_step);

            b_threadwise_transfer.Run(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                      b_global_buf,
                                      b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_thread_copy_desc,
                                      make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                      b_thread_odd_buf);

            // LDS double buffer: GEMM on 2nd-last data
            blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);

            blockwise_gemm.MoveABlockSliceWindow(make_tuple(E1PerBlock, 0, 0));

            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(a_block_buf, b_thread_odd_buf, c_thread_buf);
        }
        else // if has 1 iteration left
        {
            // LDS double buffer: GEMM on last data
            blockwise_gemm.Run(a_block_buf, b_thread_even_buf, c_thread_buf);
        }
    }

    template <typename AGridDesc_E0_E1_K0_K1_E2,
              typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
              typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
              typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
              bool HasMainE0BlockLoop>
    __device__ static void GroupConv(
        const FloatAB* __restrict__ p_a_global,
        const FloatAB* __restrict__ p_b_global,
        FloatC* __restrict__ p_c_global,
        FloatAB* __restrict__ p_shared_block,
        const index_t group_grid_size,
        const AGridDesc_E0_E1_K0_K1_E2& a_e0_e1_k0_k1_e2_grid_desc,
        const BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2& b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
        const CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2& c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
        const CBlockIdToBlockClusterAdaptor_K_N_H_W& c_blockid_to_k_n_h_w_block_cluster_adaptor,
        integral_constant<bool, HasMainE0BlockLoop>)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_a_global, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_b_global, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());
        auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_c_global, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = MakeCK1NH2W2ThreadDescriptor();

        // register allocation for output
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatAcc,
                     c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                     true>
            c_thread_buf;

        const index_t block_id = get_block_1d_id();

        const auto c_k_n_h_w_block_cluster_idx =
            GetCBlockIndex(c_blockid_to_k_n_h_w_block_cluster_adaptor, block_id % group_grid_size);

        const auto c_thread_mtx_index = GetCThreadIndex();

        static_for<0, c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(), 1>{}(
            [&](auto i) { c_thread_buf(i) = 0; });

        // GemmOp
        GemmOp(a_global_buf,
               b_global_buf,
               c_thread_buf,
               p_shared_block,
               c_k_n_h_w_block_cluster_idx,
               c_thread_mtx_index,
               a_e0_e1_k0_k1_e2_grid_desc,
               b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
               c_k1_n_h2_w2_thread_gemm_desc,
               integral_constant<bool, HasMainE0BlockLoop>{});

        // Output
        WriteOut(c_thread_buf,
                 c_global_buf,
                 c_k_n_h_w_block_cluster_idx,
                 c_thread_mtx_index,
                 c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
    }

    template <typename AGridDesc_E0_E1_K0_K1_E2,
              typename BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2,
              typename CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2,
              typename CBlockIdToBlockClusterAdaptor_K_N_H_W,
              bool HasMainE0BlockLoop,
              ActivTypeEnum_t ActivType>
    __device__ static void GroupConvBiasActiv(
        const FloatAB* __restrict__ p_a_global,
        const FloatAB* __restrict__ p_b_global,
        const FloatC* __restrict__ p_bias_global,
        FloatC* __restrict__ p_c_global,
        FloatAB* __restrict__ p_shared_block,
        const index_t group_grid_size,
        const AGridDesc_E0_E1_K0_K1_E2& a_e0_e1_k0_k1_e2_grid_desc,
        const BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2& b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
        const CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2& c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
        const CBlockIdToBlockClusterAdaptor_K_N_H_W& c_blockid_to_k_n_h_w_block_cluster_adaptor,
        integral_constant<bool, HasMainE0BlockLoop>,
        integral_constant<ActivTypeEnum_t, ActivType>)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_a_global, a_e0_e1_k0_k1_e2_grid_desc.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_b_global, b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.GetElementSpaceSize());
        auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_c_global, c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.GetElementSpaceSize());

        constexpr auto c_k1_n_h2_w2_thread_gemm_desc = MakeCK1NH2W2ThreadDescriptor();

        const auto bias_k0_k1_grid_desc =
            MakeBiasK0K1GridDescriptor(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);

        const auto bias_global_buf = make_dynamic_buffer<AddressSpaceEnum_t::Global>(
            p_bias_global, bias_k0_k1_grid_desc.GetElementSpaceSize());

        // register allocation for output
        StaticBuffer<AddressSpaceEnum_t::Vgpr,
                     FloatAcc,
                     c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(),
                     true>
            c_thread_buf;

        const index_t block_id = get_block_1d_id();

        const auto c_k_n_h_w_block_cluster_idx =
            GetCBlockIndex(c_blockid_to_k_n_h_w_block_cluster_adaptor, block_id % group_grid_size);

        const auto c_thread_mtx_index = GetCThreadIndex();

        static_for<0, c_k1_n_h2_w2_thread_gemm_desc.GetElementSpaceSize(), 1>{}(
            [&](auto i) { c_thread_buf(i) = 0; });

        // GemmOp
        GemmOp(a_global_buf,
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
        BiasOp(bias_global_buf,
               c_thread_buf,
               c_k_n_h_w_block_cluster_idx,
               c_thread_mtx_index,
               bias_k0_k1_grid_desc,
               c_k1_n_h2_w2_thread_gemm_desc);

        static constexpr auto activ_type = integral_constant<ActivTypeEnum_t, ActivType>{};

        // Activ
        Activation(c_thread_buf, c_k1_n_h2_w2_thread_gemm_desc, activ_type);

        // Output
        WriteOut(c_thread_buf,
                 c_global_buf,
                 c_k_n_h_w_block_cluster_idx,
                 c_thread_mtx_index,
                 c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
    }
};

} // namespace ck
#endif
