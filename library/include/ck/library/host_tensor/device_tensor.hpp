// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "host_tensor.hpp"

template <typename TensorDesc>
void ostream_tensor_descriptor(TensorDesc, std::ostream& os = std::cout)
{
    ostream_HostTensorDescriptor(make_HostTensorDescriptor(TensorDesc{}), os);
}
