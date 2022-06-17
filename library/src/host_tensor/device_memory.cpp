#include "device_memory.hpp"
#include "hip_check_error.hpp"

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    hip_check_error(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void* DeviceMem::GetDeviceBuffer() { return mpDeviceBuf; }

std::size_t DeviceMem::GetBufferSize() { return mMemSize; }

void DeviceMem::ToDevice(const void* p)
{
    hip_check_error(hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}

void DeviceMem::FromDevice(void* p)
{
    hip_check_error(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}

void DeviceMem::SetZero() { hip_check_error(hipMemset(mpDeviceBuf, 0, mMemSize)); }

DeviceMem::~DeviceMem() { hip_check_error(hipFree(mpDeviceBuf)); }
