///////////////////////////////////////////////////////////////////////////////
// cuda_kernel.cu
//
// Contains definitions for CUDA kernels manager
///////////////////////////////////////////////////////////////////////////////

#include "cuda_kernel.cuh"

#include <cstdio>

#include <vector>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

CudaKernel cuda_create_kernel(size_t ksize)
{
    LOG_INFO("Creating CUDA kernel of size %lux%lu\n", ksize, ksize);

    // Get size of the buffer for kernel
    const auto buffer_size = (ksize * ksize * sizeof(CudaKernel::Type));

    // Allocate kernel on the device
    void* data;
    checkCudaErrors(cudaMalloc(&data, buffer_size));

    // Return created kernel
    return CudaKernel { (CudaKernel::Type*)data, ksize };
}

void cuda_free_kernel(CudaKernel& kernel)
{
    // Release kernel on the device
    checkCudaErrors(cudaFree(kernel.data));
}

void cuda_host_kernel_register(const Kernel& kernel)
{
    const auto ksize = kernel.cols;
    const auto buffer_size = (ksize * ksize * sizeof(float));
    const auto data = kernel.data;
    const auto flags = cudaHostRegisterDefault;
    checkCudaErrors(cudaHostRegister(data, buffer_size, flags));
}

void cuda_host_kernel_unregister(const Kernel& kernel)
{
    const auto data = kernel.data;
    checkCudaErrors(cudaHostUnregister(data));
}
