///////////////////////////////////////////////////////////////////////////////
// cuda_kernel.cu
//
// Contains definitions for CUDA kernels manager
///////////////////////////////////////////////////////////////////////////////

#include "cuda_kernel.cuh"

#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

CudaKernel cuda_create_kernel(size_t ksize)
{
    LOG_INFO("Creating CUDA kernel of size %lux%lu\n", ksize, ksize);

    // Get size of the buffer for kernel
    const auto size = (ksize * ksize * sizeof(CudaKernel::Type));

    // Allocate kernel on the device
    void* data;
    checkCudaErrors(cudaMalloc(&data, size));

    // Return created kernel
    return CudaKernel { (CudaKernel::Type*)data, ksize };
}

void cuda_free_kernel(CudaKernel& kernel)
{
    // Release kernel on the device
    checkCudaErrors(cudaFree(kernel.data));
}

void cuda_kernel_fill(CudaKernel& kernel, CudaKernel::Type value)
{
    const auto ksize = kernel.size;
    const auto count = (ksize * ksize * sizeof(CudaKernel::Type));

    checkCudaErrors(cudaMemset(kernel.data, value, count));
}

CudaKernel cuda_create_mean_blurr_kernel(size_t ksize)
{
    auto kernel = cuda_create_kernel(ksize);

    // Fill kernel with same (meaning) value
    const auto ksize_sq = (ksize * ksize);
    const auto kernel_v = (1.0f / ksize_sq);
    cuda_kernel_fill(kernel, kernel_v);

    return kernel;
}
