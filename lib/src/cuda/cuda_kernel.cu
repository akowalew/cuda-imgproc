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

void cuda_kernel_mean_blurr(CudaKernel& kernel)
{
    // Fill kernel with same (meaning) value
    const auto ksize = kernel.size;
    const auto ksize_sq = (ksize * ksize);
    const auto kernel_v = (1.0f / ksize_sq);

    LOG_INFO("Making CUDA mean blurr kernel of size %lux%lu\n", ksize, ksize);
    LOG_DEBUG("kernel_v: %f\n", kernel_v);

    auto vec = std::vector<float>(ksize_sq, kernel_v);
    const auto buffer_size = (ksize_sq * sizeof(CudaKernel::Type));
    checkCudaErrors(cudaMemcpy(kernel.data, vec.data(), buffer_size, cudaMemcpyHostToDevice));
}
