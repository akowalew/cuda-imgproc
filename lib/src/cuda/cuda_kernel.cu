///////////////////////////////////////////////////////////////////////////////
// cuda_kernel.cu
//
// Contains definitions for CUDA kernels manager
///////////////////////////////////////////////////////////////////////////////

#include "cuda_kernel.cuh"

#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

CudaKernel cuda_create_kernel(CudaKernelSize ksize)
{
    printf("*** Creating CUDA kernel of size %lux%lu\n", ksize, ksize);

    // Get size of the buffer for kernel
    const auto size = (ksize * ksize * sizeof(CudaKernelType));

    // Allocate kernel on the device
    void* data;
    checkCudaErrors(cudaMalloc(&data, size));

    // Return created kernel
    return CudaKernel { (CudaKernelType*)data, ksize };
}

void cuda_free_kernel(CudaKernel& kernel)
{
    // Release kernel on the device
    checkCudaErrors(cudaFree(kernel.data));
}

void cuda_kernel_fill(CudaKernel& kernel, CudaKernelType value)
{
    const auto ksize = kernel.ksize;
    const auto count = (ksize * ksize * sizeof(CudaKernelType));

    static_assert(sizeof(int) == sizeof(CudaKernelType), 
        "cudaMemset API accepts only int sized variable");

    checkCudaErrors(cudaMemset(kernel.data, value, count));
}

CudaKernel cuda_create_mean_blurr_kernel(CudaKernelSize ksize)
{
    auto kernel = cuda_create_kernel(ksize);

    // Fill kernel with same (meaning) value
    const auto ksize_sq = (ksize * ksize);
    const auto kernel_v = (1.0f / ksize_sq);
    cuda_kernel_fill(kernel, kernel_v);

    return kernel;
}
