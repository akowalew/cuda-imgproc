///////////////////////////////////////////////////////////////////////////////
// cuda_kernel.cuh
//
// Contains declarations for CUDA kernels manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

using CudaKernelSize = size_t;

using CudaKernelType = float;

struct CudaKernel
{
	CudaKernelType* data;
	CudaKernelSize ksize;
};

CudaKernel cuda_create_kernel(CudaKernelSize ksize);

void cuda_free_kernel(CudaKernel& kernel);

void cuda_kernel_fill(CudaKernel& kernel, CudaKernelType value);

CudaKernel cuda_create_mean_blurr_kernel(CudaKernelSize ksize);