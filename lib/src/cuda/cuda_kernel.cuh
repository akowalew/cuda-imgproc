///////////////////////////////////////////////////////////////////////////////
// cuda_kernel.cuh
//
// Contains declarations for CUDA kernels manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

struct CudaKernel
{
	using Type = float;

	Type* data;
	size_t ksize;
};

CudaKernel cuda_create_kernel(size_t ksize);

void cuda_free_kernel(CudaKernel& kernel);


void cuda_kernel_fill(CudaKernel& kernel, CudaKernel::Type value);


CudaKernel cuda_create_mean_blurr_kernel(size_t ksize);