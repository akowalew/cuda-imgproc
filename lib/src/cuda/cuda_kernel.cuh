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
	size_t size;
};


CudaKernel cuda_create_kernel(size_t ksize);

void cuda_free_kernel(CudaKernel& kernel);


void cuda_kernel_mean_blurr(CudaKernel& dst);