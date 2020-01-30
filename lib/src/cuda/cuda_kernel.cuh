///////////////////////////////////////////////////////////////////////////////
// cuda_kernel.cuh
//
// Contains declarations for CUDA kernels manager
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "kernel.hpp"

struct CudaKernel
{
	using Type = float;

	Type* data;
	size_t size;
};


CudaKernel cuda_create_kernel(size_t ksize);

void cuda_free_kernel(CudaKernel& kernel);

void cuda_host_kernel_register(const Kernel& kernel);

void cuda_host_kernel_unregister(const Kernel& kernel);
