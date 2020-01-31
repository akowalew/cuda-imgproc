///////////////////////////////////////////////////////////////////////////////
// cuda_host_image.cu
//
// Contains definitions for CUDA host images manager
///////////////////////////////////////////////////////////////////////////////

#include "cuda_host_image.cuh"

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

CudaHostImage cuda_create_host_image(size_t cols, size_t rows)
{
	return create_image(cols, rows);
}

void cuda_free_host_image(CudaHostImage& h_img)
{
	free_image(h_img);	
}

void cuda_host_image_register(const CudaHostImage& h_img)
{
	const auto cols = h_img.cols;
	const auto rows = h_img.rows;
	const auto data = h_img.data;

	LOG_INFO("Registering host image %lux%lu at 0x%p\n", cols, rows, data);

	const auto size = (cols * rows * sizeof(uchar));
	const auto flags = cudaHostRegisterDefault;
	checkCudaErrors(cudaHostRegister(data, size, flags));
}

void cuda_host_image_unregister(const CudaHostImage& h_img)
{
	const auto data = h_img.data;
	
	LOG_INFO("Unregistering host image 0x%p\n", data);
	
	checkCudaErrors(cudaHostUnregister(data));
}