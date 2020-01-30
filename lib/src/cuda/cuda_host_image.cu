///////////////////////////////////////////////////////////////////////////////
// cuda_host_image.cu
//
// Contains definitions for CUDA host images manager
///////////////////////////////////////////////////////////////////////////////

#include "cuda_host_image.cuh"

#include <cuda_runtime.h>

#include <helper_cuda.h>

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
	const auto size = (cols * rows * sizeof(uchar));
	const auto data = h_img.data;
	const auto flags = cudaHostRegisterDefault;
	checkCudaErrors(cudaHostRegister(data, size, flags));
}

void cuda_host_image_unregister(const CudaHostImage& h_img)
{
	const auto data = h_img.data;
	checkCudaErrors(cudaHostUnregister(data));
}