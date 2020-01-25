///////////////////////////////////////////////////////////////////////////////
// cuda_lut.cu
//
// Contains definitions for CUDA LUTs manager
///////////////////////////////////////////////////////////////////////////////

#include "cuda_lut.cuh"

#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

//! Number of threads in block in each dimension
constexpr auto K = 32;

CudaLUT cuda_create_lut()
{
	// Get size of the buffer for the lut
	const auto size = (LUTSize * sizeof(uchar));

	// Allocate LUT on the device
	void* data;
	checkCudaErrors(cudaMalloc(&data, size));

	// Return created device LUT
	return { (uchar*) data };
}

void cuda_free_lut(CudaLUT& lut)
{
	// Release device LUT
	checkCudaErrors(cudaFree(lut.data));
}

__global__
void cuda_apply_lut(
	uchar* dst, size_t dpitch,
	const uchar* src, size_t spitch,
	size_t cols, size_t rows,
	const uchar* lut)
{
    // Get position of that thread in terms of the image
    const auto y = (threadIdx.y + blockDim.y*blockIdx.y);
    const auto x = (threadIdx.x + blockDim.x*blockIdx.x);

    // If we are out of bounds, do nothing
    if(y >= rows || x >= cols)
    {
        return;
    }

    // Apply LUT on src image pixel and store result into dst image pixel
    const auto src_v = src[x + y*spitch];
    const auto lut_v = lut[src_v];
    dst[x + y*dpitch] = lut_v;
}

void cuda_apply_lut_async(CudaImage& dst, const CudaImage& src, const CudaLUT& lut)
{
	// Ensure proper images size
	assert(src.cols == dst.cols);
	assert(src.rows == dst.rows);

	// Retrieve device image shape
	const auto cols = src.cols;
	const auto rows = src.rows;

	LOG_INFO("Applying LUT with CUDA on image %lux%lu\n", cols, rows);

	// Use const sized 2D blocks
	const auto dim_block = dim3(K, K);

	// Use as much blocks in the grid as needed
	const auto dim_grid_x = ((cols+K-1) / K);
	const auto dim_grid_y = ((rows+K-1) / K);
	const auto dim_grid = dim3(dim_grid_x, dim_grid_y);

	// Launch LUT applying 
	cuda_apply_lut<<<dim_grid, dim_block>>>(
		(uchar*)dst.data, dst.pitch,
		(uchar*)src.data, src.pitch,
		cols, rows,
		lut.data);

	// Check if launch succeeded
	checkCudaErrors(cudaGetLastError());
}

void cuda_apply_lut(CudaImage& dst, const CudaImage& src, const CudaLUT& lut)
{
	// Launch applying of the LUT
	cuda_apply_lut_async(dst, src, lut);

	// Wait for device finish
	checkCudaErrors(cudaDeviceSynchronize());
}

CudaImage cuda_apply_lut(const CudaImage& src, const CudaLUT& lut)
{
	// Allocate image on the device
	auto dst = cuda_create_image(src.cols, src.rows);

	// Perform LUT application
	cuda_apply_lut(dst, src, lut);

	// Return image after LUT apply
	return dst;
}