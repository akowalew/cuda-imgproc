///////////////////////////////////////////////////////////////////////////////
// cuda_image.cuh
//
// Contains definitions for CUDA image manager
///////////////////////////////////////////////////////////////////////////////

#include "cuda_image.cuh"

#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

CudaImage cuda_create_image(size_t cols, size_t rows)
{
	LOG_INFO("Creating CUDA image of size %lux%lu\n", cols, rows);

	// Calculate parameters needed by CUDA
	const auto width = (cols * sizeof(uchar));
	const auto height = rows;

	// Perform pitched memory allocation
	void* data;
	size_t pitch;
	checkCudaErrors(cudaMallocPitch(&data, &pitch, width, height));

	LOG_INFO("Created CUDA image at 0x%p and pitch %lu\n", data, pitch);

	// Return created image
	return CudaImage { data, pitch, cols, rows };
}

void cuda_free_image(CudaImage& d_img)
{
	LOG_INFO("Freeing CUDA image at %p\n", d_img.data);

	checkCudaErrors(cudaFree(d_img.data));
}

void cuda_image_copy(CudaImage& d_dst, const CudaImage& d_src)
{
	// Retrieve device image shape
	const auto cols = d_dst.cols;
	const auto rows = d_dst.rows;

	LOG_INFO("Copying CUDA image of size %lux%lu\n", cols, rows);

	// Calculate parameters needed by cuda
	const auto width = (cols * sizeof(uchar));
	const auto height = rows;

	// Perform data copy on device
	checkCudaErrors(cudaMemcpy2D(d_dst.data, d_dst.pitch,
		d_src.data, d_src.pitch, width, height, cudaMemcpyDeviceToDevice));
}

void cuda_image_copy_from_host_async(CudaImage& d_dst, const CudaHostImage& h_src)
{
	// Destination image cannot be smaller than source
	assert(d_dst.cols >= h_src.cols);
	assert(d_dst.rows >= h_src.rows);

	LOG_INFO("Copying CUDA image data from host %lux%lu to device %lux%lu\n", 
		h_src.cols, h_src.rows, d_dst.cols, d_dst.rows);

	// Calculate parameters needed by cuda
	const auto src_width = (h_src.cols * sizeof(uchar));
	const auto src_height = h_src.rows;
	const auto src_pitch = src_width;

	// Perform data copy from host
	checkCudaErrors(cudaMemcpy2DAsync(d_dst.data, d_dst.pitch, h_src.data, src_pitch,
		src_width, src_height, cudaMemcpyHostToDevice));
}

void cuda_image_copy_to_host_async(CudaHostImage& h_dst, const CudaImage& d_src)
{
	// Ensure proper images sizes
	assert(h_dst.cols >= d_src.cols);
	assert(h_dst.rows >= d_src.rows);
	
	const auto cols = d_src.cols;
	const auto rows = d_src.rows;

	LOG_INFO("Copying CUDA image data to host of size %lux%lu\n", cols, rows);

	// Calculate parameters needed by cuda
	const auto width = (cols * sizeof(uchar));
	const auto height = rows;

	// Destination's pitch is same as row width
	const auto spitch = d_src.pitch;
	const auto dpitch = width;

	// Perform data copy to host
	checkCudaErrors(cudaMemcpy2D(h_dst.data, dpitch, d_src.data, spitch,
		width, height, cudaMemcpyDeviceToHost));	
}

void cuda_image_fill_async(CudaImage& img, uchar value)
{
	const auto width = (img.cols * sizeof(uchar));
	const auto height = img.rows;

	// Fill image asynchronously with given value
	checkCudaErrors(cudaMemset2D(img.data, img.pitch, value, width, height));
}

CudaImage cuda_image_sub(CudaImage& img, size_t cols, size_t rows)
{
	assert(img.cols >= cols);
	assert(img.rows >= rows);

	return CudaImage { img.data, img.pitch, cols, rows };
}