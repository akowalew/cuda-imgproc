///////////////////////////////////////////////////////////////////////////////
// hist_cuda1.cu
//
// Contains definitions of functions working on images histograms
// CUDA1 version
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 20:33 CEST
///////////////////////////////////////////////////////////////////////////////

#include "hist_cuda1.hpp"

#include <stdio.h>

#include <cassert>

#include <helper_cuda.h>

CudaHistogramI8 create_cuda_histogram()
{
	int* d_data;
	checkCudaErrors(cudaMalloc(&d_data, CudaHistogramI8::size()));

	return CudaHistogramI8{d_data};
}

CudaHistogramI8 create_host_histogram()
{
	int* h_data = (int*) malloc(CudaHistogramI8::size());
	assert(h_data != nullptr);

	return CudaHistogramI8{h_data};
}

void free_cuda_histogram(const CudaHistogramI8& histogram)
{
	checkCudaErrors(cudaFree(histogram.data));
}

void free_host_histogram(const CudaHistogramI8& histogram)
{
	free(histogram.data);
}

void copy_cuda_histogram_to_host(const CudaHistogramI8& d_histogram, 
	const CudaHistogramI8& h_histogram)
{
	checkCudaErrors(cudaMemcpy(h_histogram.data, d_histogram.data, 
		CudaHistogramI8::size(), cudaMemcpyDeviceToHost));
}

__global__ 
void kernel_calculate_hist_8(CudaImage src, CudaHistogramI8 histogram)
{
	// Underlying data type of each pixel
	using DataType = unsigned char;

	// What value we will be looking for
	const auto target_value = static_cast<DataType>(threadIdx.x);

	// Initialize counter to count matches to target value
	auto counter = CudaHistogramI8::Counter{0};

	// Iterate over whole image and count pixels equal to target value
	for(size_t y = 0; y < src.height; ++y)
	{
		for(size_t x = 0; x < src.width; ++x)
		{
			// Calculate offset in source buffer for that pixel
			const auto src_offset = (y*src.pitch + x*sizeof(DataType));

			// Obtain pointer to pixel's value 
			const auto value_ptr = static_cast<DataType*>(static_cast<unsigned char*>(src.data) + src_offset);

			// Retrieve value of that pixel
			const auto value = *value_ptr;

			printf("%d ",  value);

			// Check, if retrieved value is equal to needed one
			if(value == target_value)
			{
				// We've got next value 
				++counter;
			}
		}
	}

	// Store value of target value counter
	histogram.data[threadIdx.x] = counter;
}

void calculate_hist_8_cuda1(const CudaImage& image, CudaHistogramI8& histogram)
{
	// We will use one CUDA grid with L threads (L is length of histogram)
	const auto dim_grid = 1;
	const auto dim_block = histogram.length();
	kernel_calculate_hist_8<<<dim_grid, dim_block>>>(image, histogram);

	checkCudaErrors(cudaGetLastError());
}

void equalize_hist_8_cuda1(const CudaImage& src, CudaImage& dst)
{

}	