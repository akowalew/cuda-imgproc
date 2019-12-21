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
	// Underlying data type of each pixelcomp
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

__global__
void calculate_cdf(const CudaHistogramI8& histogram, CudaHistogramI8::Counter& min_value)
{
	// We will use a shared memory buffer for two things:
	// 1) To cache histogram read from global memory
	// 2) To write back calculated CDF to global memory
	__shared__ CudaHistogramI8::Counter s_histogram_data[CudaHistogramI8::length()];

	// Get the index of counter in histogram buffer
	const auto pos = threadIdx.x;

	// Cache histogram counter from global to shared memory
	s_histogram_data[pos] = histogram.data[pos];

	// Wait until all threads finish caching
	__syncthreads();

	// Now, only first thread will perform calculation of the CDF
	if(pos == 0)
	{
		CudaHistogramI8::Counter counter = 0;
		CudaHistogramI8::Counter l_min_value = 0;

		// Calculate cumulative distribution function
		for(auto i = 0; i < histogram.length(); ++i)
		{
			// Get i-th value counter from shared memory
			const int hist_value = s_histogram_data[i];

			// Increase cumulative counter
			counter += hist_value;

			// If current counter value is not zero and this is the first time
			// We've got minimal CDF value. Store it locally
			if(counter != 0 && l_min_value == 0)
			{
				l_min_value = counter;
			}

			// Store current value of the counter to the shared memory
			s_histogram_data[i] = counter;
		}

		// Push minimal CDF value to the caller
		min_value = l_min_value;
	}


	// Wait, till the first thread finish calculating of the CDF
	__syncthreads();

	// Store cached CDF values from shared to global memory
	histogram.data[pos] = s_histogram_data[pos];
}

__global__
void generate_lut(
	int elems,
	const CudaCDFI8& cdf,
	CudaCDFI8::Counter cdf_min,
	CudaLUTU8::pointer lut_data
	)
{
	// Position in buffers, according to thread index
	const auto pos = threadIdx.x;

	// Maximum value of an 8-bit number
	constexpr int MaxValue = 255;

	// Get value of cumulative distribution function and distance to minimal one
	const auto cdf_value = cdf.data[pos];
	const auto cdf_diff = (cdf_value - cdf_min);

	// Generate LUT value
	const auto num = (cdf_diff * MaxValue);
	const auto den = (elems - cdf_min);
	const auto lut_value = (num / den);

	lut_data[pos] = lut_value;
}

__global__
void apply_lut(
	CudaLUTU8::const_pointer lut_data,
	CudaImage image)
{
	// Get position of thread in image buffer
	const auto y = threadIdx.y;
	const auto x = threadIdx.x;
	const auto pos = (y * image.pitch + x);

	// Retrieve image source value
	const auto src_value = ((unsigned char*)image.data)[pos];

	// Apply LUT on source value
	const auto lut_value = lut_data[src_value];

	// Store destination value (after LUT)
	((unsigned char*)image.data)[pos] = lut_value;
}

void calculate_hist_8_cuda1(const CudaImage& image, CudaHistogramI8& histogram)
{
	// We will use one CUDA grid with L threads (L is length of histogram)
	const auto dim_grid = 1;
	const auto dim_block = histogram.length();
	kernel_calculate_hist_8<<<dim_grid, dim_block>>>(image, histogram);

	checkCudaErrors(cudaGetLastError());
}

void calculate_cdf_8_cuda1(const CudaHistogramI8& histogram, CudaHistogramI8::Counter& min_value)
{
	// We will use one CUDA grid with L threads (L is length of the histogram)
	const auto dim_grid = 1;
	const auto dim_block = histogram.length();
	calculate_cdf<<<dim_grid, dim_block>>>(histogram, min_value);

	checkCudaErrors(cudaGetLastError());
}

void generate_lut_8_cuda1(
	int elems,
	const CudaCDFI8& cdf,
	CudaCDFI8::Counter cdf_min,
	CudaLUTU8& lut)
{
	// We will use one CUDA grid with L threads (L is length of the histogram)
	const auto dim_grid = 1;
	const auto dim_block = cdf.length();
	generate_lut<<<dim_grid, dim_block>>>(elems, cdf, cdf_min, lut.data());

	checkCudaErrors(cudaGetLastError());
}

void apply_lut_8_cuda1(
	const CudaLUTU8& lut,
	CudaImage& image)
{
	// We will use one CUDA grid with W x H threads (W, H are width and height of the image)
	const auto dim_grid = 1;
	const auto dim_block = dim3(image.height, image.width);
	apply_lut<<<dim_grid, dim_block>>>(lut.data(), image);

	checkCudaErrors(cudaGetLastError());
}

void equalize_hist_8_cuda1(
	CudaImage& image,
	CudaHistogramI8& hist,
	CudaHistogramI8::Counter& cdf_min,
	CudaLUTU8& lut)
{
	// The most naive implementation, also the slowest:
	// Apply each operation separately

	calculate_hist_8_cuda1(image, hist);
	calculate_cdf_8_cuda1(hist, cdf_min);
	generate_lut_8_cuda1(image.width*image.height, hist, cdf_min, lut);
	apply_lut_8_cuda1(lut, image);
}
