///////////////////////////////////////////////////////////////////////////////
// hist_cuda1.cu
//
// Contains definitions of functions working on images histograms
// CUDA1 version
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 20:33 CEST
///////////////////////////////////////////////////////////////////////////////

#include "hist_cuda_v1.cuh"

#include <cassert>

#include <helper_cuda.h>

CudaHistogram create_cuda_histogram()
{
	int* d_data;
	checkCudaErrors(cudaMalloc(&d_data, CudaHistogram::size()));

	return CudaHistogram{d_data};
}

void free_cuda_histogram(const CudaHistogram& histogram)
{
	checkCudaErrors(cudaFree(histogram.data));
}

CudaLUT create_cuda_lut()
{
	unsigned char* d_data;
	checkCudaErrors(cudaMalloc(&d_data, CudaLUT::size()));

	return CudaLUT{d_data};
}

void free_cuda_lut(const CudaLUT& lut)
{
	checkCudaErrors(cudaFree(lut.data));
}

__global__
void calculate_hist(
	const void* image, size_t pitch, size_t width, size_t height,
	int* histogram) 
{
	// In this algorithm, each kernel thread will iterate over whole 
	// source image and count occurences of one, specific value of the histogram.
	// Due to this, this kernel must be called with number of threads
	// equal to length of the histogram.

	// Underlying data type of each pixel
	using DataType = unsigned char;

	// What value we will be looking for
	const auto target_value = static_cast<DataType>(threadIdx.x);

	// Initialize counter to count matches to target value
	auto counter = static_cast<CudaHistogram::Counter>(0);

	// Iterate over whole image and count pixels equal to target value
	for(size_t y = 0; y < height; ++y)
	{
		for(size_t x = 0; x < width; ++x)
		{
			// Calculate offset in source buffer for that pixel
			const auto offset = (y*pitch + x*sizeof(DataType));

			// Obtain pointer to pixel's value
			const auto value_raw_ptr = (static_cast<const unsigned char*>(image) + offset);
			const auto value_ptr = static_cast<const DataType*>(value_raw_ptr);
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
	histogram[threadIdx.x] = counter;
}

__host__
void calculate_hist(
	const CudaImage& image, 
	CudaHistogram& histogram)
{
	// We will use one CUDA grid with L threads (L is length of histogram)
	const auto dim_grid = 1;
	const auto dim_block = histogram.length();
	calculate_hist<<<dim_grid, dim_block>>>(image.data, image.pitch, image.width, image.height, histogram.data);

	checkCudaErrors(cudaGetLastError());
}

__global__
void calculate_cdf(
	const int* histogram, 
	int* cdf,
	int* cdf_min)
{
	// We will use a shared memory buffer for two things:
	// 1) To cache histogram read from global memory
	// 2) To write back calculated CDF to global memory
	__shared__ int s_histogram_data[1 << 8];

	// Get the index of counter in histogram buffer
	const auto pos = threadIdx.x;

	// Length of the histogram
	constexpr auto length = 256;

	// Cache histogram counter from global to shared memory
	s_histogram_data[pos] = histogram[pos];

	// Wait until all threads finish caching
	__syncthreads();

	// Now, only first thread will perform calculation of the CDF
	if(pos == 0)
	{
		CudaHistogram::Counter counter = 0;
		CudaHistogram::Counter l_min_value = 0;

		// Calculate cumulative distribution function
		for(auto i = 0; i < length; ++i)
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
		*cdf_min = l_min_value;
	}

	// Wait, till the first thread finish calculating of the CDF
	__syncthreads();

	// Store cached CDF values from shared to global memory
	cdf[pos] = s_histogram_data[pos];
}

__host__
void calculate_cdf(
	const CudaHistogram& histogram, 
	CudaCDF& cdf,
	CudaCDF::Counter* cdf_min)
{
	// We will use one CUDA grid with L threads (L is length of the histogram)
	const auto dim_grid = 1;
	const auto dim_block = histogram.length();
	calculate_cdf<<<dim_grid, dim_block>>>(histogram.data, cdf.data, cdf_min);

	checkCudaErrors(cudaGetLastError());
}

__global__
void generate_lut(
	int elems,
	const int* cdf,
	const int* cdf_min,
	unsigned char* lut)
{	
	// Position in buffers, according to thread index
	const auto pos = threadIdx.x;

	// Maximum value of an 8-bit number
	constexpr int MaxValue = 255;

	// Get value of cumulative distribution function and distance to minimal one
	const auto cdf_value = cdf[pos];
	const auto cdf_diff = (cdf_value - *cdf_min);

	// Generate LUT value
	const auto num = (cdf_diff * MaxValue);
	const auto den = (elems - *cdf_min);
	const auto lut_value = (num / den);
	lut[pos] = lut_value;
}

__host__
void generate_lut(
	int elems,
	const CudaCDF& cdf,
	CudaCDF::Counter* cdf_min,
	CudaLUT& lut)
{
	// We will use one CUDA grid with L threads (L is length of the histogram)
	const auto dim_grid = 1;
	const auto dim_block = cdf.length();
	generate_lut<<<dim_grid, dim_block>>>(elems, cdf.data, cdf_min, lut.data);

	checkCudaErrors(cudaGetLastError());
}

__global__
void apply_lut(
	const void* src, size_t spitch, size_t width, size_t height,
	const unsigned char* lut,
	void* dst, size_t dpitch)
{
	// Get position of thread in image buffer
	const auto y = blockIdx.y * blockDim.y + threadIdx.y;
	const auto x = blockIdx.x * blockDim.x + threadIdx.x;
	const auto src_pos = (y*spitch + x);
	const auto dst_pos = (y*dpitch + x);

	// Retrieve image source value
	const auto src_value = ((unsigned char*)src)[src_pos];

	// Apply LUT on source value
	const auto lut_value = lut[src_value];

	// Store destination value (after LUT)
	((unsigned char*)dst)[dst_pos] = lut_value;
}

__host__
void apply_lut(
	const CudaImage& src,
	const CudaLUT& lut,
	CudaImage& dst)
{
	assert(src.width == dst.width);
	assert(src.height == dst.height);
	const auto width = src.width;
	const auto height = src.height;

	// We will use one CUDA grid with W x H threads
	const auto dim_grid = dim3(width / 16, height / 16);
	const auto dim_block = dim3(16, 16);
	apply_lut<<<dim_grid, dim_block>>>(src.data, src.pitch, width, height, 
		lut.data, dst.data, dst.pitch);	

	checkCudaErrors(cudaGetLastError());
}      

__host__
void equalize_hist(
	CudaImage& src,
	CudaHistogram& hist,
	CudaCDF& cdf,
	CudaCDF::Counter* cdf_min,
	CudaLUT& lut,
	CudaImage& dst)
{
	// Size of the images must be equal
	assert(src.width == dst.width);
	assert(src.height == dst.height);

	// Get number of elements in each image
	const auto elems = src.elems();

	// The most naive implementation, also the slowest.
	// Apply each operation separately
	calculate_hist(src, hist);
	calculate_cdf(hist, cdf, cdf_min);
	generate_lut(elems, cdf, cdf_min, lut);
	apply_lut(src, lut, dst);
}

void equalize_hist(const Image& src, Image& dst)
{
	// Both images must have same size
	assert(src.cols == dst.cols);
	assert(src.rows == dst.rows);
	const auto width = src.cols;
	const auto height = src.rows;

	// Pitch of CPU images must be zero
	assert(src.isContinuous() && dst.isContinuous());
	const auto src_pitch = 0;
	const auto dst_pitch = 0;

	CudaCDF::Counter* cdf_min;

	// Create CUDA device variables
	auto image = create_cuda_image(width, height);
	auto histogram = create_cuda_histogram();
	auto lut = create_cuda_lut();
	checkCudaErrors(cudaMalloc(&cdf_min, sizeof(CudaCDF::Counter)));

	// Copy source image into CUDA device
	checkCudaErrors(cudaMemcpy2D(image.data, image.pitch, 
		src.data, src_pitch, width, height, cudaMemcpyHostToDevice));

	// Do histogram equalization
	equalize_hist(image, histogram, histogram, cdf_min, lut, image);

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy2D(dst.data, dst_pitch,
		image.data, image.pitch, image.width, image.height, cudaMemcpyDeviceToHost));

	// Release CUDA device variables
	checkCudaErrors(cudaFree(cdf_min));
	free_cuda_lut(lut);
	free_cuda_histogram(histogram);
	free_cuda_image(image);
}
