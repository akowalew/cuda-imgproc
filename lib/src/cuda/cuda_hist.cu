///////////////////////////////////////////////////////////////////////////////
// cuda_hist.cu
//
// Contains definitions for CUDA histograms equalizer
///////////////////////////////////////////////////////////////////////////////

#include "cuda_hist.cuh"

#include <cassert>
#include <cstdio>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include "log.hpp"

//
// Private definitions
//

//! Number of threads in block in each dimension
static constexpr auto K = 32;

static CudaHistogram g_eq_hist;

static CudaLUT g_eq_lut;

//
// Public definitions
//

void cuda_hist_init()
{
	LOG_INFO("Initializing CUDA hist module\n");

	g_eq_hist = cuda_create_histogram();
	g_eq_lut = cuda_create_lut();
}

void cuda_hist_deinit()
{
	LOG_INFO("Deinitializing CUDA hist module\n");

	cuda_free_lut(g_eq_lut);
	cuda_free_histogram(g_eq_hist);
}

CudaHistogram cuda_create_histogram()
{
	// Get buffer size for histogram
	const auto size = (HistogramSize * sizeof(CudaHistogram::Type));

	// Allocate histogram on the device
	void* data;
	checkCudaErrors(cudaMalloc(&data, size));

	// Return created histogram
	return CudaHistogram { (CudaHistogram::Type*)data };
}

void cuda_free_histogram(CudaHistogram& hist)
{
	// Release device histogram
	checkCudaErrors(cudaFree(hist.data));
}

void cuda_histogram_zero_async(CudaHistogram& hist)
{
	// Get buffer size for histogram
	const auto size = (HistogramSize * sizeof(CudaHistogram::Type));

	// Fill asynchronously histogram with 0
	checkCudaErrors(cudaMemset(hist.data, 0, size));
}

__device__
uint cuda_calculate_cdf_n(uint* cdf, const uint* hist, int n)
{
	uint cdf_min = 0;
    uint acc = 0;
    for(auto i = 0; i <= n; ++i)
    {
    	// Accumulate next histogram element and store next CDF value
        acc += hist[i];
        cdf[i] = acc;
        
    	if(cdf_min == 0)
    	{
    		// Assign current accumulator value. Maybe it will be not zero
    		//  and we will gain a first CDF non-zero value (minimal) 
    		cdf_min = acc;
    	}
    }

    return cdf_min;
}

__global__
void cuda_gen_equalize_lut(uchar* lut, const uint* hist, size_t nelems)
{
	// Generating of equalizing LUT consists of three steps:
	//  1) Calculating CDF (continuous distribution function) of histogram
	//  2) Finding first, non-zero value of that CDF
	//  3) Calculating final LUT
	// In order to optimize the runtime, all of the code is placed in one function

    // We will need some temporary buffer for CDF values and for minimal one
	__shared__ uint s_cdf[256];
    __shared__ uint s_cdf_min;

	if(threadIdx.x == 0)
	{
		// Calculate CDF values and find minimal one
		s_cdf_min = cuda_calculate_cdf_n(s_cdf, hist, threadIdx.x);
	}

	__syncthreads();

    // Calculate LUT value and store it
    lut[threadIdx.x] = 
    	(((s_cdf[threadIdx.x] - s_cdf_min) * 255) 
    		/ (nelems - s_cdf_min));
}

void cuda_gen_equalize_lut_async(CudaLUT& lut, const CudaHistogram& hist, size_t nelems)
{
	LOG_INFO("Generating equalizing LUT with CUDA\n");

	// Use only one, linear, const sized block
	const auto dim_block = dim3(LUTSize);
	const auto dim_grid = 1;

	// Launch generation of equalizing LUT
	cuda_gen_equalize_lut<<<dim_grid, dim_block>>>(lut.data, hist.data, nelems);

	// Check if launch succeeded
	checkCudaErrors(cudaGetLastError());
}

void cuda_gen_equalize_lut(CudaLUT& lut, const CudaHistogram& hist, size_t nelems)
{
	// Launch generation of equalize LUT
	cuda_gen_equalize_lut_async(lut, hist, nelems);

	// Wait for device finish
	checkCudaErrors(cudaDeviceSynchronize());
}

CudaLUT cuda_gen_equalize_lut(const CudaHistogram& hist, size_t nelems)
{
	// Allocate lut on the device
	auto lut = cuda_create_lut();

	// Perform generation of equalize LUT 
	cuda_gen_equalize_lut(lut, hist, nelems);

	// Return generated LUT
	return lut;
}

__global__
void cuda_calculate_hist(
	uint* hist,
	const uchar* img, size_t pitch,
	size_t cols, size_t rows)
{
	// Allocate shared memory buffer for block-wise partial histograms
	__shared__ uint s_hist[HistogramSize];

	// Initialize local histogram with zeros
	const auto tid = (threadIdx.y*blockDim.x + threadIdx.x);
	if(tid < HistogramSize)
	{
		s_hist[tid] = 0;
	}

	// Wait for threads to finish
	__syncthreads();

	// Get position of that thread in terms of image
	const auto y = (blockIdx.y*blockDim.y + threadIdx.y);
	const auto x = (blockIdx.x*blockDim.x + threadIdx.x);
	if(y >= rows || x >= cols)
	{
		// We are out of bounds, do nothing
		return;
	}

	// Increment local counter of that thread's pixel value atomically
	const auto val = img[y*pitch + x];
	atomicAdd(&s_hist[val], 1);

	// Wait for all threads to finish
	__syncthreads();

	// Add local histogram to the global one atomically
	if(tid < HistogramSize)
	{
		atomicAdd(&hist[tid], s_hist[tid]);
	}
}

void cuda_calculate_hist_async(CudaHistogram& hist, const CudaImage& img)
{
	// Retrieve device image shape
	const auto cols = img.cols;
	const auto rows = img.rows;

	LOG_INFO("Calculating histogram with CUDA of image %lux%lu with pitch %lu\n", cols, rows, img.pitch);

	// Use const sized blocks
	const auto dim_block = dim3(K, K);

	// Use as much blocks in the grid as needed
	const auto dim_grid_x = ((cols+K-1) / K);
	const auto dim_grid_y = ((rows+K-1) / K);
	const auto dim_grid = dim3(dim_grid_x, dim_grid_y);

	LOG_DEBUG("cuda_calculate_hist_async: dim_grid = (%lu,%lu)\n", dim_grid_x, dim_grid_y);

	// Launch histogram calculation
	cuda_calculate_hist<<<dim_grid, dim_block>>>(
		hist.data,
		(uchar*)img.data, img.pitch,
		cols, rows);

	// Check if launch succeeded
	checkCudaErrors(cudaGetLastError());
}

void cuda_calculate_hist(CudaHistogram& hist, const CudaImage& img)
{
	// Launch histogram calculation
	cuda_calculate_hist_async(hist, img);
	
	// Wait for device finish
	checkCudaErrors(cudaDeviceSynchronize());
}

CudaHistogram cuda_calculate_hist(const CudaImage& img)
{
	// First, create a new histogram
	auto hist = cuda_create_histogram();

	// Initialize created histogram to zero and calculate it from image
	cuda_histogram_zero_async(hist);
	cuda_calculate_hist(hist, img);

	// Return calculated histogram
	return hist;
}

void cuda_equalize_hist_async(CudaImage& dst, const CudaImage& src)
{
	// Ensure proper images sizes
	assert(src.cols == dst.cols);
	assert(src.rows == dst.rows);

	const auto cols = src.cols;
	const auto rows = src.rows;
	const auto nelems = (cols * rows);

	// Launch histogram equalization sequence
	cuda_histogram_zero_async(g_eq_hist);
	cuda_calculate_hist_async(g_eq_hist, src);
	cuda_gen_equalize_lut_async(g_eq_lut, g_eq_hist, nelems);
	cuda_apply_lut_async(dst, src, g_eq_lut);
}

void cuda_equalize_hist(CudaImage& dst, const CudaImage& src)
{
	// Launch histogram equalization
	cuda_equalize_hist_async(dst, src);

	// Wait for device to finish
	checkCudaErrors(cudaDeviceSynchronize());
}

CudaImage cuda_equalize_hist(const CudaImage& src)
{
	LOG_INFO("Equalizing histogram with CUDA\n");

	// Allocate image on device
	auto dst = cuda_create_image(src.cols, src.rows);

	// Perform histogram equalization
	cuda_equalize_hist(dst, src);

	// Return image with equalized histogram
	return dst;
}
