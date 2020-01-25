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

__global__
void cuda_gen_equalize_lut(uchar* lut, const uint* hist)
{
	// Generating of equalizing LUT consists of three steps:
	//  1) Calculating CDF (continuous distribution function) of histogram
	//  2) Finding first, non-zero value of that CDF
	//  3) Calculating final LUT
	// In order to optimize the runtime, all of the code is placed in one function

    // We will need some temporary buffer for CDF values
    // Shared memory will be used, because it is too big for local memory
	__shared__ uint s_cdf[256];

	// Calculate CDF values and find minimal one
	// Note that every thread calculates whole CDF.
	// Though this process is truly sequential, there is no need to do it by
	//  one thread, because this way we will need then  __syncthreads, which costs.
    uint cdf_min;
    uint acc = 0;
    for(auto i = 0; i < 256; ++i)
    {
    	const auto hist_v = hist[i];
    	if(acc == 0 && hist_v != 0)
    	{
    		// We've got minimal CDF value
    		cdf_min = hist_v;
    	}

    	// Accumulate next histogram element and store next CDF value
        acc += hist_v;
        s_cdf[i] = acc;
    }

    // Calculate LUT value and store it
    lut[threadIdx.x] = 
    	(((s_cdf[threadIdx.x] - cdf_min) * 255) 
    		/ (s_cdf[255] - cdf_min));
}

void cuda_gen_equalize_lut_async(CudaLUT& lut, const CudaHistogram& hist)
{
	LOG_INFO("Generating equalizing LUT with CUDA\n");

	// Use only one, linear, const sized block
	const auto dim_block = dim3(LUTSize);
	const auto dim_grid = 1;

	// Launch generation of equalizing LUT
	cuda_gen_equalize_lut<<<dim_grid, dim_block>>>(lut.data, hist.data);

	// Check if launch succeeded
	checkCudaErrors(cudaGetLastError());
}

void cuda_gen_equalize_lut(CudaLUT& lut, const CudaHistogram& hist)
{
	// Launch generation of equalize LUT
	cuda_gen_equalize_lut_async(lut, hist);

	// Wait for device finish
	checkCudaErrors(cudaDeviceSynchronize());
}

CudaLUT cuda_gen_equalize_lut(const CudaHistogram& hist)
{
	// Allocate lut on the device
	auto lut = cuda_create_lut();

	// Perform generation of equalize LUT 
	cuda_gen_equalize_lut(lut, hist);

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

	// Get position of that thread in terms of image
	const auto y = (blockIdx.y*blockDim.y + threadIdx.y);
	const auto x = (blockIdx.x*blockDim.x + threadIdx.x);
	if(y > rows || x > cols)
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
	const auto tid = (threadIdx.y*blockDim.x + threadIdx.x);
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

	LOG_INFO("Calculating histogram with CUDA of image %lux%lu\n", cols, rows);

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
	// First, calculate histogram of the source image
	auto hist = cuda_create_histogram();

	// Perform histogram calculation
	cuda_calculate_hist(hist, img);

	// Return calculated histogram
	return hist;
}

void cuda_equalize_hist_async(CudaImage& dst, const CudaImage& src)
{
	// Ensure proper images sizes
	assert(src.cols == dst.cols);
	assert(src.rows == dst.rows);

	// Launch histogram equalization sequence
	cuda_calculate_hist_async(g_eq_hist, src);
	cuda_gen_equalize_lut_async(g_eq_lut, g_eq_hist);
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