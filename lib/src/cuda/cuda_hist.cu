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
	// Allocate histogram on the device
	void* data;
	checkCudaErrors(cudaMalloc(&data, CudaHistogram::BufferSize));

	// Return created histogram
	return CudaHistogram { (CudaHistogram::Type*)data };
}

void cuda_free_histogram(CudaHistogram& hist)
{
	// Release device histogram
	checkCudaErrors(cudaFree(hist.data));
}

void cuda_histogram_fill_async(CudaHistogram& hist, CudaHistogram::Type value)
{
	// Fill asynchronously histogram with value
	checkCudaErrors(cudaMemset(hist.data, value, CudaHistogram::BufferSize));
}

void cuda_histogram_copy_data_to_host(CudaHistogram::Type* h_data, CudaHistogram& d_hist)
{
	checkCudaErrors(cudaMemcpy(h_data, d_hist.data, CudaHistogram::BufferSize, cudaMemcpyDeviceToHost));
}

__device__
uint cuda_calculate_cdf_in_place(uint* buf)
{
	// We need some variable to remember minimal CDF value 
	uint cdf_min = 0;

	// We've got first CDF value, so we're starting directly from the second
    for(auto i = 1; i < CudaHistogram::Size; ++i)
    {
    	// Calculate next CDF value
        buf[i] += buf[i-1];
        
    	if(cdf_min == 0)
    	{
    		// Assign current CDF value. Maybe it will be not zero
    		//  and we will gain a first CDF non-zero value (minimal) 
    		cdf_min = buf[i];
    	}
    }

    return cdf_min;
}

__global__
void cuda_gen_equalize_lut(uchar* lut, const uint* hist)
{
	// Generating of equalizing LUT consists of three steps:
	//  1) Calculating CDF (continuous distribution function) of histogram
	//  2) Finding first, non-zero value of that CDF
	//  3) Calculating final LUT
	// In order to optimize the runtime, all of the code is placed in one function

    // We will need some temporary buffer for CDF values and CDF min
    // We will use same buffer both for CDF and for histogram caching
	__shared__ uint s_buf[CudaHistogram::Size];
    __shared__ uint s_cdf_min;

    // Cache histogram values into shared memory
    s_buf[threadIdx.x] = hist[threadIdx.x];

    // Wait for all threads to finish caching
    __syncthreads();

	// Calculate CDF of histogram and CDF_min (in place)
	// This will be done only by first thread, because this is truly sequential
	if(threadIdx.x == 0)
	{
	    s_cdf_min = cuda_calculate_cdf_in_place(s_buf);
	}

	// Wait for first thread to finish CDF calculation
	__syncthreads();

    // Calculate LUT value and store it
    // where: 
    //  - (CudaHistogram::Size-1) is both maximum value of image and index of last CDF value
    //  - s_buf[CudaHistogram::Size-1] is last CDF value -> number of elements in the image
    lut[threadIdx.x] = 
    	(((s_buf[threadIdx.x] - s_cdf_min) * (CudaHistogram::Size-1)) 
    		/ (s_buf[CudaHistogram::Size-1] - s_cdf_min));
}

void cuda_gen_equalize_lut_async(CudaLUT& lut, const CudaHistogram& hist)
{
	LOG_INFO("Generating equalizing LUT with CUDA\n");

	static_assert(CudaLUT::Size == CudaHistogram::Size,
		"Sizes of LUT and Histograms should be the same");

	// Use only one, linear, const sized block
	const auto dim_block = dim3(CudaLUT::Size);
	const auto dim_grid = dim3(1, 1);

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
void cuda_calculate_hist_kernel(
	uint* hist,
	const uchar* img, size_t pitch,
	size_t cols, size_t rows)
{
	// Allocate shared memory buffer for block-wise partial histograms
	__shared__ uint s_hist[CudaHistogram::Size];

	// Initialize local histogram with zeros
	const auto tid = (threadIdx.y*blockDim.x + threadIdx.x);
	if(tid < CudaHistogram::Size)
	{
		s_hist[tid] = 0;
	}

	// Wait for threads to finish
	__syncthreads();

	// Get position of that thread in terms of image
	const auto y = (blockIdx.y*blockDim.y + threadIdx.y);
	const auto x = (blockIdx.x*blockDim.x + threadIdx.x);

	// Check, if we are in image bounds
	if(y < rows && x < cols)
	{
		// Increment local counter of that thread's pixel value atomically
		const auto val = img[y*pitch + x];
		atomicAdd(&s_hist[val], 1);
	}

	// Wait for all threads to finish
	__syncthreads();

	// Add local histogram to the global one atomically
	if(tid < CudaHistogram::Size)
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
	cuda_histogram_fill_async(hist, 0);
	cuda_calculate_hist_kernel<<<dim_grid, dim_block>>>(
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
