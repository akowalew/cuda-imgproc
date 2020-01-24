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

//! Number of threads in block in each dimension
constexpr auto K = 32;

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

__device__
uint cuda_calc_cdf(const uint* hist, uint* cdf)
{
    uint cdf_min;
    uint acc = 0;

    // Iterate over whole histogram sn
    for(auto i = 0; i < 256; ++i)
    {
    	// Get next histogram element
        const auto hist_v = hist[i];
        if(hist_v != 0 && acc == 0)
        {
        	// If this is first non-zero CDF value, store it
            cdf_min = hist_v;
        }

        // Increase histogram accumulator and use it as next CDF value
        acc += hist_v;
        cdf[i] = acc;
    }

    return cdf_min;
}

__global__
void cuda_gen_equalize_lut(uchar* lut, const uint* hist)
{
    // We will need shared memory buffer for CDF values and its minimal one
	__shared__ uint s_cdf[256];
	__shared__ uint s_cdf_min;

	if(threadIdx.x == 0)
	{
		// CDF calculation is truly sequential, so only first thread may do it.
		// Calculate CDF for given histogram and get its first non-zero value
		s_cdf_min = cuda_calc_cdf(hist, s_cdf);
	}

	// Wait for thread #0 to complete CDF calculation
	__syncthreads();

	// Obtain value of CDF for that thread
    const auto cdf_v = s_cdf[threadIdx.x];

    // Get the difference between that CDF value and minimal one
    const auto cdf_diff = (cdf_v - s_cdf_min);

	// Obtain number of pixels in the image from the last CDF element
    const auto elems = s_cdf[255];

    // Calculate LUT value and store it
    const auto lut_v = ((cdf_diff * 255) / (elems - s_cdf_min));
    lut[threadIdx.x] = lut_v;
}

void cuda_gen_equalize_lut(CUDALUT& lut, const CudaHistogram& hist)
{
	// Use only one, linear, const sized block
	const auto dim_block = dim3(LUTSize);
	const auto dim_grid = 1;

	// Launch generation of equalizing LUT
	cuda_gen_equalize_lut<<<dim_grid, dim_block>>>(lut.data, hist.data);

	// Check if launch succeeded
	checkCudaErrors(cudaGetLastError());

	// Wait for device finish
	checkCudaErrors(cudaDeviceSynchronize());
}

CUDALUT cuda_gen_equalize_lut(const CudaHistogram& hist)
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

void cuda_calculate_hist(CudaHistogram& hist, const CudaImage& img)
{
	// Retrieve device image shape
	const auto cols = img.cols;
	const auto rows = img.rows;

	printf("*** Calculating histogram with CUDA of image %lux%lu\n", cols, rows);

	// Use const sized blocks
	const auto dim_block = dim3(K, K);

	// Use as much blocks in the grid as needed
	const auto dim_grid_x = ((cols+K-1) / K);
	const auto dim_grid_y = ((rows+K-1) / K);
	const auto dim_grid = dim3(dim_grid_x, dim_grid_y);

	// Launch histogram calculation
	cuda_calculate_hist<<<dim_grid, dim_block>>>(
		hist.data,
		(uchar*)img.data, img.pitch,
		cols, rows);

	// Check if launch succeeded
	checkCudaErrors(cudaGetLastError());
	
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

void cuda_equalize_hist(CudaImage& dst, const CudaImage& src)
{
	// Ensure proper images sizes
	assert(src.cols == dst.cols);
	assert(src.rows == dst.rows);

	auto hist = cuda_calculate_hist(src);
	auto lut = cuda_gen_equalize_lut(hist);
	cuda_apply_lut(dst, src, lut);

	cuda_free_lut(lut);
	cuda_free_histogram(hist);
}

CudaImage cuda_equalize_hist(const CudaImage& src)
{
	printf("*** Equalizing histogram with CUDA\n");

	// Allocate image on device
	auto dst = cuda_create_image(src.cols, src.rows);

	// Perform histogram equalization
	cuda_equalize_hist(dst, src);

	// Return image with equalized histogram
	return dst;
}
