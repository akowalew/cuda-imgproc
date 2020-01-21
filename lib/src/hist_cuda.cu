///////////////////////////////////////////////////////////////////////////////
// hist_cuda1.cu
//
// Contains definitions of functions working on images histograms
// CUDA1 version
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 20:33 CEST
///////////////////////////////////////////////////////////////////////////////

#include "hist_cuda.cuh"

#include <cassert>

#include <helper_cuda.h>

//! Number of threads in block in each dimension
constexpr auto K = 32;

CudaHistogram::CudaHistogram()
{
	// Allocate histogram on the device
	checkCudaErrors(cudaMalloc(&data, size()));
}

CudaHistogram::~CudaHistogram()
{
	if(data)
	{
		// Release histogram on the device
		checkCudaErrors(cudaFree(data));
	}
}

void CudaHistogram::copy_to_host(Histogram& hist)
{
	checkCudaErrors(cudaMemcpy(hist.data(), data, size(), cudaMemcpyDeviceToHost));
}

CudaLUT::CudaLUT()
{
	// Allocate lut on the device
	checkCudaErrors(cudaMalloc(&data, CudaLUT::size()));
}

CudaLUT::~CudaLUT()
{
	if(data)
	{
		// Release lut on the device
		checkCudaErrors(cudaFree(data));
	}
}

__device__
unsigned int calc_cdf(const unsigned int* hist, unsigned int* cdf)
{
    unsigned int cdf_min;
    unsigned int acc = 0;

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
void gen_lut(const unsigned int* hist, unsigned char* lut)
{
    // We will need shared memory buffer for CDF values and its minimal one
	__shared__ unsigned int s_cdf[256];
	__shared__ unsigned int s_cdf_min;

	if(threadIdx.x == 0)
	{
		// CDF calculation is truly sequential, so only first thread may do it.
		// Calculate CDF for given histogram and get its first non-zero value
		s_cdf_min = calc_cdf(hist, s_cdf);
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

__global__
void apply_lut(
    const unsigned char* lut,
    const unsigned char* src, size_t spitch,
    size_t cols, size_t rows,
    unsigned char* dst, size_t dpitch)
{
    // Get position of that thread in terms of the image
    const auto y = (threadIdx.y + blockDim.y*blockIdx.y);
    const auto x = (threadIdx.x + blockDim.x*blockIdx.x);

    // If we are out of bounds, do nothing
    if(y > rows || x > cols)
    {
        return;
    }

    // Apply LUT on src image pixel and store result into dst image pixel
    const auto src_v = src[x + y*spitch];
    const auto lut_v = lut[src_v];
    dst[x + y*dpitch] = lut_v;
}

__global__
void calculate_hist(
	const uchar* img, size_t pitch,
	size_t width, size_t height,
	unsigned int* hist)
{
	// Allocate shared memory buffer for block-wise partial histograms
	__shared__ unsigned int s_hist[256];

	// Get position of that thread in terms of image
	const auto i = (blockIdx.y*blockDim.y + threadIdx.y);
	const auto j = (blockIdx.x*blockDim.x + threadIdx.x);
	if(i > width || j > height)
	{
		// We are out of bounds, do nothing
		return;
	}

	// Increment local counter of that thread's pixel value atomically
	const auto val = img[i*pitch + j];
	const auto counter = &s_hist[val];
	atomicAdd(counter, 1);

	// Wait for all threads to finish
	__syncthreads();

	// Add local histogram to the global one atomically
	const auto tid = (threadIdx.y*blockDim.x + threadIdx.x);
	atomicAdd(&hist[tid], s_hist[tid]);
}

__host__
void calculate_hist(const CudaImage& img, CudaHistogram& hist)
{
	// Launch histogram calculation on the device
	const auto dim_grid_x = ((img.width+K-1) / K);
	const auto dim_grid_y = ((img.height+K-1) / K);
	const auto dim_grid = dim3(dim_grid_x, dim_grid_y);
	const auto dim_block = dim3(K, K);
	calculate_hist<<<dim_grid, dim_block>>>(
		(uchar*)img.data, img.pitch,
		img.width, img.height,
		hist.data);

	// Check if launch succeeded
	checkCudaErrors(cudaGetLastError());
}

void calculate_hist(const Image& img, Histogram& hist)
{
	// Create memories on the device
	auto d_img = CudaImage(img.cols, img.rows);
	auto d_hist = CudaHistogram();

	// Invoke calculation on device
	d_img.copy_from_host(img);
	calculate_hist(d_img, d_hist);
	d_hist.copy_to_host(hist);
}

__host__
void equalize_hist(const CudaImage& src, CudaImage& dst)
{
	// Ensure proper images sizes
	assert(src.width == dst.width);
	assert(src.height == dst.height);
	const auto width = src.width;
	const auto height = src.height;

	// First, calculate histogram of the source image
	auto hist = CudaHistogram();
	calculate_hist(src, hist);

	// Check if launch succeeded
	checkCudaErrors(cudaGetLastError());
}

void equalize_hist(const Image& src, Image& dst)
{
	assert(src.cols == dst.cols);
	assert(src.rows == dst.rows);
	const auto width = src.cols;
	const auto height = src.rows;

	// Create CUDA image and fill it with host copy
	auto image = CudaImage(width, height);
	image.copy_from_host(src);

	// Do histogram equalization (in place)
	equalize_hist(image, image);

	// Get back results to host
	image.copy_to_host(dst);
}
