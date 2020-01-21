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

// __global__
// void calculate_hist(
// 	const void* image, size_t pitch, size_t width, size_t height,
// 	int* histogram)
// {
// 	// In this algorithm, each kernel thread will iterate over whole
// 	// source image and count occurences of one, specific value of the histogram.
// 	// Due to this, this kernel must be called with number of threads
// 	// equal to length of the histogram.

// 	// Underlying data type of each pixel
// 	using DataType = unsigned char;

// 	// What value we will be looking for
// 	const auto target_value = static_cast<DataType>(threadIdx.x);

// 	// Initialize counter to count matches to target value
// 	auto counter = static_cast<CudaHistogram::Counter>(0);

// 	// Iterate over whole image and count pixels equal to target value
// 	for(size_t y = 0; y < height; ++y)
// 	{
// 		for(size_t x = 0; x < width; ++x)
// 		{
// 			// Calculate offset in source buffer for that pixel
// 			const auto offset = (y*pitch + x*sizeof(DataType));

// 			// Obtain pointer to pixel's value
// 			const auto value_raw_ptr = (static_cast<const unsigned char*>(image) + offset);
// 			const auto value_ptr = static_cast<const DataType*>(value_raw_ptr);
// 			const auto value = *value_ptr;

// 			// Check, if retrieved value is equal to needed one
// 			if(value == target_value)
// 			{
// 				// We've got next value
// 				++counter;
// 			}
// 		}
// 	}

// 	// Store value of target value counter
// 	histogram[threadIdx.x] = counter;
// }

// __host__
// void calculate_hist(
// 	const CudaImage& image,
// 	CudaHistogram& histogram)
// {
// 	// We will use one CUDA grid with L threads (L is length of histogram)
// 	const auto dim_grid = 1;
// 	const auto dim_block = histogram.length();
// 	calculate_hist<<<dim_grid, dim_block>>>(image.data, image.pitch, image.width, image.height, histogram.data);

// 	checkCudaErrors(cudaGetLastError());
// }

// __global__
// void calculate_cdf(
// 	const int* histogram,
// 	int* cdf,
// 	int* cdf_min)
// {
// 	// We will use a shared memory buffer for two things:
// 	// 1) To cache histogram read from global memory
// 	// 2) To write back calculated CDF to global memory
// 	__shared__ int s_histogram_data[1 << 8];

// 	// Get the index of counter in histogram buffer
// 	const auto pos = threadIdx.x;

// 	// Length of the histogram
// 	constexpr auto length = 256;

// 	// Cache histogram counter from global to shared memory
// 	s_histogram_data[pos] = histogram[pos];

// 	// Wait until all threads finish caching
// 	__syncthreads();

// 	// Now, only first thread will perform calculation of the CDF
// 	if(pos == 0)
// 	{
// 		CudaHistogram::Counter counter = 0;
// 		CudaHistogram::Counter l_min_value = 0;

// 		// Calculate cumulative distribution function
// 		for(auto i = 0; i < length; ++i)
// 		{
// 			// Get i-th value counter from shared memory
// 			const int hist_value = s_histogram_data[i];

// 			// Increase cumulative counter
// 			counter += hist_value;

// 			// If current counter value is not zero and this is the first time
// 			// We've got minimal CDF value. Store it locally
// 			if(counter != 0 && l_min_value == 0)
// 			{
// 				l_min_value = counter;
// 			}

// 			// Store current value of the counter to the shared memory
// 			s_histogram_data[i] = counter;
// 		}

// 		// Push minimal CDF value to the caller
// 		*cdf_min = l_min_value;
// 	}

// 	// Wait, till the first thread finish calculating of the CDF
// 	__syncthreads();

// 	// Store cached CDF values from shared to global memory
// 	cdf[pos] = s_histogram_data[pos];
// }

// __host__
// void calculate_cdf(
// 	const CudaHistogram& histogram,
// 	CudaCDF& cdf,
// 	CudaCDF::Counter* cdf_min)
// {
// 	// We will use one CUDA grid with L threads (L is length of the histogram)
// 	const auto dim_grid = 1;
// 	const auto dim_block = histogram.length();
// 	calculate_cdf<<<dim_grid, dim_block>>>(histogram.data, cdf.data, cdf_min);

// 	checkCudaErrors(cudaGetLastError());
// }

// __global__
// void generate_lut(
// 	int elems,
// 	const int* cdf,
// 	const int* cdf_min,
// 	unsigned char* lut)
// {
// 	// Position in buffers, according to thread index
// 	const auto pos = threadIdx.x;

// 	// Maximum value of an 8-bit number
// 	constexpr int MaxValue = 255;

// 	// Get value of cumulative distribution function and distance to minimal one
// 	const auto cdf_value = cdf[pos];
// 	const auto cdf_diff = (cdf_value - *cdf_min);

// 	// Generate LUT value
// 	const auto num = (cdf_diff * MaxValue);
// 	const auto den = (elems - *cdf_min);
// 	const auto lut_value = (num / den);
// 	lut[pos] = lut_value;
// }

// __host__
// void generate_lut(
// 	int elems,
// 	const CudaCDF& cdf,
// 	CudaCDF::Counter* cdf_min,
// 	CudaLUT& lut)
// {
// 	// We will use one CUDA grid with L threads (L is length of the histogram)
// 	const auto dim_grid = 1;
// 	const auto dim_block = cdf.length();
// 	generate_lut<<<dim_grid, dim_block>>>(elems, cdf.data, cdf_min, lut.data);

// 	checkCudaErrors(cudaGetLastError());
// }

// __global__
// void apply_lut(
// 	const void* src, size_t spitch, size_t width, size_t height,
// 	const unsigned char* lut,
// 	void* dst, size_t dpitch)
// {
// 	// Get position of thread in image buffer
// 	const auto y = blockIdx.y * blockDim.y + threadIdx.y;
// 	const auto x = blockIdx.x * blockDim.x + threadIdx.x;
// 	const auto src_pos = (y*spitch + x);
// 	const auto dst_pos = (y*dpitch + x);

// 	// Retrieve image source value
// 	const auto src_value = ((unsigned char*)src)[src_pos];

// 	// Apply LUT on source value
// 	const auto lut_value = lut[src_value];

// 	// Store destination value (after LUT)
// 	((unsigned char*)dst)[dst_pos] = lut_value;
// }

// __host__
// void apply_lut(
// 	const CudaImage& src,
// 	const CudaLUT& lut,
// 	CudaImage& dst)
// {
// 	assert(src.width == dst.width);
// 	assert(src.height == dst.height);
// 	const auto width = src.width;
// 	const auto height = src.height;

// 	// We will use one CUDA grid with W x H threads
// 	const auto dim_grid = dim3(width / 16, height / 16);
// 	const auto dim_block = dim3(16, 16);
// 	apply_lut<<<dim_grid, dim_block>>>(src.data, src.pitch, width, height,
// 		lut.data, dst.data, dst.pitch);

// 	checkCudaErrors(cudaGetLastError());
// }

// __host__
// void equalize_hist(
// 	CudaImage& src,
// 	CudaHistogram& hist,
// 	CudaCDF& cdf,
// 	CudaCDF::Counter* cdf_min,
// 	CudaLUT& lut,
// 	CudaImage& dst)
// {
// 	// Size of the images must be equal
// 	assert(src.width == dst.width);
// 	assert(src.height == dst.height);

// 	// Get number of elements in each image
// 	const auto elems = src.elems();

// 	// The most naive implementation, also the slowest.
// 	// Apply each operation separately
// 	calculate_hist(src, hist);
// 	calculate_cdf(hist, cdf, cdf_min);
// 	generate_lut(elems, cdf, cdf_min, lut);
// 	apply_lut(src, lut, dst);
// }

// void equalize_hist(const Image& src, Image& dst)
// {
// 	// Both images must have same size
// 	assert(src.cols == dst.cols);
// 	assert(src.rows == dst.rows);
// 	const auto width = src.cols;
// 	const auto height = src.rows;

// 	// Pitch of CPU images must be zero
// 	assert(src.isContinuous() && dst.isContinuous());
// 	const auto src_pitch = 0;
// 	const auto dst_pitch = 0;

// 	// Create CUDA device variables
// 	auto image = CudaImage(width, height);
// 	auto histogram = CudaHistogram();
// 	auto lut = CudaLUT();
// 	CudaCDF::Counter* cdf_min;
// 	checkCudaErrors(cudaMalloc(&cdf_min, sizeof(CudaCDF::Counter)));

// 	// Copy source image into CUDA device
// 	checkCudaErrors(cudaMemcpy2D(image.data, image.pitch,
// 		src.data, src_pitch, width, height, cudaMemcpyHostToDevice));

// 	// Do histogram equalization
// 	equalize_hist(image, histogram, histogram, cdf_min, lut, image);
// 	checkCudaErrors(cudaDeviceSynchronize());

// 	checkCudaErrors(cudaMemcpy2D(dst.data, dst_pitch,
// 		image.data, image.pitch, image.width, image.height, cudaMemcpyDeviceToHost));

// 	// Release CUDA device variables
// 	checkCudaErrors(cudaFree(cdf_min));
// }

//! Number of threads in block in each dimension
constexpr auto K = 32;

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
    const auto y = (threadIdx.y + blockDim.y*blockIdx.y);
    const auto x = (threadIdx.x + blockDim.x*blockIdx.x);
    if(y > rows || x > cols)
    {
        return;
    }

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
