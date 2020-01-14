///////////////////////////////////////////////////////////////////////////////
// hist_cuda_v1.cuh
//
// Contains declarations of functions working on images histograms
// CUDA v1 implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 27.12.2019 12:24 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <limits>

#include <cuda_runtime.h>

#include "image.hpp"
#include "image_cuda.hpp"

//! Helper typedef - host histogram
using Histogram = std::array<int, 256>;

/**
 * @brief Represents histogram with given counter type and analyzed data width
 * @details
 */
struct CudaHistogram
{
	//
	// Helper typedefs
	//

	using Counter = unsigned int;
	static constexpr auto Length = 256;

	//
	// Public data
	//

	Counter* data;

	//
	// Public methods
	//

	/**
	 * @brief Constructor
	 * @details Allocates histogram on the device
	 */
	CudaHistogram();

	/**
	 * @brief Destructor
	 * @details Releases histogram on the device
	 */
	~CudaHistogram();

	/**
	 * @brief Copies contens of histogram to host
	 * @details 
	 */
	void copy_to_host(Histogram& hist);

	/**
	 * @brief Returns length of the histogram
	 * @details
	 *
	 * @return length of the histogram
	 */
	static size_t length()
	{
		return Length;
	}

	/**
	 * @brief Returns size in bytes of histogram contents
	 * @details 
	 * 
	 * @return size in bytes of histogram contents
	 */
	static size_t size()
	{
		return (sizeof(Counter) * length());
	}

	Counter* begin()
	{
		return data;
	}

	Counter* end()
	{
		return (data + length());
	}
};

using CudaCDF = CudaHistogram;

struct CudaLUT
{
	//
	// Helper typedefs
	//

	using Type = unsigned char;

	//
	// Public data
	//

	Type* data;

	//
	// Public methods
	//

	/**
	 * @brief Creates lookup table on the device
	 * @details
	 */
	CudaLUT();

	/**
	 * @brief Releases lookup table on the device
	 * @details
	 */
	~CudaLUT();

	static size_t length()
	{
		return (1 + (size_t)std::numeric_limits<Type>::max());
	}

	static size_t size()
	{
		return (sizeof(Type) * length());
	}

	Type* begin()
	{
		return data;
	}

	Type* end()
	{
		return (data + length());
	}
};

/**
 * @brief Initializes hist module
 * @details
 */
void hist_init();

/**
 * @brief Deinitializes hist module
 * @details
 */
void hist_deinit();

// __global__
// void calculate_hist(
// 	const void* image, size_t pitch, size_t width, size_t height,
// 	int* histogram);

// __host__
// void calculate_hist(
// 	const CudaImage& image,
// 	CudaHistogram& histogram);

// __global__
// void calculate_cdf(
// 	const int* histogram,
// 	int* cdf,
// 	int* cdf_min);

// __host__
// void calculate_cdf(
// 	const CudaHistogram& histogram,
// 	CudaCDF& cdf,
// 	CudaCDF::Counter* cdf_min);

// __global__
// void generate_lut(
// 	int elems,
// 	const int* cdf,
// 	const int* cdf_min,
// 	unsigned char* lut);

// __host__
// void generate_lut(
// 	int elems,
// 	const CudaCDF& cdf,
// 	CudaCDF::Counter* cdf_min,
// 	CudaLUT& lut);

// __global__
// void apply_lut(
// 	const void* src, size_t spitch, size_t width, size_t height,
// 	const unsigned char* lut,
// 	void* dst, size_t dpitch);

// __host__
// void apply_lut(
// 	const CudaImage& src,
// 	const CudaLUT& lut,
// 	CudaImage& dst);

// __host__
// void equalize_hist(
// 	CudaImage& src,
// 	CudaHistogram& hist,
// 	CudaCDF& cdf,
// 	CudaCDF::Counter* cdf_min,
// 	CudaLUT& lut,
// 	CudaImage& dst);

__global__
void calculate_hist(
	const uchar* img, size_t pitch,
	size_t width, size_t height,
	unsigned int* hist);

/**
 * @brief Performs calculation of image histogram
 * @details 
 * 
 * @param img source device image
 * @param hist device histogram to be calculated
 */
__host__
void calculate_hist(const CudaImage& img, CudaHistogram& hist); 

/**
 * @brief Performs calculation of image histogram
 * @details 
 * 
 * @param src source host image
 * @param hist host histogram to be calculated
 */
void calculate_hist(const Image& img, Histogram& hist);

/**
 * @brief Performs histogram equalization of source image
 * @details 
 * 
 * @param src source device image
 * @param dst destination device image
 */
__host__
void equalize_hist(const CudaImage& src, CudaImage& dst);

/**
 * @brief Performs histogram equalization of source image
 * @details
 *
 * @param src source image
 * @param dst destination image
 */
void equalize_hist(const Image& src, Image& dst);


// cv calculate_hist()
// calculate_hist(cv, cv)

// ref::calculate_hist(cv, cv)

// omp::calculate_hist(cv, cv)

// cuda::calculate_hist(cv, cv)
// cuda::calculate_hist(cu, cu)
// __global__ cuda::calculate_hist(...)
