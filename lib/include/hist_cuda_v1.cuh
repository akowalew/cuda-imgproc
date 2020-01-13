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

/**
 * @brief Represents histogram with given counter type and analyzed data width
 * @details
 *
 * @tparam TCounter Type of the values' counter
 * @tparam DataWidth Width of analyzed data. Determines length of histogram
 */
struct CudaHistogram
{
	//
	// Helper typedefs
	//

	using Counter = int;
	static constexpr auto DataWidth = 8;

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
	 * @brief Returns length of the histogram
	 * @details
	 *
	 * @return length of the histogram
	 */
	static size_t length()
	{
		return (1 << DataWidth);
	}

	static size_t size()
	{
		return (sizeof(Counter) * length());
	}

	static size_t data_width()
	{
		return DataWidth;
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

__global__
void calculate_hist(
	const void* image, size_t pitch, size_t width, size_t height,
	int* histogram);

__host__
void calculate_hist(
	const CudaImage& image,
	CudaHistogram& histogram);

__global__
void calculate_cdf(
	const int* histogram,
	int* cdf,
	int* cdf_min);

__host__
void calculate_cdf(
	const CudaHistogram& histogram,
	CudaCDF& cdf,
	CudaCDF::Counter* cdf_min);

__global__
void generate_lut(
	int elems,
	const int* cdf,
	const int* cdf_min,
	unsigned char* lut);

__host__
void generate_lut(
	int elems,
	const CudaCDF& cdf,
	CudaCDF::Counter* cdf_min,
	CudaLUT& lut);

__global__
void apply_lut(
	const void* src, size_t spitch, size_t width, size_t height,
	const unsigned char* lut,
	void* dst, size_t dpitch);

__host__
void apply_lut(
	const CudaImage& src,
	const CudaLUT& lut,
	CudaImage& dst);

__host__
void equalize_hist(
	CudaImage& src,
	CudaHistogram& hist,
	CudaCDF& cdf,
	CudaCDF::Counter* cdf_min,
	CudaLUT& lut,
	CudaImage& dst);

/**
 * @brief Performs histogram equalization of source image
 * @details
 *
 * @param src source image
 * @param dst destination image
 */
void equalize_hist(const Image& src, Image& dst);
