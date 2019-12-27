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
template<typename TCounter, size_t DataWidth>
struct _CudaHistogram
{
	//
	// Helper typedefs
	//

	using Counter = TCounter;

	//
	// Public data
	//

	Counter* data;

	//
	// Public methods
	//

	constexpr static size_t length()
	{
		return (1 << DataWidth);
	}

	constexpr static size_t size()
	{
		return (sizeof(Counter) * length());
	}

	constexpr static size_t data_width()
	{
		return DataWidth;
	}

	constexpr Counter* begin()
	{
		return data;
	}

	constexpr Counter* end()
	{
		return (data + length());
	}
};

//! Helper typedef - CUDA histogram of 8-bit values with counter of type `int`
using CudaHistogram = _CudaHistogram<int, 8>;

using CudaCDF = CudaHistogram;

template<typename T>
struct _CudaLUT
{
	//
	// Helper typedefs
	//

	using Type = T;

	//
	// Public data
	//

	Type* data;

	//
	// Public methods
	//

	constexpr static size_t length()
	{
		return (1 + (size_t)std::numeric_limits<Type>::max());

	}

	constexpr static size_t size()
	{
		return (sizeof(Type) * length());
	}

	constexpr Type* begin()
	{
		return data;
	}

	constexpr Type* end()
	{
		return (data + length());
	}
};

using CudaLUT = _CudaLUT<unsigned char>;

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

/**
 * @brief Creates CUDA histogram
 * @details 
 * @return 
 */
CudaHistogram create_cuda_histogram();

/**
 * @brief Frees CUDA histogram
 * @details 
 * 
 * @param histogram 
 */
void free_cuda_histogram(const CudaHistogram& histogram);

/**
 * @brief Creates CUDA LookUpTable
 * @details 
 * @return 
 */
CudaLUT create_cuda_lut();

/**
 * @brief Frees CUDA LookUpTable
 * @details 
 * 
 * @param lut 
 */
void free_cuda_lut(const CudaLUT& lut);

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