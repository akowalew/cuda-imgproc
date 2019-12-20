///////////////////////////////////////////////////////////////////////////////
// hist_cuda1.hpp
//
// Contains declarations of functions working on images histograms
// CUDA1 implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 17:04 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <limits>

#include "cuda_image.hpp"

#include "hist_types.hpp"

/**
 * @brief Represents histogram with given counter type and analyzed data width
 * @details 
 * 
 * @tparam TCounter Type of the values' counter
 * @tparam DataWidth Width of analyzed data. Determines length of histogram
 */
template<typename TCounter, size_t DataWidth>
struct CudaHistogram
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
using CudaHistogramI8 = CudaHistogram<int, 8>;

CudaHistogramI8 create_cuda_histogram();

CudaHistogramI8 create_host_histogram();

void free_cuda_histogram(const CudaHistogramI8& histogram);

void free_host_histogram(const CudaHistogramI8& histogram);

void copy_cuda_histogram_to_host(const CudaHistogramI8& d_histogram, 
	const CudaHistogramI8& h_histogram);

/**
 * @brief Calculates histogram for given image
 * @details Histogram is calculated by iterating over each pixel in source image
 * and counting presence of each value.
 *
 * @param src source image
 * @param histogram result histogram
 */
void calculate_hist_8_cuda1(const CudaImage& image, CudaHistogramI8& histogram);

/**
 * @brief Performs histogram equalization of source image
 * @details
 *
 * @param src source image
 * @param dst destination image
 */
void equalize_hist_8_cuda1(const CudaImage& src, CudaImage& dst);

