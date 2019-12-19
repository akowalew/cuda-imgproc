///////////////////////////////////////////////////////////////////////////////
// hist_cuda.hpp
//
// Contains declarations of functions working on images histograms
// CUDA implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 17:04 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <limits>

#include "image.hpp"

//! Helper typedef - defines Histogram container for image with given depth
template<typename T>
using Histogram = std::array<
	int, //! Internal type of values counter
	(int)std::numeric_limits<T>::max() + 1 //! Number of elements - image depth
>;

//! Helper typedef - definition of Histogram for 8-bit images
using HistogramU8 = Histogram<unsigned char>;

/**
 * @brief Calculates histogram for given image
 * @details Histogram is calculated by iterating over each pixel in source image
 * and counting presence of each value.
 *
 * @param src source image
 * @param histogram result histogram
 */
void calculate_hist_8(const GrayImageU8& src, HistogramU8& histogram);

/**
 * @brief Performs histogram equalization of source image
 * @details
 *
 * @param src source image
 * @param dst destination image
 */
void equalize_hist_8(const GrayImageU8& src, GrayImageU8& dst);

