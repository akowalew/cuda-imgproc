///////////////////////////////////////////////////////////////////////////////
// hist.hpp
//
// Contains declarations of functions working on images histograms
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 20:24 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <limits>

#include "image.hpp"

#include "hist_types.hpp"

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

