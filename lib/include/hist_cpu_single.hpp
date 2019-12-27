///////////////////////////////////////////////////////////////////////////////
// hist_single.hpp
//
// Contains declarations of functions working on images histograms
// Single CPU implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 22.12.2019 15:40 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <limits>

#include "image.hpp"

//! Helper typedef - definition of Histogram for 8-bit images
using Histogram = std::array<
	int, //! Internal type of values counter
	(int)std::numeric_limits<unsigned char>::max() + 1 //! Number of elements - image depth
>;

//! Helper typedef - cumulative distribution function of 8-bit image
using CDF = Histogram;

//! Helper typedef - lookup table with 8-bit 256 values
using LUT = std::array<unsigned char, 256>;

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
 * @brief Calculates histogram of given image
 * @details
 *
 * @param src source image
 * @return calculated histogram
 */
Histogram calculate_hist(const Image& src);

/**
 * @brief Calculates cumulative distribution function based on image histogram
 * @details
 *
 * @param histogram image histogram
 * @return cumulative distribution function
 */
CDF calculate_cdf(const Histogram& histogram);

/**
 * @brief Finds first non-zero minima of the CDF
 * @details
 *
 * @param cdf cumulative distribution function
 * @return
 */
CDF::value_type find_cdf_min(const CDF& cdf);

/**
 * @brief Generates lookup table based on cumulative distribution function (CDF)
 * @details
 *
 * @param cdf cumulative distribution function
 * @param cdf_min non-zero minima of the CDF
 *
 * @return generated lookup table
 */
LUT generate_lut(const CDF& cdf, CDF::value_type cdf_min);

/**
 * @brief Applies LookUpTable into the source image and stores result
 * @details For each pixel in the source image it performs following operation:
 *  value = lut[src[i]]
 *  dst[i] = value
 *
 * @param src source image
 * @param dst destination image
 * @param lut LookUpTable
 */
void apply_lut(const Image& src, Image& dst, const LUT& lut);

/**
 * @brief Performs histogram equalization of source image
 * @details
 *
 * @param src source image
 * @param dst destination image
 */
void equalize_hist(const Image& src, Image& dst);
