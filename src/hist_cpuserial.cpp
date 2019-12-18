///////////////////////////////////////////////////////////////////////////////
// hist.cpp
//
// Contains definitions of functions working on images histograms
// CPU serial version
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 20:28 CEST
///////////////////////////////////////////////////////////////////////////////

#include "hist.hpp"

#include <cassert>
#include <cstdio>

#include <array>
#include <limits>

//! Helper typedef - cumulative distribution function of image with given type
template<typename T>
using CDF = Histogram<T>;

//! Helper typedef - cumulative distribution function of 8-bit image
using CDFU8 = CDF<unsigned char>;

//! Helper typedef - lookup table for given type and specified number of values
template<typename T, size_t Size>
using LUT = std::array<T, Size>;

//! Helper typedef - lookup table with 8-bit 256 values
using LUTU8 = LUT<unsigned char, 256>;

/**
 * @brief Calculates histogram of given image
 * @details
 *
 * @param src source image
 * @return calculated histogram
 */
HistogramU8 calculate_hist_8(const GrayImageU8& src)
{
	// Initialize histogram with zeros (!)
	HistogramU8 histogram = {0};

	// Calculate histogram
	const auto elems = (src.rows * src.cols);
	for(auto i = 0; i < elems; ++i)
	{
		// Get value (brightness) of current pixel
		const unsigned char value = src.data[i];

		// Increase this value's counter
		++histogram[value];
	}

	return histogram;
}

/**
 * @brief Calculates cumulative distribution function based on image histogram
 * @details
 *
 * @param histogram image histogram
 * @return cumulative distribution function
 */
CDFU8 calculate_cdf(const HistogramU8& histogram)
{
	CDFU8 cdf;

	static_assert(cdf.size() == histogram.size(),
		"Size of the histogram and the CDF should be equal");

	// We are going to accumulate histogram values in next iterations
	CDFU8::value_type accumulator = 0;

	// Calculate cumulative distribution function
	for(auto i = 0; i < histogram.size(); ++i)
	{
		// Get i-th value counter
		const int hist_value = histogram[i];

		// Increase cumulative counter
		accumulator += hist_value;

		// Store current value of the accumulator
		cdf[i] = accumulator;
	}

	// At the end, from well defined histogram, accumulator should be non-zero
	// Because histogram must be non-zeroed
	assert(accumulator > 0);

	return cdf;
}

/**
 * @brief Finds first non-zero minima of the CDF
 * @details
 *
 * @param cdf cumulative distribution function
 * @return
 */
CDFU8::value_type find_cdf_min(const CDFU8& cdf)
{
	// Find first, non-null cdf value - minimal one
	for(const auto cdf_value : cdf)
	{
		if(cdf_value != 0)
		{
			return cdf_value;
		}
	}

	// If we are here, cdf is zeroed, what means that histogram is zeroed
	// Which means basically an ERROR!
	assert("This should not happen");
}

/**
 * @brief Generates lookup table based on cumulative distribution function (CDF)
 * @details
 *
 * @param cdf cumulative distribution function
 * @param cdf_min non-zero minima of the CDF
 *
 * @return generated lookup table
 */
LUTU8 generate_lut(const CDFU8& cdf, CDFU8::value_type cdf_min)
{
	LUTU8 lut;

	static_assert(cdf.size() == lut.size(),
		"Size of CDF and size of LUT must be equal");

	// Number of elements is equal to the last element of the CDF
	const auto elems = cdf.back();

	// Maximum value of an 8-bit number
	constexpr int MaxValue = 255;

	// Store copy of CDF data end pointer
	const auto cdf_end = cdf.end();

	// Generate lookup table
	auto cdf_it = cdf.begin();
	auto lut_it = lut.begin();
	for(; cdf_it != cdf_end; ++cdf_it, ++lut_it)
	{
		const auto cdf_value = *cdf_it;
		const auto cdf_diff = (cdf_value - cdf_min);
		const auto lut_value = ((cdf_diff * MaxValue) / (elems - cdf_min));

		*lut_it = lut_value;
	}

	return lut;
}

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
void apply_lut(const GrayImageU8& src, GrayImageU8& dst, const LUTU8& lut)
{
	// Source and destination images must have same size and types
	assert(src.rows == dst.rows);
	assert(src.cols == dst.cols);

	// Store local copy of image size
	const auto rows = src.rows;
	const auto cols = src.cols;

	const auto src_end = src.dataend;

	// Apply LUT on the source image
	auto src_it = src.data;
	auto dst_it = dst.data;
	for(; src_it != src_end; ++src_it, ++dst_it)
	{
		const auto src_value = *src_it;
		const auto dst_value = lut[src_value];

		*dst_it = dst_value;
	}
}

void equalize_hist_8(const GrayImageU8& src, GrayImageU8& dst)
{
	// Source and destination images must have same size and types
	assert(src.rows == dst.rows);
	assert(src.cols == dst.cols);
	assert(src.type() == dst.type());

	const auto histogram = calculate_hist_8(src);
	const auto cdf = calculate_cdf(histogram);

	// The last element od CDF should be equal to total number of elements
	assert(cdf.back() == (src.rows * src.cols));

	const auto cdf_min = find_cdf_min(cdf);
	const auto lut = generate_lut(cdf, cdf_min);
	apply_lut(src, dst, lut);
}
