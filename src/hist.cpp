///////////////////////////////////////////////////////////////////////////////
// hist.cpp
//
// Contains definitions of functions working on images histograms
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 20:28 CEST
///////////////////////////////////////////////////////////////////////////////

#include "hist.hpp"

#include <cassert>
#include <cstdio>

#include <array>
#include <limits>

void equalize_hist_8(const GrayImageU8& src, GrayImageU8& dst)
{
	assert(src.rows == dst.rows);
	assert(src.cols == dst.cols);

	constexpr auto MaxValue = (int)std::numeric_limits<unsigned char>::max();
	constexpr auto HistogramSize = (MaxValue + 1);

	// We need helper variable in order to count values
	// Initialize it with zeros
	std::array<int, HistogramSize> histogram = {0};

	// Calculate histogram
	const auto elems = (src.rows * src.cols);
	for(auto i = 0; i < elems; ++i)
	{
		// Get value (brightness) of current pixel
		const unsigned char value = src.data[i];

		// Increase this value's counter
		++histogram[value];
	}

	// We need another variable, as big as our histogram
	// It represents cumulative distribution function
	std::array<int, HistogramSize> cdf;

	// Perform calculation of cumulative distribution function
	int accumulator = 0;
	for(auto i = 0; i < HistogramSize; ++i)
	{
		// Get i-th value counter
		const int hist_value = histogram[i];

		// Increase cumulative counter
		accumulator += hist_value;

		// Store current value of the accumulator
		cdf[i] = accumulator;
	}

	// The last element should be equal to total number of elements
	assert(*std::prev(cdf.end()) == elems);

	// Find first, non-null cdf value - minimal one
	const int cdf_min = [&cdf]() {
		int cdf_value = cdf[0];
		for(auto i = 0; i < HistogramSize; ++i)
		{
			cdf_value = cdf[i];
			if(cdf_value != 0)
			{
				break;
			}
		}

		return cdf_value;
	}();

	// We need another array, also in size of histogram, but now
	// for acting like LookUpTable
	std::array<unsigned char, HistogramSize> lut;

	// Generate lookup table
	for(auto i = 0; i < HistogramSize; ++i)
	{
		const int cdf_value = cdf[i];
		const unsigned char lut_value =
			((cdf_value - cdf_min) * MaxValue) / (elems - cdf_min);
		lut[i] = lut_value;
	}

	// Apply LUT on the source image
	for(auto i = 0; i < elems; ++i)
	{
		const auto src_value = src.data[i];

		const auto dst_value = lut[src_value];

		dst.data[i] = dst_value;
	}
}
