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

#include <omp.h>

template<typename T>
using CDF = Histogram<T>;

using CDFU8 = CDF<unsigned char>;

void calculate_hist_8(const GrayImageU8& src, HistogramU8& histogram)
{
	// Calculate histogram
	const auto elems = (src.rows * src.cols);
	#pragma omp simd
	for(auto i = 0; i < elems; ++i)
	{
		// Get value (brightness) of current pixel
		const unsigned char value = src.data[i];

		// Increase this value's counter
		++histogram[value];
	}
}

void equalize_hist_8(const GrayImageU8& src, GrayImageU8& dst)
{
	assert(src.rows == dst.rows);
	assert(src.cols == dst.cols);

	const auto rows = src.rows;
	const auto cols = src.cols;
	const auto elems = (rows * cols);

	HistogramU8 histogram = {0};
	calculate_hist_8(src, histogram);

	// We need another variable, as big as our histogram
	// It represents cumulative distribution function
	CDFU8 cdf;
	int accumulator = 0;

	#pragma omp simd
	for(auto i = 0; i < histogram.size(); ++i)
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
		for(auto i = 0; i < cdf.size(); ++i)
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
	std::array<unsigned char, histogram.size()> lut;

	// Generate lookup table
	// #pragma omp parallel for simd schedule(dynamic, 32)
	for(auto i = 0; i < histogram.size(); ++i)
	{
		const int cdf_value = cdf[i];
		const unsigned char lut_value =
			((cdf_value - cdf_min) * 255) / (elems - cdf_min);

		lut[i] = lut_value;
	}

	// Apply LUT on the source image
	#pragma omp parallel for
	for(auto i = 0; i < rows; ++i)
	{
		for(auto j = 0; j < cols; ++j)
		{
			const auto src_value = src.data[i * cols + j];
			const auto dst_value = lut[src_value];

			dst.data[i * cols + j] = dst_value;
		}
	}
}
