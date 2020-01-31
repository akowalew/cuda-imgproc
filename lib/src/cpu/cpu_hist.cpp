///////////////////////////////////////////////////////////////////////////////
// cpu_hist.cpp
//
// Contains implementation of CPU histogram equalizer
///////////////////////////////////////////////////////////////////////////////

#include "cpu_hist.hpp"

#include <cassert>

#include "log.hpp"

Histogram cpu_calculate_hist(const Image& src)
{
	// Initialize histogram with zeros (!)
	Histogram histogram = {0};

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

CDF cpu_calculate_cdf(const Histogram& histogram)
{
	CDF cdf;

	// static_assert(cdf.size() == histogram.size(),
		// "Size of the histogram and the CDF should be equal");

	// We are going to accumulate histogram values in next iterations
	CDF::value_type accumulator = 0;

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

CDF::value_type cpu_find_cdf_min(const CDF& cdf)
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
	// Which means basically an ERROR, because histogram must be non-zero
	assert("This should not happen");
}

LUT cpu_generate_lut(const CDF& cdf, CDF::value_type cdf_min)
{
	LUT lut;

	// static_assert(cdf.size() == lut.size(),
		// "Size of CDF and size of LUT must be equal");

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

void cpu_apply_lut(const Image& src, Image& dst, const LUT& lut)
{
	// Source and destination images must have same size and types
	assert(src.rows == dst.rows);
	assert(src.cols == dst.cols);

	// Store local copy of image size
	const auto rows = src.rows;
	const auto cols = src.cols;
	const auto elems = (rows * cols);

	const auto src_data = src.data;
	const auto dst_data = dst.data;

	// Apply LUT on the source image
	#pragma omp parallel for
	for(auto i = 0; i < rows; ++i)
	{
		auto src_it_i = (src_data + i * cols);
		auto dst_it_i = (dst_data + i * cols);

		for(auto j = 0; j < cols; ++j)
		{
			const auto src_value = src_it_i[j];
			const auto dst_value = lut[src_value];

			dst_it_i[j] = dst_value;
		}
	}
}

void cpu_equalize_hist(Image& dst, const Image& src)
{
	// Source and destination images must have same size and types
	assert(src.rows == dst.rows);
	assert(src.cols == dst.cols);
	assert(src.type() == dst.type());

	const auto histogram = cpu_calculate_hist(src);
	const auto cdf = cpu_calculate_cdf(histogram);

	// The last element od CDF should be equal to total number of elements
	assert(cdf.back() == (src.rows * src.cols));

	const auto cdf_min = cpu_find_cdf_min(cdf);
	const auto lut = cpu_generate_lut(cdf, cdf_min);
	cpu_apply_lut(src, dst, lut);
}

Image cpu_equalize_hist(const Image& src)
{
	const auto cols = src.cols;
	const auto rows = src.rows;

	LOG_INFO("Equalizing histogram on CPU image %dx%d\n", cols, rows);

    auto dst = Image(rows, cols);

    cpu_equalize_hist(dst, src);

    return dst;
}