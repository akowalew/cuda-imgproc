///////////////////////////////////////////////////////////////////////////////
// histogram.cpp
//
// Contains definitions of functions working on images histograms
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 20:28 CEST
///////////////////////////////////////////////////////////////////////////////

#include "imgproc/histogram.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

namespace imgproc {

void equalize_hist_8(const unsigned char* src, unsigned char* dst, int length)
{
	assert(src != nullptr);
	assert(dst != nullptr);
	assert(length >= 0);

	const auto src_array = cv::_InputArray(src, length);
	auto dst_array = cv::_OutputArray(dst, length);
	cv::equalizeHist(src_array, dst_array);
}

} // namespace imgproc
