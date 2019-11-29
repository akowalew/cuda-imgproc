///////////////////////////////////////////////////////////////////////////////
// filter.cpp
//
// Contains definitions of functions related to image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#include "filter.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

void filter2d_8(const Image& src, Image& dst, const Image& kernel)
{
	const auto ddepth = -1; // Keep depth in destination same as in source
	cv::filter2D(src, dst, ddepth, kernel);
}
