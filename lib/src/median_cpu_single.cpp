///////////////////////////////////////////////////////////////////////////////
// median_cpu_single.cpp
//
// Contains definitions of functions related to median image filtering
// Single CPU implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 27.12.2019 16:17 CEST
///////////////////////////////////////////////////////////////////////////////

#include "median_cpu_single.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

void median(const Image& src, Image& dst, int kernel_size)
{
	cv::medianBlur(src, dst, kernel_size);
}
