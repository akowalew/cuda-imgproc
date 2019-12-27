///////////////////////////////////////////////////////////////////////////////
// median_cpu_multi.cpp
//
// Contains definitions of functions related to median image filtering
// Multi CPU implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 27.12.2019 16:24 CEST
///////////////////////////////////////////////////////////////////////////////

#include "median_cpu_multi.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

void median(const Image& src, Image& dst, int kernel_size)
{
	cv::medianBlur(src, dst, kernel_size);
}
