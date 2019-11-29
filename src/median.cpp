///////////////////////////////////////////////////////////////////////////////
// median.cpp
//
// Contains definitions of functions related to median image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 28.11.2019 23:23 CEST
///////////////////////////////////////////////////////////////////////////////

#include "median.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

void median2d_8(const Image& src, Image& dst, int kernel_size)
{
	cv::medianBlur(src, dst, kernel_size);
}
