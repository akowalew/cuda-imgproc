///////////////////////////////////////////////////////////////////////////////
// median_ref.cpp
//
// Contains definitions of functions related to median image filtering
// Reference implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 22.12.2019 16:19 CEST
///////////////////////////////////////////////////////////////////////////////

#include "median_ref.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

void median(const Image& src, Image& dst, int kernel_size)
{
	cv::medianBlur(src, dst, kernel_size);
}
