///////////////////////////////////////////////////////////////////////////////
// median_cuda_v1.cpp
//
// Contains definitions of functions related to median image filtering
// CUDA  implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 22.12.2019 16:19 CEST
///////////////////////////////////////////////////////////////////////////////

#include "median_cuda_v1.cuh"

#include <cassert>

#include <opencv2/imgproc.hpp>

void median(const Image& src, Image& dst, int kernel_size)
{
	cv::medianBlur(src, dst, kernel_size);
}
