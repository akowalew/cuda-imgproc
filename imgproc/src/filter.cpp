///////////////////////////////////////////////////////////////////////////////
// filter.cpp
//
// Contains definitions of functions related to image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#include "imgproc/filter.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

namespace imgproc {

void filter2d_8(const unsigned char* src, unsigned char* dst,
	int ncols, int nrows,
	const float* kernel, int kernel_size)
{
	assert(src != nullptr);
	assert(dst != nullptr);

	const auto type = CV_8UC1;
	const auto src_mat = cv::Mat(nrows, ncols, type, const_cast<unsigned char*>(src));
	auto dst_mat = cv::Mat(nrows, ncols, type, dst);

	const auto kernel_type = CV_32F;
	const auto kernel_mat = cv::Mat(kernel_size, kernel_size, kernel_type, const_cast<float*>(kernel));

	const auto ddepth = -1; // Keep depth in destination same as in source
	cv::filter2D(src_mat, dst_mat, ddepth, kernel_mat);
}

} // namespace imgproc
