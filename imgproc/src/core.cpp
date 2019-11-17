///////////////////////////////////////////////////////////////////////////////
// core.cpp
//
// Contains definitions of core image processing functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#include "imgproc/core.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

namespace imgproc {

void split_888(int nrows, int ncols,
	const unsigned char* src,
	unsigned char* x, unsigned char* y, unsigned char* z)
{
	assert(nrows >= 0);
	assert(ncols >= 0);
	assert(src != nullptr);
	assert(x != nullptr);
	assert(y != nullptr);
	assert(z != nullptr);

	const auto src_mat = cv::Mat(nrows, ncols, CV_8UC3, const_cast<unsigned char*>(src));

    auto dst_mats = std::vector<cv::Mat>{
		cv::Mat(nrows, ncols, CV_8UC1, x),
		cv::Mat(nrows, ncols, CV_8UC1, y),
		cv::Mat(nrows, ncols, CV_8UC1, z),
	};

    cv::split(src_mat, dst_mats);
}

void merge_888(int nrows, int ncols,
	const unsigned char* src_x, const unsigned char* src_y, const unsigned char* src_z,
	unsigned char* dst)
{
	assert(nrows >= 0);
	assert(ncols >= 0);
	assert(src_x != nullptr);
	assert(src_y != nullptr);
	assert(src_z != nullptr);
	assert(dst != nullptr);

    const auto src_mats = std::vector<cv::Mat>{
		cv::Mat(nrows, ncols, CV_8UC1, const_cast<unsigned char*>(src_x)),
		cv::Mat(nrows, ncols, CV_8UC1, const_cast<unsigned char*>(src_y)),
		cv::Mat(nrows, ncols, CV_8UC1, const_cast<unsigned char*>(src_z)),
	};

	auto dst_mat = cv::Mat(nrows, ncols, CV_8UC3, dst);

	cv::merge(src_mats, dst_mat);
}

} // namespace imgproc
