///////////////////////////////////////////////////////////////////////////////
// core.cpp
//
// Contains definitions of core image processing functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 19:06 CEST
///////////////////////////////////////////////////////////////////////////////

#include "core.hpp"

#include <cassert>

#include <opencv2/imgproc.hpp>

void split_888(const Image& src, std::array<Image, 3>& dst)
{
	auto dst_array = std::vector<Image>{dst[0], dst[1], dst[2]};
    cv::split(src, dst_array);
}

void merge_888(const std::array<Image, 3>& src, Image& dst)
{
	auto src_array = std::vector<Image>{src[0], src[1], src[2]};
	cv::merge(src_array, dst);
}
