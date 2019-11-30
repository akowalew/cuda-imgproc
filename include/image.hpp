///////////////////////////////////////////////////////////////////////////////
// image.hpp
//
// Contains declarations of image types classes
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 28.11.2019 23:26 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <opencv2/core/mat.hpp>

using Image = cv::Mat;

//! Image template instantiation - 8-bit image
using ImageU8 = cv::Mat_<unsigned char>;

//! Image template instantiation - 32-bit float image
using Image32F = cv::Mat_<float>;

//! Image template instantiation - 8-bit mono-color image
using GrayImageU8 = cv::Mat;
