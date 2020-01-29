///////////////////////////////////////////////////////////////////////////////
// cpu_filter.hpp
//
// Contains declarations for CPU convolution filterer
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

//
// Public declarations
//

void cpu_filter(Image& dst, const Image& src, const cv::Mat_<float>& kernel);

/**
 * @brief Applies convolution filter to an image
 * @details 
 * 
 * @param src source image
 * @param kernel convolution kernel
 * @return filtered image
 */
Image cpu_filter(const Image& src, const cv::Mat_<float>& kernel);
