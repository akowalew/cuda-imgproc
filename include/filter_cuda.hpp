///////////////////////////////////////////////////////////////////////////////
// filter_cuda.hpp
//
// Contains declarations of functions related to image filtering
// CUDA implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.11.2019 19:16 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

/**
 * @brief Performs convolution filtering on given image
 * @details
 *
 * @param src source image
 * @param dst destination image
 * @param kernel kernel of the filter (squared)
 */
void cuda_filter2d_8(const Image& src, Image& dst, const Image& kernel);
