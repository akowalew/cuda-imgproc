///////////////////////////////////////////////////////////////////////////////
// filter.hpp
//
// Contains declarations of functions related to image filtering
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 17:19 CEST
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
void filter2d_8(const Image& src, Image& dst, const Image& kernel);

/**
 * @brief Implementation of convolution filter
 * @details
 *
 * @param src source image
 * @param dst destination image
 * @param kernel kernel of the filter (squared)
 */
void _filter2d_8(const Image& src, Image& dst, const Image& kernel);
