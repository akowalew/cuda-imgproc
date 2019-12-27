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
void filter(const Image& src, Image& dst, const Image& kernel);
