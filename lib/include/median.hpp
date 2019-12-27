///////////////////////////////////////////////////////////////////////////////
// median.hpp
//
// Contains declarations of functions related to median image filtering
// Generic implementation 
// 
// Author: akowalew (ram.techen@gmail.com)
// Date: 28.11.2019 23:22 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

/**
 * @brief Performs median filtering on image
 * @details
 *
 * @param src source image
 * @param dst destination image
 * @param kernel_size size of squared median filte kernel
 */
void median(const Image& src, Image& dst, int kernel_size);
