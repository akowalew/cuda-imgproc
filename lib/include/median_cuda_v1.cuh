///////////////////////////////////////////////////////////////////////////////
// median_cuda_v1.cuh
//
// Contains declarations of functions related to median image filtering
// CUDA v1 implementation 
// 
// Author: akowalew (ram.techen@gmail.com)
// Date: 27.12.2019 20:19 CEST
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
