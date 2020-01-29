///////////////////////////////////////////////////////////////////////////////
// cpu_median.hpp
//
// Contains declarations for CPU median filterer
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

//
// Public declarations
//

void cpu_median(Image& dst, const Image& src, int ksize);

/**
 * @brief Applies median filter to an image
 * @details 
 * 
 * @param src source image
 * @param ksize size of the kernel
 * @return filtered image
 */
Image cpu_median(const Image& src, int ksize);
