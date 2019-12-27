///////////////////////////////////////////////////////////////////////////////
// hist_ref.hpp
//
// Contains declarations of functions working on images histograms
// Reference implementation
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 22.12.2019 16:13 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

/**
 * @brief Performs histogram equalization of source image
 * @details
 *
 * @param src source image
 * @param dst destination image
 */
void equalize_hist(const Image& src, Image& dst);

