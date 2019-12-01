///////////////////////////////////////////////////////////////////////////////
// hist.hpp
//
// Contains declarations of functions working on images histograms
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 17.11.2019 20:24 CEST
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
void equalize_hist_8(const GrayImageU8& src, GrayImageU8& dst);

