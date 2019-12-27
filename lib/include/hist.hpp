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
 * @brief Initializes hist module
 * @details 
 */
void hist_init();

/**
 * @brief Deinitializes hist module
 * @details 
 */
void hist_deinit();

/**
 * @brief Performs histogram equalization of source image
 * @details
 *
 * @param src source image
 * @param dst destination image
 */
void equalize_hist(const Image& src, Image& dst);

