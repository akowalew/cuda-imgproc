///////////////////////////////////////////////////////////////////////////////
// processor.hpp
//
// Contains declaration of image processor functions
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 29.11.2019 1:17 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

/**
 * @brief Executes processing pipeline on given image
 * @details Processing algorithm consists of five steps:
 * 1) Splitting BGR image into B, G and R components
 * 2) Applying median filter on each component separately
 * 3) Applying Gaussian blur on each component separately
 * 4) Equalizing histograms of each component separately
 * 5) Merging B, G and R components into final image
 *
 * @param image image to be processed
 * @return image after processing
 */
Image process_image(Image image);
