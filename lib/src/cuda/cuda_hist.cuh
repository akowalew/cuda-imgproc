///////////////////////////////////////////////////////////////////////////////
// cuda_hist.cuh
//
// Contains declarations for CUDA histograms equalizer
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_image.cuh"

//
// Public declarations
//

/**
 * @brief Performs histogram equalization
 * @details 
 * 
 * @param src source image
 * @return image with equalized image
 */
CudaImage cuda_equalize_hist(const CudaImage& src);