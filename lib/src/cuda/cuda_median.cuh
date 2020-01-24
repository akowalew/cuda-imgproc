///////////////////////////////////////////////////////////////////////////////
// cuda_median.cuh
//
// Contains declarations for CUDA median filterer
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_image.cuh"

//
// Public declarations
//

using CudaMedianKernelSize = size_t;

/**
 * @brief Applies median filter to an image
 * @details 
 * 
 * @param src source device image
 * @param ksize kernel size
 * 
 * @return medianed image
 */
CudaImage cuda_median(const CudaImage& src, CudaMedianKernelSize ksize);