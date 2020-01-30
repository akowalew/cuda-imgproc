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
void cuda_median_async(CudaImage& dst, const CudaImage& src, CudaMedianKernelSize ksize);
