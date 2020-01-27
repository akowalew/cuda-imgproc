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

void cuda_median_async(CudaImage& dst, const CudaImage& src, CudaMedianKernelSize ksize);

void cuda_median(CudaImage& dst, const CudaImage& src, CudaMedianKernelSize ksize);

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