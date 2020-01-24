///////////////////////////////////////////////////////////////////////////////
// cuda_filter.cuh
//
// Contains declarations for CUDA convolution filterer
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_image.cuh"

//
// Public declarations
//

using CudaFilterKernelSize = size_t;

/**
 * @brief Applies mean-blurr convolution filter to an image
 * @details 
 * 
 * @param src source image
 * @param ksize size of the kernel
 * @return filtered image
 */
CudaImage cuda_filter(const CudaImage& src, CudaFilterKernelSize ksize);
