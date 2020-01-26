///////////////////////////////////////////////////////////////////////////////
// cuda_filter.cuh
//
// Contains declarations for CUDA convolution filterer
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cuda_image.cuh"
#include "cuda_kernel.cuh"

//
// Public declarations
//

void cuda_filter(CudaImage& dst, const CudaImage& src, size_t ksize);

/**
 * @brief Applies mean-blurr convolution filter to an image
 * @details 
 * 
 * @param src source image
 * @param ksize size of the kernel
 * @return filtered image
 */
CudaImage cuda_filter(const CudaImage& src, size_t ksize);
