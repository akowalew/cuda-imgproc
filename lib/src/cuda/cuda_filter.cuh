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

void cuda_set_filter_kernel_async(const CudaKernel& kernel);

/**
 * @brief Applies mean-blurr convolution filter to an image
 * @details 
 * 
 * @param src source image
 * @param ksize size of the kernel
 * @return filtered image
 */
void cuda_filter_async(CudaImage& dst, const CudaImage& src);
