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

void cuda_filter_init();

void cuda_filter_deinit();

void cuda_filter_copy_kernel_from_host_async(const Kernel& kernel);

/**
 * @brief Applies mean-blurr convolution filter to an image
 * @details 
 * 
 * @param dst destination image
 * @param src source image
 * @param ksize size of the kernel
 */
void cuda_filter_async(CudaImage& dst, const CudaImage& src, size_t ksize);
