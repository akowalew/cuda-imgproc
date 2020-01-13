///////////////////////////////////////////////////////////////////////////////
// image_cuda.cpp
//
// Contains definitions of stuff related to images for CUDA
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 20.12.2019 1:10 CEST
///////////////////////////////////////////////////////////////////////////////

#include "image_cuda.hpp"

#include <cassert>

#include <cuda_runtime.h>

#include <helper_cuda.h>

CudaImage::CudaImage(size_t width, size_t height)
    :   width(width)
    ,   height(height)
{
    // Allocate memory for 2D image
    checkCudaErrors(cudaMallocPitch(&data, &pitch, width, height));
}

CudaImage::~CudaImage()
{
    if(data)
    {
        // Free memory of 2D image
        checkCudaErrors(cudaFree(data));
    }
}

void CudaImage::fill(int value)
{
    // Fill 2D image with value
	checkCudaErrors(cudaMemset2D(data, pitch, value, width, height));
}
