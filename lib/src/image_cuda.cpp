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

CudaImage create_cuda_image(size_t width, size_t height)
{
    void* data;
    size_t pitch;
    checkCudaErrors(cudaMallocPitch(&data, &pitch, width, height));

    return CudaImage{data, pitch, width, height};	
}

void free_cuda_image(const CudaImage& image)
{
    checkCudaErrors(cudaFree(image.data));
}

void fill_cuda_image(const CudaImage& image, int value)
{
	checkCudaErrors(cudaMemset2D(image.data, image.pitch, value, image.width, image.height));
}