///////////////////////////////////////////////////////////////////////////////
// image.cu
//
// Contains definitions of stuff related to images for CUDA
///////////////////////////////////////////////////////////////////////////////

#include "image.hpp"

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

void CudaImage::copy_from_host(const Image& img)
{
    // Both images must have same size
    assert(img.cols == width);
    assert(img.rows == height);

    // Pitch of host image must be zero
    assert(img.isContinuous());
    const auto img_pitch = 0;

    // Perform copy of image from host to device
    checkCudaErrors(cudaMemcpy2D(data, pitch,
        img.data, img_pitch, width, height, cudaMemcpyHostToDevice));
}

void CudaImage::copy_to_host(Image& img)
{
    // Both images must have same size
    assert(img.cols == width);
    assert(img.rows == height);

    // Pitch of host image must be zero
    assert(img.isContinuous());
    const auto img_pitch = 0;

    // Perform copy of image from device to host
    checkCudaErrors(cudaMemcpy2D(img.data, img_pitch,
        data, pitch, width, height, cudaMemcpyDeviceToHost));
}
