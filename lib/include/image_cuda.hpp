///////////////////////////////////////////////////////////////////////////////
// image_cuda.hpp
//
// Contains declarations of image types classes for CUDA
//
// Author: akowalew (ram.techen@gmail.com)
// Date: 19.12.2019 16:51 CEST
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>

/**
 * @brief Represents image to be used in CUDA devices, 8-bit depth
 * @details 
 * 
 */
struct CudaImage
{
    void* data;
    size_t pitch;
    size_t width;
    size_t height;

    constexpr size_t size() const
    {
    	return (pitch * height);
    }

    constexpr size_t elems() const
    {
    	return (width * height);
    }
};

CudaImage create_cuda_image(size_t width, size_t height);

void free_cuda_image(const CudaImage& image);

void fill_cuda_image(const CudaImage& image, int value);
