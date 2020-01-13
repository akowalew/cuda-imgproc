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

    /**
     * @brief Constructor
     * @details Creates image on the device with given size
     *
     * @param width width of the image
     * @param height height of the image
     */
    CudaImage(size_t width, size_t height);

    /**
     * @brief Destructor
     * @details Releases image from the device
     */
    ~CudaImage();

    /**
     * @brief Fills image with given value
     * @details
     *
     * @param value value with which to fill
     */
    void fill(int value);

    /**
     * @brief Returns size in bytes of the image
     * @details
     *
     * @return size in bytes of the image
     */
    size_t size() const
    {
    	return (pitch * height * sizeof(char));
    }

    /**
     * @brief Returns number of elements in the image
     * @details
     *
     * @return number of elements in the image
     */
    size_t elems() const
    {
    	return (width * height);
    }
};
