///////////////////////////////////////////////////////////////////////////////
// filter.hpp
//
// Contains declarations of image filtering functions
///////////////////////////////////////////////////////////////////////////////

#pragma once

//
// Forward declarations
//

struct Image;
struct Kernel;

/**
 * @brief Sets given kernel for next convolution filter operation 
 * @details 
 * 
 * @param kernel kernel to use
 */
void bind_filter_kernel(const Kernel* kernel);

/**
 * @brief Applies convolution filter to an image
 * @details 
 * 
 * @param dst destination image
 * @param src source image
 * @param kernel convolution kernel
 */
void filter(Image* dst, const Image* src);

void filter(Image* dst, const Image* src, const Kernel* kernel);

Image* filter(const Image* src, const Kernel* kernel);
