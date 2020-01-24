///////////////////////////////////////////////////////////////////////////////
// proc.hpp
// 
// Contains declarations for image processor module
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"

/**
 * @brief Initializes image processor module
 * @details It queries for available compute devices and selects one of them
 * to use in calculations. 
 */
void init();

/**
 * @brief Deinitializes image processor module
 * @details It freeds allocated memories and releases acquired device
 */
void deinit();

struct MedianKernelSize
{
	size_t value;
	constexpr operator size_t() const noexcept { return value; }
};

struct FilterKernelSize
{
	size_t value;
	constexpr operator size_t() const noexcept { return value; }
};

/**
 * @brief Image processing configuration
 * @details It describes the parameters used for image processing
 */
struct ProcessConfig
{
	MedianKernelSize median_ksize {3}; //! Size of median filter kernel
	FilterKernelSize filter_ksize {3}; //! Size of convolution filter
};

/**
 * @brief Processes the image according to specified configuration
 * @details Image is processed in following steps:
 *  1) Median filtration
 *  2) Convolution filtration
 *  3) Histogram equalization
 * 
 * @param img image to process
 * @param config processing configuration
 * 
 * @return processing result
 */
Image process_image(const Image& img, const ProcessConfig& config);