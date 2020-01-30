///////////////////////////////////////////////////////////////////////////////
// proc.hpp
// 
// Contains declarations for image processor module
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "image.hpp"
#include "kernel.hpp"

/**
 * @brief Initializes image processor module
 * @details It queries for available compute devices and selects one of them
 * to use in calculations. 
 */
void proc_init();

/**
 * @brief Deinitializes image processor module
 * @details It freeds allocated memories and releases acquired device
 */
void proc_deinit();

/**
 * @brief Processes the image according to specified configuration
 * @details Image is processed in following steps:
 *  1) Median filtration
 *  2) Convolution filtration
 *  3) Histogram equalization
 * 
 * @param img image to process
 * @param filter_kernel convolution filter kernel
 * @param median_ksize size of kernel for median filtering
 * 
 * @return processing result
 */
Image process_image(const Image& img, const Kernel& filter_kernel, size_t median_ksize);