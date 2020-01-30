///////////////////////////////////////////////////////////////////////////////
// cuda_proc.cuh
// 
// Contains declarations for CUDA image processor module
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "proc.hpp"

/**
 * @brief Initializes image processor module
 * @details It queries for available compute devices and selects one of them
 * to use in calculations. 
 */
void cuda_proc_init();

/**
 * @brief Deinitializes image processor module
 * @details It freeds allocated memories and releases acquired device
 */
void cuda_proc_deinit();

/**
 * @brief Processes the image according to specified configuration
 * @details Image is processed in following steps:
 * 	0) Provided image is copied to the device
 *  1) Median filtration
 *  2) Convolution filtration
 *  3) Histogram equalization
 *  4) Image is copied back to host
 * 
 * @param img image to process
 * @param config processing configuration
 * 
 * @return processing result
 */
Image cuda_process_image(const Image& src, const ProcessConfig& config);